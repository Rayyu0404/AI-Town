# ================================================================
# agent/agent_manager.py
# 多角色排程管理：兩段式時間軸設計
#
# 每個時間段分為兩個階段：
#   Phase 1 — 執行（execute_phase）：
#     執行上一輪 decide() 存下的 pending_action
#     - 對話行動：配對雙方後執行 N 輪對話
#     - 睡覺行動：觸發 STM→LTM 濃縮
#     - 其他行動：更新狀態即可
#
#   Phase 2 — 決策（decide_phase）：
#     所有角色針對下一時段做決策
#     - STM 路徑（簡單思考）：馬可夫鏈機率（K < 0.4 且 C < threshold）
#     - LTM 路徑（複雜思考）：模型自主推理（K ≥ 0.4 或 C ≥ threshold）
#
#   同步點：所有角色決策完成後推進時鐘
#
# 對話機制：
#   - _forward_dialogue(sender, receiver, msg, scene)：
#       讓 sender 把訊息傳遞給 receiver，receiver 立刻 step() 並快取回應
#   - step_character() 中若命中快取，直接回傳 receiver 的快取回應
#   - 沒有對話對象（獨處）時，自動 override 為「休息」防止對空氣說話
#   - pop_dialogue_history()：取出並清空本時間段配對完成的對話記錄
#   - clear_conversation_cache()：每時段開始前清除跨段殘留快取
# ================================================================

from PIL import Image
from typing import Optional

from core.character import Character
from core.stm import STM
from core.ltm import LTM
from agent.agent import Agent
from world.world_clock import WorldClock
from utils.file_io import load_all_characters, save_character
from utils.logger import get_logger, log_turn, log_consolidation
from config.world_config import (CHARACTER_NAMES, CHARACTER_CODES,
                                  MAX_CONVERSATION_TURNS_PER_SLOT)

logger = get_logger("agent_manager")


class AgentManager:
    """
    管理全部角色的 Agent 實例，協調兩段式時間軸。
    loader : ModelLoader（已呼叫 load() 的實例，所有 Agent 共用）
    clock  : WorldClock 實例
    """

    def __init__(self, loader, clock: WorldClock):
        self.loader = loader
        self.clock  = clock

        raw_data = load_all_characters()

        self._characters: dict = {}
        self._stms:       dict = {}
        self._ltms:       dict = {}
        self._agents:     dict = {}
        # 對話快取：A 呼叫 _forward_dialogue 後，B 的回應暫存於此；
        # B 自己的 step_character 呼叫時直接回傳快取，確保雙方顯示同一輪對話
        self._conv_result_cache: dict = {}
        # 對話配對記錄：傳送方資訊暫存；接收方消費 cache 時補上回應，完成完整配對
        self._pending_pair:   dict = {}   # receiver_code → partial pair dict
        self._dialogue_history: list = [] # 已完成的配對（每個時間段 pop 一次）
        # 今天已入睡的角色代號集合（每天重置）
        self._sleeping_today: set = set()

        for code, data in raw_data.items():
            char = Character(data)
            stm  = STM(data)
            ltm  = LTM(data)
            self._characters[code] = char
            self._stms[code]       = stm
            self._ltms[code]       = ltm
            self._agents[code]     = Agent(char, stm, ltm, loader)

        logger.info(f"AgentManager 初始化，角色：{list(self._agents.keys())}")

    # ================================================================
    # 對外主要介面：兩段式時間段執行
    # ================================================================

    def run_time_slot(self,
                      scene: str,
                      image: Optional[Image.Image] = None,
                      input_text: str = "") -> dict:
        """
        執行一個完整時間段（Phase 1 執行 + Phase 2 決策）。
        所有角色完成後時鐘自動推進。

        回傳：
        {
            "time":    當前時間字串,
            "execute": {code: execute_result},
            "decide":  {code: decide_result},
        }
        """
        full_scene = f"{self.clock.scene_prefix()} {scene}".strip()
        time_str   = self.clock.time_str

        logger.info(f"===== 時間段 {time_str} 開始 =====")

        # Phase 1：重置各角色本輪暫存狀態
        for char in self._characters.values():
            char.reset_slot_state()

        # Phase 1：初始化 pending（第一輪從時間表取）
        self._init_pending_from_schedule(time_str)

        # Phase 1：執行
        execute_results = self._execute_phase(full_scene, image)

        # Phase 2：決策（決定下一時段要做什麼）
        decide_results = self._decide_phase(full_scene, image, input_text)

        # 同步推進時鐘
        self.clock.tick()
        logger.info(f"===== 時間段 {time_str} 結束，推進至 {self.clock.time_str} =====")

        return {
            "time":    time_str,
            "execute": execute_results,
            "decide":  decide_results,
        }

    # ================================================================
    # 向後相容介面（main.py demo 與舊測試用）
    # ================================================================

    def step_character(self, code: str,
                       scene: str,
                       image: Optional[Image.Image] = None,
                       input_text: str = "",
                       image_desc: str = "",
                       target_code: Optional[str] = None,
                       _forwarded: bool = False) -> dict:
        """
        單角色執行一輪（舊介面，向後相容）。
        直接呼叫 decide()，並在對話時傳遞給目標。
        """
        agent     = self._agents[code]
        char      = self._characters[code]
        full_scene = f"{self.clock.scene_prefix()} {scene}".strip()

        # 若此角色已透過對話轉傳回應過，直接回傳快取結果（避免重複決策）
        if not _forwarded and code in self._conv_result_cache:
            cached = self._conv_result_cache.pop(code)
            logger.info(f"[{code}] 使用對話回應快取（已與對方配對）")
            log_turn(
                logger,
                code    = code,
                turn_id = f"D{char.day:03d}_T{self._stms[code].count():03d}",
                action  = cached.get("action", ""),
                c_value = cached.get("confusion", {}).get("C", 0.0),
                mode    = cached.get("mode", "deliberate"),
            )
            # 完成對話配對記錄（response 已在 _forward_dialogue 中設好，不覆蓋）
            if code in self._pending_pair:
                entry = self._pending_pair.pop(code)
                if entry.get("response") is None:
                    resp = cached.get("action", "")
                    entry["response"] = resp[len("對話:"):] if resp.startswith("對話:") else resp
                self._dialogue_history.append(entry)
            return cached

        # 決定同地點角色名字列表
        co_names = self._get_co_located_names(code)

        result = agent.decide(
            scene            = full_scene,
            co_located_names = co_names,
            image            = image,
            input_text       = input_text,
            image_desc       = image_desc,
        )

        log_turn(
            logger,
            code    = code,
            turn_id = f"D{char.day:03d}_T{self._stms[code].count():03d}",
            action  = result["action"],
            c_value = result["confusion"]["C"],
            mode    = result["mode"],
        )

        # 對話傳遞：明確 target_code 優先，否則推斷同地點第一人
        if result["verb"] == "對話" and not _forwarded:
            effective_target = target_code
            if not effective_target and co_names:
                from config.world_config import CHARACTER_CODES as _CC
                effective_target = _CC.get(co_names[0], "")
            if effective_target and effective_target in self._agents:
                # markov 路徑時 result["target"] 是人名，不是對話內容
                dial_msg = result["target"]
                if result.get("mode") == "markov" or dial_msg in CHARACTER_NAMES.values():
                    dial_msg = ""
                self._forward_dialogue(
                    code, effective_target, dial_msg, full_scene,
                    sender_thought=result.get("thought", ""),
                    sender_mode=result.get("mode", ""),
                )
            elif not effective_target:
                # 沒有對話對象（獨處）→ 改為休息，不對空氣說話
                logger.debug(f"[{code}] 想對話但無對象，改為休息")
                result["verb"]   = "休息"
                result["action"] = "休息"
                result["target"] = ""

        # 睡覺觸發濃縮
        if result["should_sleep"]:
            self._do_sleep(code)
        elif self._stms[code].is_full():
            logger.warning(f"[{code}] STM 已達容量上限，觸發中途濃縮")
            self._do_sleep(code)

        return result

    def step_all(self, scene: str,
                 image: Optional[Image.Image] = None,
                 input_text: str = "") -> dict:
        """對所有角色依序執行一輪（舊介面）。"""
        results = {}
        for code in sorted(self._agents.keys()):
            try:
                results[code] = self.step_character(
                    code       = code,
                    scene      = scene,
                    image      = image,
                    input_text = input_text,
                )
            except Exception as e:
                logger.error(f"[{code}] step 失敗：{e}")
                results[code] = {"error": str(e)}
        return results

    # ================================================================
    # Phase 1：執行階段
    # ================================================================

    def _execute_phase(self, scene: str,
                        image: Optional[Image.Image]) -> dict:
        """執行所有角色的 pending_action。"""
        results = {}

        # 找出對話配對
        conv_pairs = self._find_conversation_pairs()
        executed_conv = set()

        for code in sorted(self._agents.keys()):
            char   = self._characters[code]
            action = char.pending_action or "休息"
            verb   = action.split(":")[0]

            logger.info(f"[{code}] 執行：{action}")

            if verb == "睡覺":
                results[code] = {"action": action, "verb": "睡覺",
                                  "target": "", "should_sleep": True}
                self._do_sleep(code)

            elif verb == "對話" and code in conv_pairs and code not in executed_conv:
                partner = conv_pairs[code]
                if partner in conv_pairs:
                    # 雙方都想對話 → 執行對話
                    conv_result = self._run_conversation(
                        code, partner, scene, image
                    )
                    results[code]    = conv_result["a"]
                    results[partner] = conv_result["b"]
                    executed_conv.add(code)
                    executed_conv.add(partner)
                else:
                    # 對方不想對話，改為休息
                    char.current_action = "休息"
                    char.add_today_action("休息")
                    results[code] = {"action": "休息", "verb": "休息",
                                      "target": "", "should_sleep": False}

            elif code in executed_conv:
                # 已由對話處理，跳過
                pass

            else:
                # 一般行動：更新狀態
                tgt = action.split(":")[1] if ":" in action else ""
                char.current_action = verb
                if verb == "前往" and tgt:
                    char.current_location = tgt
                char.add_today_action(action)
                results[code] = {"action": action, "verb": verb,
                                  "target": tgt, "should_sleep": False}

            # STM 滿載觸發中途濃縮
            if self._stms[code].is_full():
                logger.warning(f"[{code}] STM 容量上限，觸發中途濃縮")
                self._do_sleep(code)

        return results

    # ================================================================
    # Phase 2：決策階段
    # ================================================================

    def _decide_phase(self, scene: str,
                       image: Optional[Image.Image],
                       input_text: str) -> dict:
        """所有角色針對下一時段做決策。"""
        results = {}
        for code in sorted(self._agents.keys()):
            agent = self._agents[code]
            char  = self._characters[code]

            co_names = self._get_co_located_names(code)

            try:
                result = agent.decide(
                    scene            = scene,
                    co_located_names = co_names,
                    image            = image,
                    input_text       = input_text,
                )
                log_turn(
                    logger,
                    code    = code,
                    turn_id = f"D{char.day:03d}_T{self._stms[code].count():03d}",
                    action  = result["action"],
                    c_value = result["confusion"]["C"],
                    mode    = result["mode"],
                )
                results[code] = result

                # 決策為睡覺 → 觸發濃縮，並清除 pending 防止 Phase 1 重複執行
                if result["should_sleep"]:
                    self._do_sleep(code)
                    char.pending_action = ""
                elif self._stms[code].is_full():
                    logger.warning(f"[{code}] STM 容量上限，觸發中途濃縮")
                    self._do_sleep(code)
                    char.pending_action = ""

            except Exception as e:
                logger.error(f"[{code}] decide 失敗：{e}")
                results[code] = {"error": str(e)}

        return results

    # ================================================================
    # 對話執行（Phase 1 內部）
    # ================================================================

    def _run_conversation(self, code_a: str, code_b: str,
                           scene: str,
                           image: Optional[Image.Image]) -> dict:
        """
        執行兩個代理人之間的對話（最多 MAX_CONVERSATION_TURNS_PER_SLOT 輪）。
        任一方回應不是「對話」時提前結束。
        """
        agent_a = self._agents[code_a]
        agent_b = self._agents[code_b]
        char_a  = self._characters[code_a]
        char_b  = self._characters[code_b]
        name_a  = CHARACTER_NAMES.get(code_a, code_a)
        name_b  = CHARACTER_NAMES.get(code_b, code_b)

        char_a.conversation_partner = code_b
        char_b.conversation_partner = code_a

        last_message = f"（{name_a} 與 {name_b} 開始對話）"
        turns_log = []

        for turn_idx in range(MAX_CONVERSATION_TURNS_PER_SLOT):
            # A 回應
            resp_a = agent_a.step(
                scene       = scene,
                image       = image,
                input_text  = last_message,
                target_code = code_b,
            )
            char_a.conversation_turns_this_slot += 1
            turns_log.append({"speaker": code_a, "result": resp_a})

            if resp_a["verb"] != "對話":
                logger.info(f"[{code_a}] 決定結束對話（{resp_a['verb']}）")
                break

            said_a       = resp_a["target"] or ""
            last_message = f"{name_a}對你說：{said_a}"

            # B 回應
            resp_b = agent_b.step(
                scene       = scene,
                image       = image,
                input_text  = last_message,
                target_code = code_a,
            )
            char_b.conversation_turns_this_slot += 1
            turns_log.append({"speaker": code_b, "result": resp_b})

            if resp_b["verb"] != "對話":
                logger.info(f"[{code_b}] 決定結束對話（{resp_b['verb']}）")
                break

            said_b       = resp_b["target"] or ""
            last_message = f"{name_b}對你說：{said_b}"

        total_turns = char_a.conversation_turns_this_slot
        logger.info(
            f"[{code_a}↔{code_b}] 對話結束，共 {total_turns} 輪"
        )

        last_a = turns_log[-2]["result"] if len(turns_log) >= 2 else turns_log[-1]["result"] if turns_log else {}
        last_b = turns_log[-1]["result"] if turns_log else {}

        return {
            "a": {**last_a, "conversation_turns": total_turns,
                  "turns_log": turns_log},
            "b": {**last_b, "conversation_turns": total_turns,
                  "turns_log": turns_log},
        }

    # ================================================================
    # 對話配對
    # ================================================================

    def _find_conversation_pairs(self) -> dict:
        """
        根據位置配對對話角色。
        - 只要一方 pending_action 以「對話」開頭，且同地點有其他角色，就可配對。
        - 優先配兩方都想對話的組合；退而求其次配單方主動的組合。
        回傳 {code: partner_code} dict（雙向都有）。
        """
        want_talk = set()
        for code, char in self._characters.items():
            if (char.pending_action or "").startswith("對話"):
                want_talk.add(code)

        # 按地點分組所有角色
        at_location: dict = {}
        for code, char in self._characters.items():
            loc = char.current_location or ""
            at_location.setdefault(loc, []).append(code)

        pairs = {}
        used  = set()

        for code_a in sorted(want_talk):
            if code_a in used:
                continue

            my_loc   = self._characters[code_a].current_location or ""
            at_loc   = at_location.get(my_loc, [])
            best_b   = None

            # 優先選同地點也想對話的人
            for code_b in at_loc:
                if code_b == code_a or code_b in used:
                    continue
                if code_b in want_talk:
                    best_b = code_b
                    break

            # 退而求其次：選同地點任意角色
            if best_b is None:
                for code_b in at_loc:
                    if code_b == code_a or code_b in used:
                        continue
                    best_b = code_b
                    break

            if best_b:
                pairs[code_a] = best_b
                pairs[best_b] = code_a
                used.add(code_a)
                used.add(best_b)

        return pairs

    # ================================================================
    # 初始化 pending（從時間表取第一輪行動）
    # ================================================================

    def _init_pending_from_schedule(self, time_str: str):
        """
        若某角色還沒有 pending_action，
        從當前時間表找到應執行的時段作為 pending。
        """
        for code, char in self._characters.items():
            if char.pending_action:
                continue  # 已有待執行決策

            slot = self.clock.get_pending_slots(char)
            if slot:
                first = slot[0]
                char.pending_action = first["action"]
                char.pending_target = first.get("location", "")
                logger.debug(
                    f"[{code}] 從時間表取得初始行動：{char.pending_action}"
                )

    # ================================================================
    # 同地點角色查詢
    # ================================================================

    def _get_co_located_names(self, code: str) -> list:
        """回傳與 code 角色在同一地點的其他角色名字列表。"""
        my_loc = self._characters[code].current_location
        names  = []
        for other_code, other_char in self._characters.items():
            if other_code != code and other_char.current_location == my_loc:
                names.append(CHARACTER_NAMES.get(other_code, other_code))
        return names

    # ================================================================
    # 取得角色 / Agent
    # ================================================================

    def get_character(self, code: str) -> Character:
        return self._characters[code]

    def get_agent(self, code: str) -> Agent:
        return self._agents[code]

    def all_codes(self) -> list:
        return list(self._agents.keys())

    def pop_dialogue_history(self) -> list:
        """取出並清空本時間段已完成的對話配對記錄（供報告使用）。"""
        history = list(self._dialogue_history)
        self._dialogue_history.clear()
        return history

    def clear_conversation_cache(self):
        """清除跨時間段的對話快取殘留（每個時間段開始前呼叫）。"""
        self._conv_result_cache.clear()
        self._pending_pair.clear()
        self._dialogue_history.clear()

    def all_sleeping(self) -> bool:
        """是否所有角色都已睡覺（STM 清空）。"""
        return all(self._stms[c].count() == 0 for c in self._agents)

    def all_sleeping_today(self) -> bool:
        """今天所有角色是否都已入睡（用於自主模式）。"""
        return self._sleeping_today >= set(self._agents.keys())

    def reset_sleeping_today(self):
        """新的一天開始時重置今日入睡記錄。"""
        self._sleeping_today.clear()

    def is_char_sleeping_today(self, code: str) -> bool:
        return code in self._sleeping_today

    # ================================================================
    # 自主模擬：一天完整流程
    # ================================================================

    def run_autonomous_day(self, day_n: int) -> dict:
        """
        執行一天完整自主模擬（06:00 → 自主決定睡覺，最晚 04:00 強制）。
        每個 tick（1 小時）所有角色自主決策並執行。

        回傳格式與 run_four_day_simulation 的 day_data 相同，供 generate_report 使用。
        """
        from config.world_config import MAX_TICKS_PER_DAY

        self.reset_sleeping_today()
        day_data = {'day': day_n, 'slots': [], 'sleep': {}, 'anomalies': []}

        def _sleep_and_record(code: str):
            """睡覺前記錄狀態，睡覺後寫入 day_data['sleep']。"""
            if code in day_data['sleep']:
                return  # 已記錄過
            agent  = self._agents[code]
            char   = self._characters[code]
            stm_b  = agent.stm.count()
            ltm_b  = agent.ltm.count()
            emo_b  = char.emotion
            res    = self._do_sleep(code)
            day_data['sleep'][code] = {
                'name':           char.name,
                'new_props':      res['new_propositions'],
                'ltm_total':      res['ltm_total'],
                'ltm_pruned':     res['ltm_pruned'],
                'stm_before':     stm_b,
                'ltm_before':     ltm_b,
                'ltm_after':      agent.ltm.count(),
                'emotion_before': emo_b,
                'emotion_after':  char.emotion,
                'summary':        res.get('summary', ''),
                'day_after':      char.day,
            }

        for tick_i in range(MAX_TICKS_PER_DAY + 1):
            # 強制入睡（凌晨 04:00 / 第 22 tick）
            if self.clock.is_forced_sleep_time():
                for code in sorted(self._agents.keys()):
                    if not self.is_char_sleeping_today(code):
                        logger.info(f"[{code}] 凌晨 04:00 強制入睡")
                        _sleep_and_record(code)
                break

            if self.all_sleeping_today():
                break

            time_str   = self.clock.time_str
            hour       = int(time_str.split(":")[0])
            # 時間感知場景前綴
            if hour < 8:
                time_ctx = "早晨"
            elif hour < 12:
                time_ctx = "上午工作時間"
            elif hour < 14:
                time_ctx = "午休時間"
            elif hour < 18:
                time_ctx = "下午"
            elif hour < 21:
                time_ctx = "傍晚收工時間"
            else:
                time_ctx = "夜晚，感到疲憊"
            full_scene = f"{self.clock.scene_prefix()}，{time_ctx}"

            logger.info(f"===== 自主時間段 {time_str} 開始 =====")
            self.clear_conversation_cache()

            # 各角色獨立決策（跳過今日已入睡者）
            slot_data = {
                'time':          time_str,
                'desc':          '自主決策時段',
                'clock':         full_scene,
                'agents':        {},
                'conversations': [],
            }

            for code in sorted(self._agents.keys()):
                if self.is_char_sleeping_today(code):
                    continue

                char      = self._characters[code]
                agent     = self._agents[code]
                co_names  = self._get_co_located_names(code)

                # 若此角色已由對話轉傳回應，直接取快取
                if code in self._conv_result_cache:
                    cached = self._conv_result_cache.pop(code)
                    log_turn(logger, code=code,
                             turn_id=f"D{char.day:03d}_T{self._stms[code].count():03d}",
                             action=cached.get("action", ""),
                             c_value=cached.get("confusion", {}).get("C", 0.0),
                             mode=cached.get("mode", "deliberate"))
                    if code in self._pending_pair:
                        entry = self._pending_pair.pop(code)
                        if entry.get("response") is None:
                            resp = cached.get("action", "")
                            entry["response"] = resp[len("對話:"):] if resp.startswith("對話:") else resp
                        self._dialogue_history.append(entry)
                    slot_data['agents'][code] = self._make_agent_slot_data(code, cached)
                    continue

                try:
                    result = agent.decide(
                        scene            = full_scene,
                        co_located_names = co_names,
                    )
                    log_turn(logger, code=code,
                             turn_id=f"D{char.day:03d}_T{self._stms[code].count():03d}",
                             action=result["action"],
                             c_value=result["confusion"]["C"],
                             mode=result["mode"])

                    # 對話傳遞
                    if result["verb"] == "對話":
                        effective_target = None
                        if co_names:
                            from config.world_config import CHARACTER_CODES as _CC
                            effective_target = _CC.get(co_names[0], "")
                        if effective_target and effective_target in self._agents:
                            # markov 路徑時 result["target"] 是人名，不是對話內容
                            dial_msg = result["target"]
                            if result.get("mode") == "markov" or dial_msg in CHARACTER_NAMES.values():
                                dial_msg = ""
                            self._forward_dialogue(
                                code, effective_target, dial_msg, full_scene,
                                sender_thought=result.get("thought", ""),
                                sender_mode=result.get("mode", ""),
                            )
                        else:
                            result["verb"]   = "休息"
                            result["action"] = "休息"

                    # 入睡（20:00 前不允許主動入睡，太早睡改成休息）
                    wants_sleep = result["should_sleep"] or result["verb"] == "睡覺"
                    if wants_sleep and hour < 20:
                        result["verb"]         = "休息"
                        result["action"]       = "休息"
                        result["should_sleep"] = False
                    elif wants_sleep:
                        _sleep_and_record(code)
                        result["action"] = "睡覺"
                    elif self._stms[code].is_full():
                        logger.warning(f"[{code}] STM 容量上限，觸發中途濃縮")
                        _sleep_and_record(code)

                    slot_data['agents'][code] = self._make_agent_slot_data(code, result)

                except Exception as e:
                    logger.error(f"[{code}] 自主決策失敗：{e}")

            slot_data['conversations'] = self.pop_dialogue_history()
            day_data['slots'].append(slot_data)
            self.clock.tick()

        return day_data

    def _make_agent_slot_data(self, code: str, result: dict) -> dict:
        """把 agent.decide() / step() 的結果轉換為 slot_data['agents'] 格式。"""
        char   = self._characters[code]
        agent  = self._agents[code]
        c_dict = result.get("confusion", {})
        return {
            'name':         char.name,
            'action':       result.get("action", ""),
            'verb':         result.get("verb", ""),
            'mode':         result.get("mode", "markov"),
            'confusion_C':  round(c_dict.get("C", 0.0), 4),
            'confusion_U':  round(c_dict.get("U", 0.0), 4),
            'confusion_K':  round(c_dict.get("K", 0.0), 4),
            'confusion_S':  round(c_dict.get("S", 0.0), 4),
            'action_probs': result.get("action_probs", {}),
            'thought':      result.get("thought", ""),
            'ham_count':    len(result.get("ham", [])),
            'stm_count':    agent.stm.count(),
            'ltm_count':    agent.ltm.count(),
            'location':     char.current_location,
            'emotion':      char.emotion,
            'should_sleep': result.get("should_sleep", False),
        }

    # ================================================================
    # 私有：舊對話傳遞（step_character 向後相容用）
    # ================================================================

    def _forward_dialogue(self, sender_code: str, receiver_code: str,
                          message: str, scene: str,
                          sender_thought: str = "",
                          sender_mode: str = ""):
        """呼叫接收方的 agent.step() 生成對話回應並存入快取。"""
        sender_name   = CHARACTER_NAMES.get(sender_code, sender_code)
        receiver_name = CHARACTER_NAMES.get(receiver_code, receiver_code)
        logger.info(f"[{sender_code}→{receiver_code}] 對話傳遞")
        agent  = self._agents[receiver_code]
        # message 為空時（markov 路徑選對話，沒有具體台詞）用自然開場
        if message:
            input_text = f"{sender_name}對你說：{message}"
        else:
            input_text = f"（{sender_name} 主動走近，開口跟你說話）"
        result = agent.step(
            scene       = scene,
            input_text  = input_text,
            target_code = sender_code,
        )
        self._conv_result_cache[receiver_code] = result

        # 立即從 step() 結果擷取接收方資訊，完整記錄雙方這輪對話
        recv_action = result.get("action", "")
        recv_text   = recv_action[len("對話:"):] if recv_action.startswith("對話:") else recv_action

        self._pending_pair[receiver_code] = {
            "sender_code":      sender_code,
            "sender_name":      sender_name,
            "receiver_code":    receiver_code,
            "receiver_name":    receiver_name,
            "message":          message,
            "sender_thought":   sender_thought,
            "sender_mode":      sender_mode,
            "response":         recv_text,
            "receiver_thought": result.get("thought", ""),
            "receiver_mode":    result.get("mode", ""),
        }

    # ================================================================
    # 私有：睡眠濃縮 + 存檔
    # ================================================================

    def _do_sleep(self, code: str) -> dict:
        agent  = self._agents[code]
        char   = self._characters[code]
        result = agent.sleep()
        log_consolidation(
            logger,
            code      = char.code,
            day       = char.day,
            stm_count = result["new_propositions"],
            ltm_count = result["ltm_total"],
        )
        save_character(code, char.to_dict())
        logger.info(f"[{code}] 存檔完成，進入第 {char.day} 天")
        self._sleeping_today.add(code)

        if self.all_sleeping_today():
            self.clock.advance_day()
            logger.info(f"所有角色已入睡，世界推進到第 {self.clock.day} 天")

        return result

    def collect_sleep_data(self, pre_sleep: dict) -> dict:
        """
        收集所有角色今天的睡眠濃縮數據，供 generate_report 使用。
        pre_sleep: {code: {'stm_before': n, 'ltm_before': n, 'emotion_before': str}}
        """
        result = {}
        for code in sorted(self._agents.keys()):
            agent = self._agents[code]
            char  = self._characters[code]
            pre   = pre_sleep.get(code, {})
            result[code] = {
                'name':           char.name,
                'new_props':      0,
                'ltm_total':      agent.ltm.count(),
                'ltm_pruned':     0,
                'stm_before':     pre.get('stm_before', 0),
                'ltm_before':     pre.get('ltm_before', 0),
                'ltm_after':      agent.ltm.count(),
                'emotion_before': pre.get('emotion_before', char.emotion),
                'emotion_after':  char.emotion,
                'summary':        agent.ltm.get_summary(),
                'day_after':      char.day,
            }
        return result
