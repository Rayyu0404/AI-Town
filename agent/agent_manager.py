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
            # 完成對話配對記錄：補上接收方的回應
            if code in self._pending_pair:
                entry = self._pending_pair.pop(code)
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
                self._forward_dialogue(code, effective_target, result["target"], full_scene)
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

                # 決策為睡覺 → 觸發濃縮
                if result["should_sleep"]:
                    self._do_sleep(code)
                elif self._stms[code].is_full():
                    logger.warning(f"[{code}] STM 容量上限，觸發中途濃縮")
                    self._do_sleep(code)

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
        找出互相都想對話的角色配對。
        回傳 {code: partner_code} dict（雙向都有）。
        """
        want_talk = {}  # code -> target_code

        for code, char in self._characters.items():
            action = char.pending_action or ""
            if not action.startswith("對話"):
                continue

            target_str = action.split(":")[1].strip() if ":" in action else ""
            if not target_str:
                continue

            # target 可能是名字或代號
            target_code = CHARACTER_CODES.get(target_str, "")
            if not target_code:
                # 嘗試用代號反查
                for c, n in CHARACTER_NAMES.items():
                    if n == target_str:
                        target_code = c
                        break
            if target_code and target_code in self._characters:
                want_talk[code] = target_code

        pairs = {}
        for code_a, code_b in want_talk.items():
            if want_talk.get(code_b) == code_a:
                pairs[code_a] = code_b
                pairs[code_b] = code_a

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

    # ================================================================
    # 私有：舊對話傳遞（step_character 向後相容用）
    # ================================================================

    def _forward_dialogue(self, sender_code: str, receiver_code: str,
                          message: str, scene: str):
        """呼叫接收方的 agent.step() 生成對話回應並存入快取。"""
        sender_name   = CHARACTER_NAMES.get(sender_code, sender_code)
        receiver_name = CHARACTER_NAMES.get(receiver_code, receiver_code)
        logger.info(f"[{sender_code}→{receiver_code}] 對話傳遞")
        agent  = self._agents[receiver_code]
        result = agent.step(
            scene       = scene,
            input_text  = f"{sender_name}對你說：{message}",
            target_code = sender_code,
        )
        self._conv_result_cache[receiver_code] = result
        # 暫存傳送方資訊，等待接收方消費 cache 時補上回應
        self._pending_pair[receiver_code] = {
            "sender_code":   sender_code,
            "sender_name":   sender_name,
            "receiver_code": receiver_code,
            "receiver_name": receiver_name,
            "message":       message,
            "response":      None,
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

        if self.all_sleeping():
            self.clock.advance_day()
            logger.info(f"所有角色已入睡，世界推進到第 {self.clock.day} 天")

        return result
