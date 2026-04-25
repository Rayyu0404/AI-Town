# ================================================================
# agent/agent.py
# 單一角色完整推論流程
#
# 兩段式時間軸設計：
#   decide()  : 決策階段 — 決定下一時段要做什麼
#               STM 路徑（簡單思考）：馬可夫鏈機率分布 → 選最高機率
#                 觸發條件：K < 0.4 且 C < threshold（各角色閾值不同）
#               LTM 路徑（複雜思考）：完整模型推理 → 自主決策
#                 觸發條件：K ≥ 0.4（情緒複雜/邏輯衝突 override）
#                         或 C ≥ threshold（整體困惑度超過閾值）
#
#   step()    : 對話執行階段 — 對話時段內生成實際對話內容
#               由 agent_manager._forward_dialogue() 觸發，
#               不重新計算困惑度，沿用完整推論流程生成對話
#
#   sleep()   : 睡眠濃縮 — STM → LTM → 衰減 → day+1
#
# 各角色困惑閾值：
#   A(Amy):   threshold=0.45  w1=0.4, w2=0.3, w3=0.3
#   B(Ben):   threshold=0.6   w1=0.3, w2=0.2, w3=0.5
#   C(Claire):threshold=0.5   w1=0.3, w2=0.4, w3=0.3
#   D(David): threshold=0.55  w1=0.4, w2=0.4, w3=0.2
#   E(Emma):  threshold=0.5   w1=0.3, w2=0.3, w3=0.4
# ================================================================

from PIL import Image
from typing import Optional

from core.character import Character
from core.stm import STM
from core.ltm import LTM
from core.confusion import evaluate as eval_confusion
from core.markov import (compute_action_probs, resolve_dialogue_target,
                          format_probs_display, select_best_action)
from core.memory_consolidation import consolidate
from model.prompt_builder import PromptBuilder
from model.output_parser import parse_output
from model.fusion_decoder import GenerationConfig
from config.world_config import SLEEP_ACTION
from config.action_list import VALID_ACTIONS, VALID_LOCATIONS
from utils.logger import get_logger

logger = get_logger("agent")


class Agent:
    """
    單一角色的推論代理人。
    loader : ModelLoader（已呼叫 load() 的實例）
    """

    def __init__(self, character: Character, stm: STM,
                 ltm: LTM, loader):
        self.character      = character
        self.stm            = stm
        self.ltm            = ltm
        self.loader         = loader
        self.prompt_builder = PromptBuilder(character, stm, ltm)

    # ================================================================
    # 決策階段（兩段式時間軸 Phase 2）
    # ================================================================

    def decide(self,
               scene: str,
               co_located_names: list = None,
               image: Optional[Image.Image] = None,
               input_text: str = "",
               image_desc: str = "") -> dict:
        """
        決策階段：決定下一個時間段要執行的行動。

        scene             : 當前場景描述（含時間前綴）
        co_located_names  : 同地點的其他角色名字列表（供對話行動配對）
        image             : 視覺輸入（可為 None）
        input_text        : 外部輸入文字（可為空）
        image_desc        : YOLO 語意描述（可為空）

        STM 路徑（簡單思考）：
          - 馬可夫鏈計算行動機率分布
          - 選出最高機率行動
          - 不呼叫語言模型，速度快
          - STM 只記錄行動與機率，無 HAM 命題

        LTM 路徑（複雜思考）：
          - 困惑指數 C >= 0.5 觸發
          - 完整記憶注入：全部 STM + 相關 LTM 命題
          - 模型自主推理，自己選擇決策
          - STM 記錄完整 turn（含 THOUGHT + HAM 命題）

        回傳 dict：
        {
            "action": str,              # 完整行動，例如 "工作" / "對話:Amy"
            "verb": str,                # 行動動詞
            "target": str,              # 目標（地點名或角色名）
            "thought": str,             # 內心想法（LTM 路徑才有）
            "ham": list,                # HAM 命題（LTM 路徑才有）
            "mode": str,                # "markov" 或 "deliberate"
            "action_probs": dict,       # 機率分布（markov 路徑才有）
            "confusion": dict,          # 困惑指數詳情
            "should_sleep": bool,
        }
        """
        char = self.character

        # ── 計算困惑指數，決定路徑 ───────────────────────────────
        # 用 Markov 機率分布作為 action_candidates，使 U 有意義：
        # 分布越均勻（不確定）→ U 越高 → 更容易觸發複雜思考。
        stm_turns_for_u = self.stm.get_all()
        markov_probs_for_u = compute_action_probs(stm_turns_for_u)
        action_candidates_for_u = [
            {"action": a, "score": p}
            for a, p in markov_probs_for_u.items()
        ]

        confusion = eval_confusion(
            image_desc       = image_desc,
            input_text       = input_text,
            current_action   = char.current_action,
            current_scene    = scene,
            ltm_summary      = self.ltm.get_summary(),
            today_actions    = char.get_today_actions(),
            weights          = char.get_confusion_weights(),
            action_candidates= action_candidates_for_u,
        )
        mode = confusion["mode"]

        if mode == "intuitive":
            return self._decide_markov(
                scene, co_located_names or [], confusion, image_desc, input_text
            )
        else:
            return self._decide_deliberate(
                scene, co_located_names or [], confusion,
                image, input_text, image_desc
            )

    # ── 簡單思考路徑（馬可夫鏈）─────────────────────────────────

    def _decide_markov(self, scene: str, co_located_names: list,
                        confusion: dict, image_desc: str,
                        input_text: str) -> dict:
        char = self.character

        # 計算機率分布
        action_probs = compute_action_probs(self.stm.get_all())

        # 解析對話目標（若選到「對話」需要找人）
        verb, target = resolve_dialogue_target(action_probs, co_located_names)

        # 若是「前往」則從時間表取目標地點
        if verb == "前往" and not target:
            target = self._next_schedule_location() or ""

        # 組裝完整行動字串
        if target:
            action = f"{verb}:{target}"
        else:
            action = verb

        # 更新角色狀態
        char.current_action = verb
        if verb == "前往" and target:
            char.current_location = target

        # 更新 pending
        char.pending_action = action
        char.pending_target = target
        char.pending_decision_done = True
        char.add_today_action(action)

        # 寫入 STM（精簡格式，不含 HAM）
        turn_num = self.stm.next_turn_number()
        turn_id  = STM.make_turn_id(char.day, turn_num)
        self.stm.add_turn(
            turn_id          = turn_id,
            scene            = scene,
            image_desc       = image_desc,
            input_text       = input_text,
            action           = action,
            ham_propositions = [],
        )

        prob_str = format_probs_display(action_probs, top_n=4)
        logger.debug(
            f"[{char.code}] MARKOV 決策: {action}  "
            f"機率分布: {prob_str}"
        )

        return {
            "action":       action,
            "verb":         verb,
            "target":       target,
            "thought":      "",
            "ham":          [],
            "mode":         "markov",
            "action_probs": action_probs,
            "confusion":    confusion,
            "should_sleep": verb == SLEEP_ACTION,
        }

    # ── 複雜思考路徑（LTM 完整推理）────────────────────────────

    def _decide_deliberate(self, scene: str, co_located_names: list,
                            confusion: dict, image: Optional[Image.Image],
                            input_text: str, image_desc: str) -> dict:
        char = self.character

        # 決定對話對象代號（如果有同地點的人）
        target_code = self._name_to_code(co_located_names[0]) if co_located_names else None

        # 組裝完整 LTM prompt
        prompt_text = self.prompt_builder.build(scene, "deliberate", target_code)

        # 準備模型輸入
        images     = [image] if image else []
        num_images = len(images)
        text_prompt = self.loader.text.build_prompt(
            prompt_text, num_images=num_images
        )
        if images:
            vision_batch = self.loader.vision.encode(images)
            image_inputs = vision_batch.to_dict()
        else:
            image_inputs = {}

        fused = self.loader.fusion.fuse_inputs(
            text=text_prompt.prompt, image_inputs=image_inputs
        )

        # 模型推論（複雜思考路徑，token 預算較大）
        raw_output = self.loader.fusion.generate(
            fused, GenerationConfig.deliberate()
        )
        parsed = parse_output(raw_output)
        action = parsed["action"]
        verb   = parsed["verb"]
        tgt    = parsed["target"]

        # 更新角色狀態
        char.current_action = verb
        if verb == "前往" and tgt:
            char.current_location = tgt

        # 更新 pending
        char.pending_action = action
        char.pending_target = tgt
        char.pending_decision_done = True
        char.add_today_action(action)

        # 寫入 STM（完整格式，含 HAM 命題）
        turn_num = self.stm.next_turn_number()
        turn_id  = STM.make_turn_id(char.day, turn_num)
        self.stm.add_turn(
            turn_id          = turn_id,
            scene            = scene,
            image_desc       = image_desc,
            input_text       = input_text,
            action           = action,
            ham_propositions = parsed["ham"],
        )

        logger.debug(
            f"[{char.code}] DELIBERATE 決策: {action}  "
            f"HAM: {len(parsed['ham'])} 筆"
        )

        return {
            "action":       action,
            "verb":         verb,
            "target":       tgt,
            "thought":      parsed["thought"],
            "ham":          parsed["ham"],
            "mode":         "deliberate",
            "action_probs": {},
            "confusion":    confusion,
            "should_sleep": verb == SLEEP_ACTION,
        }

    # ================================================================
    # 對話執行階段（time slot 內生成對話內容）
    # ================================================================

    def step(self, scene: str,
             image: Optional[Image.Image] = None,
             input_text: str = "",
             image_desc: str = "",
             target_code: Optional[str] = None) -> dict:
        """
        對話執行：生成一輪對話內容。
        用於執行階段中，兩個已配對代理人的實際對話輪次。

        不負責宏觀決策（decide() 已在上一個時段完成），
        只負責根據對方說的話生成本輪回應。

        回傳格式與原本相同（向後相容）。
        """
        char = self.character

        # 始終使用 LTM（deliberate）路徑來生成對話內容
        confusion = eval_confusion(
            image_desc     = image_desc,
            input_text     = input_text,
            current_action = char.current_action,
            current_scene  = scene,
            ltm_summary    = self.ltm.get_summary(),
            today_actions  = char.get_today_actions(),
            weights        = char.get_confusion_weights(),
        )

        prompt_text = self.prompt_builder.build(
            scene, "deliberate", target_code, input_text=input_text
        )
        images      = [image] if image else []
        num_images  = len(images)

        text_prompt = self.loader.text.build_prompt(
            prompt_text, num_images=num_images
        )
        if images:
            vision_batch = self.loader.vision.encode(images)
            image_inputs = vision_batch.to_dict()
        else:
            image_inputs = {}

        fused      = self.loader.fusion.fuse_inputs(
            text=text_prompt.prompt, image_inputs=image_inputs
        )
        raw_output = self.loader.fusion.generate(
            fused, GenerationConfig.deliberate()
        )
        parsed = parse_output(raw_output)
        action = parsed["action"]

        char.current_action = parsed["verb"]
        if parsed["verb"] == "前往" and parsed["target"]:
            char.current_location = parsed["target"]

        turn_num = self.stm.next_turn_number()
        turn_id  = STM.make_turn_id(char.day, turn_num)
        self.stm.add_turn(
            turn_id          = turn_id,
            scene            = scene,
            image_desc       = image_desc,
            input_text       = input_text,
            action           = action,
            ham_propositions = parsed["ham"],
        )
        char.add_today_action(action)

        return {
            "action":       action,
            "verb":         parsed["verb"],
            "target":       parsed["target"],
            "thought":      parsed["thought"],
            "ham":          parsed["ham"],
            "mode":         "deliberate",
            "action_probs": {},
            "confusion":    confusion,
            "should_sleep": parsed["verb"] == SLEEP_ACTION,
        }

    # ================================================================
    # 睡眠濃縮
    # ================================================================

    def sleep(self) -> dict:
        """
        執行睡眠濃縮流程：STM → LTM → 衰減 → day+1。
        回傳濃縮摘要 dict。
        """
        logger.info(f"[{self.character.code}] 開始睡眠濃縮...")
        result = consolidate(
            self.character, self.stm, self.ltm, self.loader.make_model_fn
        )
        logger.info(
            f"[{self.character.code}] 濃縮完成："
            f"新增 {result['new_propositions']} 筆 LTM，"
            f"修剪 {result['ltm_pruned']} 筆，"
            f"總計 {result['ltm_total']} 筆"
        )
        return result

    # ── 私有工具 ─────────────────────────────────────────────────

    def _next_schedule_location(self) -> str:
        """從角色時間表取得下一個目標地點。"""
        slot = self.character.get_current_slot()
        if slot and slot.get("location"):
            return slot["location"]
        return ""

    def _name_to_code(self, name: str) -> str:
        """將角色名字轉換為代號（供 target_code 使用）。"""
        from config.world_config import CHARACTER_CODES
        return CHARACTER_CODES.get(name, "")
