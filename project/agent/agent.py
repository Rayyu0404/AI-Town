# ================================================================
# agent/agent.py
# 單一角色完整一輪推論流程
# 整合：感知 → 困惑計算 → Prompt 組裝 → 模型推論 → 輸出解析 → STM 寫入
# ================================================================

from PIL import Image
from typing import Optional

from core.character import Character
from core.stm import STM
from core.ltm import LTM
from core.confusion import evaluate as eval_confusion
from core.memory_consolidation import consolidate
from model.prompt_builder import PromptBuilder
from model.output_parser import parse_output
from model.fusion_decoder import GenerationConfig
from config.world_config import SLEEP_ACTION
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

    # ── 主要推論流程（每輪呼叫一次）─────────────────────────────

    def step(self, scene: str,
             image: Optional[Image.Image] = None,
             input_text: str = "",
             image_desc: str = "",
             target_code: Optional[str] = None) -> dict:
        """
        執行一輪完整的感知—決策—記憶流程。

        scene       : 場景文字描述（時間、地點、天氣等）
        image       : 當前畫面（PIL Image），可為 None
        input_text  : 接收到的對話或事件文字
        image_desc  : YOLO 轉換的語意描述（yolo_handler 提供）
        target_code : 當前對話對象的角色代號（可為 None）

        回傳：
        {
            "action": str, "verb": str, "target": str,
            "thought": str, "ham": list,
            "mode": str, "confusion": dict,
            "should_sleep": bool,
        }
        """
        char = self.character

        # ── Step 1：計算困惑指數，決定直覺/思考路徑 ──────────────
        confusion = eval_confusion(
            image_desc     = image_desc,
            input_text     = input_text,
            current_action = char.current_action,
            current_scene  = scene,
            ltm_summary    = self.ltm.get_summary(),
            today_actions  = char.get_today_actions(),
            weights        = char.get_confusion_weights(),
        )
        mode = confusion["mode"]

        # ── Step 2：組裝 Prompt ────────────────────────────────────
        prompt_text = self.prompt_builder.build(scene, mode, target_code)

        # ── Step 3：準備模型輸入 ───────────────────────────────────
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
            text       = text_prompt.prompt,
            image_inputs = image_inputs,
        )

        # ── Step 4：模型推論 ──────────────────────────────────────
        gen_cfg = (GenerationConfig.intuitive()
                   if mode == "intuitive"
                   else GenerationConfig.deliberate())
        raw_output = self.loader.fusion.generate(fused, gen_cfg)

        # ── Step 5：解析輸出 ──────────────────────────────────────
        parsed = parse_output(raw_output)
        action = parsed["action"]

        # 更新角色當前行動與位置
        char.current_action = parsed["verb"]
        if parsed["verb"] == "前往" and parsed["target"]:
            char.current_location = parsed["target"]

        # ── Step 6：寫入 STM ──────────────────────────────────────
        turn_num = self.stm.next_turn_number()
        turn_id  = STM.make_turn_id(char.day, turn_num)
        self.stm.add_turn(
            turn_id         = turn_id,
            scene           = scene,
            image_desc      = image_desc,
            input_text      = input_text,
            action          = action,
            ham_propositions= parsed["ham"],
        )
        char.add_today_action(action)

        return {
            "action":       action,
            "verb":         parsed["verb"],
            "target":       parsed["target"],
            "thought":      parsed["thought"],
            "ham":          parsed["ham"],
            "mode":         mode,
            "confusion":    confusion,
            "should_sleep": parsed["verb"] == SLEEP_ACTION,
        }

    # ── 睡眠濃縮（world_clock 偵測到睡覺時呼叫）─────────────────

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
