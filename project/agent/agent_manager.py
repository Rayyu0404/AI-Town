# ================================================================
# agent/agent_manager.py
# 多角色排程管理：建立並協調所有角色的 Agent
# 負責 AI 間對話傳遞、睡眠觸發、資料存回磁碟
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
from config.world_config import CHARACTER_NAMES

logger = get_logger("agent_manager")


class AgentManager:
    """
    管理全部 5 個角色的 Agent 實例。
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

        for code, data in raw_data.items():
            char = Character(data)
            stm  = STM(data)
            ltm  = LTM(data)
            self._characters[code] = char
            self._stms[code]       = stm
            self._ltms[code]       = ltm
            self._agents[code]     = Agent(char, stm, ltm, loader)

        logger.info(f"AgentManager 初始化，角色：{list(self._agents.keys())}")

    # ── 對外主要介面 ──────────────────────────────────────────────

    def step_character(self, code: str,
                       scene: str,
                       image: Optional[Image.Image] = None,
                       input_text: str = "",
                       image_desc: str = "",
                       target_code: Optional[str] = None,
                       _forwarded: bool = False) -> dict:
        """
        對單一角色執行一輪推論。
        若輸出對話行動，自動將內容傳給目標角色。
        若選擇睡覺，自動觸發濃縮並存回磁碟。
        """
        agent     = self._agents[code]
        char      = self._characters[code]
        full_scene = f"{self.clock.scene_prefix()} {scene}".strip()

        result = agent.step(
            scene       = full_scene,
            image       = image,
            input_text  = input_text,
            image_desc  = image_desc,
            target_code = target_code,
        )

        log_turn(
            logger,
            code     = code,
            turn_id  = f"D{char.day:03d}_T{self._stms[code].count():03d}",
            action   = result["action"],
            c_value  = result["confusion"]["C"],
            mode     = result["mode"],
        )

        # 對話傳遞：只在非轉發情境下執行，防止 A→B→A 無限遞迴
        if (result["verb"] == "對話" and target_code
                and target_code in self._agents and not _forwarded):
            self._forward_dialogue(code, target_code, result["target"], full_scene)

        # 睡覺觸發濃縮；STM 滿載也觸發中途濃縮
        if result["should_sleep"]:
            self._do_sleep(code)
        elif self._stms[code].is_full():
            logger.warning(f"[{code}] STM 已達容量上限，觸發中途濃縮")
            self._do_sleep(code)

        return result

    def step_all(self, scene: str,
                 image: Optional[Image.Image] = None,
                 input_text: str = "") -> dict:
        """對所有角色依序執行一輪推論，回傳 {code: result}。"""
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

    # ── 取得角色/Agent ─────────────────────────────────────────────

    def get_character(self, code: str) -> Character:
        return self._characters[code]

    def get_agent(self, code: str) -> Agent:
        return self._agents[code]

    def all_codes(self) -> list:
        return list(self._agents.keys())

    def all_sleeping(self) -> bool:
        """是否所有角色都已睡覺（STM 清空）。"""
        return all(self._stms[c].count() == 0 for c in self._agents)

    # ── 私有：AI 間對話傳遞 ───────────────────────────────────────

    def _forward_dialogue(self, sender_code: str, receiver_code: str,
                          message: str, scene: str):
        sender_name = CHARACTER_NAMES.get(sender_code, sender_code)
        logger.info(f"[{sender_code}→{receiver_code}] 對話傳遞")
        self.step_character(
            code        = receiver_code,
            scene       = scene,
            input_text  = f"{sender_name}對你說：{message}",
            target_code = sender_code,
            _forwarded  = True,   # 防止 A→B→A 無限遞迴
        )

    # ── 私有：睡眠濃縮 + 存檔 ─────────────────────────────────────

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

        # 所有角色都已睡覺時，WorldClock 同步推進到下一天
        if self.all_sleeping():
            self.clock.advance_day()
            logger.info(f"所有角色已入睡，世界推進到第 {self.clock.day} 天")

        return result
