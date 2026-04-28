# ================================================================
# core/stm.py
# STM 短期記憶管理
# 每輪對話寫入一筆，超過容量上限觸發濃縮
# ================================================================

from config.world_config import STM_DEFAULT_CAPACITY


class STM:
    """
    短期記憶管理。
    直接操作 Character._data["stm"]，不複製資料。
    """

    def __init__(self, character_data: dict):
        # 直接參考 character dict 裡的 stm 區塊
        self._stm = character_data["stm"]

    # ── 寫入 ─────────────────────────────────────────────────────

    def add_turn(self, turn_id: str, scene: str, image_desc: str,
                 input_text: str, action: str,
                 ham_propositions: list) -> dict:
        """
        寫入一筆新的 STM 紀錄。
        回傳寫入的 turn dict。

        turn_id         : 唯一 ID，格式 D{day:03d}_T{turn:03d}，例如 D001_T003
        scene           : 場景文字描述（時間、地點、天氣等）
        image_desc      : YOLO 轉語意後的圖片描述
        input_text      : 接收到的對話或事件文字
        action          : 模型決定的行動
        ham_propositions: 本輪抽出的 HAM 命題 list
        """
        turn = {
            "turn_id":         turn_id,
            "scene":           scene,
            "image_desc":      image_desc,
            "input_text":      input_text,
            "action":          action,
            "ham_propositions": ham_propositions,
        }
        self._stm["turns"].append(turn)
        return turn

    # ── 讀取 ─────────────────────────────────────────────────────

    def get_all(self) -> list:
        """回傳所有 STM 紀錄。"""
        return self._stm["turns"]

    def get_recent(self, n: int) -> list:
        """回傳最近 n 筆 STM 紀錄（直覺路徑用）。"""
        return self._stm["turns"][-n:]

    def get_all_propositions(self) -> list:
        """
        取出所有輪次的 HAM 命題，合併成一個 list。
        睡眠濃縮時使用。
        """
        props = []
        for turn in self._stm["turns"]:
            props.extend(turn.get("ham_propositions", []))
        return props

    def count(self) -> int:
        """目前 STM 的筆數。"""
        return len(self._stm["turns"])

    def capacity(self) -> int:
        """STM 容量上限。"""
        return self._stm.get("capacity", STM_DEFAULT_CAPACITY)

    # ── 容量判斷 ─────────────────────────────────────────────────

    def is_full(self) -> bool:
        """是否達到容量上限，觸發濃縮。"""
        return self.count() >= self.capacity()

    # ── 清空（睡覺後呼叫）────────────────────────────────────────

    def clear(self):
        """清空所有 STM 紀錄，保留 capacity 設定。"""
        self._stm["turns"] = []

    # ── 產生 turn_id ─────────────────────────────────────────────

    @staticmethod
    def make_turn_id(day: int, turn_number: int) -> str:
        """
        產生唯一的 turn_id。
        例如：day=1, turn=3 -> "D001_T003"
        """
        return f"D{day:03d}_T{turn_number:03d}"

    def next_turn_number(self) -> int:
        """目前 STM 的下一個 turn 編號。"""
        return self.count() + 1