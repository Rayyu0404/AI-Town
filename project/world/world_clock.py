# ================================================================
# world/world_clock.py
# 世界時間管理：模擬時鐘推進與睡眠時機偵測
#
# 世界時間是虛擬時間，與現實時間無關
# 每次 tick() 推進固定分鐘數，agent_manager 每輪呼叫一次
# ================================================================

from config.world_config import SLEEP_ACTION, DAY_START


class WorldClock:
    """
    虛擬世界時鐘。
    用字串 "HH:MM" 表示當前時間，方便與時間表比對。
    """

    def __init__(self, start_time: str = DAY_START,
                 minutes_per_tick: int = 30):
        """
        start_time       : 起始時間字串，格式 "HH:MM"
        minutes_per_tick : 每次 tick 推進的分鐘數
        """
        self._minutes        = _parse_time(start_time)
        self._minutes_per_tick = minutes_per_tick
        self._day            = 1

    # ── 時間存取 ─────────────────────────────────────────────────

    @property
    def time_str(self) -> str:
        """當前時間字串，例如 "09:30"。"""
        return _format_time(self._minutes)

    @property
    def day(self) -> int:
        return self._day

    # ── 推進時間 ─────────────────────────────────────────────────

    def tick(self):
        """推進一個時間單位。若超過午夜自動跨日。"""
        self._minutes += self._minutes_per_tick
        if self._minutes >= 24 * 60:
            self._minutes -= 24 * 60
            self._day += 1

    def advance_day(self):
        """強制推進到下一天早晨（供所有角色睡著後呼叫）。"""
        self._day += 1
        self._minutes = _parse_time(DAY_START)

    # ── 時間表判斷 ────────────────────────────────────────────────

    def should_trigger_slot(self, slot: dict) -> bool:
        """
        判斷當前時間是否已到達或超過某個時間表時段。
        slot : {"time": "HH:MM", "action": str, ...}
        """
        slot_minutes = _parse_time(slot["time"])
        return self._minutes >= slot_minutes

    def is_sleep_time(self, character) -> bool:
        """
        判斷角色現在是否應該睡覺。
        條件：當前待完成時段是睡覺 且 時間已到達該時段。
        """
        slot = character.get_current_slot()
        if slot is None:
            return False
        if slot["action"] != SLEEP_ACTION:
            return False
        return self.should_trigger_slot(slot)

    def get_pending_slots(self, character) -> list:
        """
        取得角色當前時間應執行但尚未完成的時段列表。
        """
        pending = []
        for slot in character.get_schedule():
            if slot.get("completed", False):
                continue
            if self.should_trigger_slot(slot):
                pending.append(slot)
        return pending

    # ── 場景描述（供 prompt 使用）────────────────────────────────

    def scene_prefix(self) -> str:
        """回傳可注入 prompt 的時間場景前綴，例如 "第1天 09:30"。"""
        return f"第{self._day}天 {self.time_str}"


# ── 工具函式 ─────────────────────────────────────────────────────

def _parse_time(time_str: str) -> int:
    """將 "HH:MM" 解析為從午夜算起的分鐘數。"""
    try:
        h, m = time_str.split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return 7 * 60  # 預設 07:00


def _format_time(minutes: int) -> str:
    """將分鐘數格式化為 "HH:MM"。"""
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"
