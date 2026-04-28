# ================================================================
# world/world_clock.py
# 世界時間管理：模擬時鐘推進
#
# 每天從 DAY_START（06:00）開始，每 tick 推進 MINUTES_PER_TICK 分鐘（1 小時）
# 最多 MAX_TICKS_PER_DAY（22）個 tick，到 04:00 強制入睡
# 日期只在 advance_day() 時推進（所有角色入睡後），不在 tick() 中自動推進
# ================================================================

from config.world_config import SLEEP_ACTION, DAY_START, MINUTES_PER_TICK, MAX_TICKS_PER_DAY


class WorldClock:
    """虛擬世界時鐘。"""

    def __init__(self, start_time: str = DAY_START,
                 minutes_per_tick: int = MINUTES_PER_TICK):
        self._minutes          = _parse_time(start_time)
        self._minutes_per_tick = minutes_per_tick
        self._day              = 1
        self._ticks_this_day   = 0   # 今天已推進幾個 tick

    # ── 時間存取 ─────────────────────────────────────────────────

    @property
    def time_str(self) -> str:
        return _format_time(self._minutes)

    @property
    def day(self) -> int:
        return self._day

    @property
    def ticks_today(self) -> int:
        return self._ticks_this_day

    # ── 推進時間 ─────────────────────────────────────────────────

    def tick(self):
        """推進一個小時，跨越午夜時只繞回，不自動增加天數。"""
        self._minutes += self._minutes_per_tick
        if self._minutes >= 24 * 60:
            self._minutes -= 24 * 60
        self._ticks_this_day += 1

    def advance_day(self):
        """所有角色入睡後呼叫：推進到下一天並重置計數器。"""
        self._day            += 1
        self._minutes         = _parse_time(DAY_START)
        self._ticks_this_day  = 0

    # ── 強制入睡判斷 ─────────────────────────────────────────────

    def is_forced_sleep_time(self) -> bool:
        """True 表示已到達或超過凌晨4點（22 個 tick 後）。"""
        return self._ticks_this_day >= MAX_TICKS_PER_DAY

    # ── 場景描述（供 prompt 使用）────────────────────────────────

    def scene_prefix(self) -> str:
        return f"第{self._day}天 {self.time_str}"

    # ── 舊介面相容（simulate.py FOUR_DAY_PLAN 用）───────────────

    def should_trigger_slot(self, slot: dict) -> bool:
        slot_minutes = _parse_time(slot["time"])
        return self._minutes >= slot_minutes

    def is_sleep_time(self, character) -> bool:
        slot = character.get_current_slot()
        if slot is None:
            return False
        if slot["action"] != SLEEP_ACTION:
            return False
        return self.should_trigger_slot(slot)

    def get_pending_slots(self, character) -> list:
        pending = []
        for slot in character.get_schedule():
            if slot.get("completed", False):
                continue
            if self.should_trigger_slot(slot):
                pending.append(slot)
        return pending


# ── 工具函式 ─────────────────────────────────────────────────────

def _parse_time(time_str: str) -> int:
    try:
        h, m = time_str.split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return 6 * 60  # 預設 06:00


def _format_time(minutes: int) -> str:
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"
