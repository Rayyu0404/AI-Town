# ================================================================
# core/character.py
# 角色物件：封裝角色 dict 的所有存取與狀態管理
# 其他模組透過這個類別操作角色，不直接碰 dict
# ================================================================

from config.world_config import CHARACTER_NAMES, SLEEP_ACTION


class Character:
    """
    單一角色的狀態管理。
    data : 從 file_io.load_character() 讀進來的 dict（記憶體中操作）
    """

    def __init__(self, data: dict):
        self._data = data

    # ── 基本屬性 ─────────────────────────────────────────────────

    @property
    def code(self) -> str:
        """角色代號，程式內部用（A B C D E）"""
        return self._data["name_code"]

    @property
    def name(self) -> str:
        """角色名字，組 prompt 時用（Amy Ben Claire David Emma）"""
        return self._data["name"]

    @property
    def role(self) -> str:
        return self._data["role"]

    @property
    def emotion(self) -> str:
        return self._data["emotion"]

    @emotion.setter
    def emotion(self, value: str):
        self._data["emotion"] = value

    @property
    def day(self) -> int:
        return self._data["state"]["day"]

    @property
    def current_location(self) -> str:
        return self._data["state"]["current_location"]

    @current_location.setter
    def current_location(self, value: str):
        self._data["state"]["current_location"] = value

    @property
    def current_action(self) -> str:
        return self._data["state"]["current_action"]

    @current_action.setter
    def current_action(self, value: str):
        self._data["state"]["current_action"] = value

    # ── 個性與習慣（供 prompt_builder 使用）─────────────────────

    def get_personality(self, short: bool = False) -> str:
        """
        short=True  -> personality_short（直覺路徑用）
        short=False -> personality 完整版（思考路徑用）
        """
        key = "personality_short" if short else "personality"
        return self._data.get(key, "")

    def get_habit(self) -> str:
        return self._data.get("habit", "")

    # ── 關係（供 prompt_builder 使用）───────────────────────────

    def get_relationship(self, target_code: str) -> dict:
        """
        取得與 target_code 角色的關係資料。
        回傳 {"initial": "...", "summary": "..."} 或空 dict。
        """
        return self._data["relationships"].get(target_code, {})

    def get_relationship_text(self, target_code: str,
                               include_summary: bool = True) -> str:
        """
        回傳可直接注入 prompt 的關係描述文字。
        include_summary=True  -> initial + summary（思考路徑用）
        include_summary=False -> 只有 initial（直覺路徑用）
        """
        rel = self.get_relationship(target_code)
        if not rel:
            target_name = CHARACTER_NAMES.get(target_code, target_code)
            return f"{self.name} 與 {target_name} 不認識。"

        text = rel.get("initial", "")
        if include_summary and rel.get("summary"):
            text += f" 最近的狀況：{rel['summary']}"
        return text

    def update_relationship_summary(self, target_code: str, summary: str):
        """睡眠濃縮時更新關係摘要。"""
        if target_code in self._data["relationships"]:
            self._data["relationships"][target_code]["summary"] = summary

    # ── 今日行動紀錄 ─────────────────────────────────────────────

    def add_today_action(self, action: str):
        """把行動加進今天的行動紀錄。"""
        self._data["state"]["today_actions"].append(action)

    def get_today_actions(self) -> list:
        return self._data["state"]["today_actions"]

    def clear_today_actions(self):
        """睡覺後清空。"""
        self._data["state"]["today_actions"] = []

    # ── 時間表 ───────────────────────────────────────────────────

    def get_schedule(self) -> list:
        return self._data["schedule"]["slots"]

    def get_current_slot(self) -> dict | None:
        """
        找到第一個尚未完成的時段回傳。
        若全部完成或時間表為空回傳 None。
        """
        for slot in self._data["schedule"]["slots"]:
            if not slot.get("completed", False):
                return slot
        return None

    def mark_slot_completed(self, time_str: str):
        """將指定時間的時段標記為已完成。"""
        for slot in self._data["schedule"]["slots"]:
            if slot["time"] == time_str:
                slot["completed"] = True
                break

    def insert_dynamic_slot(self, time_str: str, action: str,
                             location: str):
        """插入一個動態時段（臨時事件）。"""
        new_slot = {
            "time": time_str,
            "action": action,
            "location": location,
            "type": "dynamic",
            "completed": False
        }
        slots = self._data["schedule"]["slots"]
        # 按時間順序插入
        slots.append(new_slot)
        slots.sort(key=lambda s: s["time"])

    def is_sleep_time(self) -> bool:
        """當前待完成的時段是否為睡覺。"""
        slot = self.get_current_slot()
        return slot is not None and slot["action"] == SLEEP_ACTION

    def reset_schedule(self):
        """新的一天開始，重置所有時段為未完成。"""
        for slot in self._data["schedule"]["slots"]:
            if slot.get("type") == "dynamic":
                slot["_remove"] = True
        # 移除動態時段，保留固定時段
        self._data["schedule"]["slots"] = [
            s for s in self._data["schedule"]["slots"]
            if not s.get("_remove", False)
        ]
        for slot in self._data["schedule"]["slots"]:
            slot["completed"] = False

    # ── 困惑指數參數 ─────────────────────────────────────────────

    def get_confusion_weights(self) -> dict:
        return self._data["confusion_weights"]

    # ── 新增一天 ─────────────────────────────────────────────────

    def advance_day(self):
        """睡眠濃縮完成後呼叫，day +1 並重置當天狀態。"""
        self._data["state"]["day"] += 1
        self.clear_today_actions()
        self.reset_schedule()

    # ── 取得原始 dict（供 file_io 寫回磁碟用）───────────────────

    def to_dict(self) -> dict:
        return self._data