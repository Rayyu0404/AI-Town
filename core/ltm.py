# ================================================================
# core/ltm.py
# LTM 長期記憶管理（基於 HAM 命題結構）
# 負責命題的存入、提取、衰減、修剪
# 不涉及模型呼叫，純資料結構操作
# ================================================================

from config.world_config import LTM_DECAY_RATE, LTM_FORGET_THRESHOLD


class LTM:
    """
    長期記憶管理。
    直接操作 Character._data["ltm"]，不複製資料。

    命題格式：
    {
        "id":           "L001",
        "subject":      "Amy",
        "relation":     "遇見",
        "object":       "David",
        "location":     "咖啡廳",   # 可為 null
        "time":         "早上",     # 可為 null
        "strength":     1.0,        # 衰減值，0.0~1.0
        "access_count": 0,          # 被提取次數
        "encoded_day":  1           # 第幾天存入
    }
    """

    def __init__(self, character_data: dict):
        self._ltm = character_data["ltm"]

    # ── 寫入 ─────────────────────────────────────────────────────

    def encode(self, subject: str, relation: str, obj: str,
               location: str = None, time: str = None,
               day: int = 1) -> dict:
        """
        寫入一筆 HAM 命題。
        回傳寫入的命題 dict。
        """
        prop_id = self._next_id()
        prop = {
            "id":           prop_id,
            "subject":      subject,
            "relation":     relation,
            "object":       obj,
            "location":     location,
            "time":         time,
            "strength":     1.0,
            "access_count": 0,
            "encoded_day":  day,
        }
        self._ltm["propositions"].append(prop)
        return prop

    def encode_batch(self, propositions: list, day: int = 1):
        """
        批次寫入命題 list。
        每個命題 dict 需包含 subject / relation / object，
        location 和 time 可省略。
        """
        for p in propositions:
            self.encode(
                subject  = p.get("subject", ""),
                relation = p.get("relation", ""),
                obj      = p.get("object", ""),
                location = p.get("location"),
                time     = p.get("time"),
                day      = day,
            )

    # ── 提取 ─────────────────────────────────────────────────────

    def retrieve(self, query_subject: str = None,
                 query_relation: str = None,
                 query_object: str = None,
                 top_k: int = 10,
                 update_access: bool = True) -> list:
        """
        條件式提取命題，命中時 access_count +1、strength 重置為 1.0。
        所有條件為 AND，None 表示不限制該欄位。
        回傳符合條件的命題 list（最多 top_k 筆）。
        update_access=False 可避免同一命題在連續 retrieve 中被重複計數。
        """
        results = []
        for prop in self._ltm["propositions"]:
            if query_subject  and prop["subject"]  != query_subject:  continue
            if query_relation and prop["relation"] != query_relation: continue
            if query_object   and prop["object"]   != query_object:   continue

            if update_access:
                prop["access_count"] += 1
                prop["strength"] = 1.0

            results.append(prop)
            if len(results) >= top_k:
                break

        return results

    def get_all(self) -> list:
        """回傳所有命題（不更新 access_count）。"""
        return self._ltm["propositions"]

    def count(self) -> int:
        return len(self._ltm["propositions"])

    # ── 摘要（供直覺路徑注入 prompt）────────────────────────────

    def get_summary(self) -> str:
        """回傳 LTM 壓縮摘要文字。"""
        return self._ltm.get("ltm_summary", "")

    def set_summary(self, summary: str):
        """睡眠濃縮後由 memory_consolidation 更新摘要。"""
        self._ltm["ltm_summary"] = summary

    # ── 衰減（每天睡覺時呼叫）───────────────────────────────────

    def apply_decay(self):
        """
        對所有命題套用時間衰減。
        access_count 越高，衰減越慢：
            實際衰減率 = DECAY_RATE / (1 + access_count * 0.5)
        """
        for prop in self._ltm["propositions"]:
            ac = prop.get("access_count", 0)
            actual_decay = LTM_DECAY_RATE / (1 + ac * 0.5)
            prop["strength"] = max(0.0, prop["strength"] - actual_decay)

    # ── 修剪（衰減後清除低於門檻的命題）────────────────────────

    def prune(self) -> int:
        """
        刪除 strength 低於 LTM_FORGET_THRESHOLD 的命題。
        回傳刪除的筆數。
        """
        before = self.count()
        self._ltm["propositions"] = [
            p for p in self._ltm["propositions"]
            if p["strength"] >= LTM_FORGET_THRESHOLD
        ]
        removed = before - self.count()
        return removed

    # ── 格式化輸出（供 prompt_builder 使用）─────────────────────

    def to_text(self, props: list = None) -> str:
        """
        將命題 list 轉為可注入 prompt 的文字。
        props=None 時使用全部命題。
        格式：- Amy 遇見 David（咖啡廳，早上）
        """
        if props is None:
            props = self._ltm["propositions"]
        lines = []
        for p in props:
            line = f"- {p['subject']} {p['relation']} {p['object']}"
            ctx = []
            if p.get("location"): ctx.append(p["location"])
            if p.get("time"):     ctx.append(p["time"])
            if ctx:
                line += f"（{'，'.join(ctx)}）"
            lines.append(line)
        return "\n".join(lines) if lines else "（目前沒有相關記憶）"

    # ── 私有工具 ─────────────────────────────────────────────────

    def _next_id(self) -> str:
        """產生下一個命題 ID，格式 L001 L002 ...
        使用已有 ID 的最大值 +1，避免 prune 後產生重複 ID。"""
        props = self._ltm["propositions"]
        if not props:
            return "L001"
        max_n = max(int(p["id"][1:]) for p in props)
        return f"L{max_n + 1:03d}"