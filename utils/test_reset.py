# ================================================================
# utils/test_reset.py
# 測試用：將角色 JSON 可變欄位重置為乾淨初始狀態
# 不影響靜態設定（個性、職業、人際關係 initial）
# ================================================================

import json
import os
from config.world_config import AI_DATA_DIR, CHARACTER_NAMES


def reset_character(code: str):
    """重置指定角色的可變狀態為乾淨初始值。"""
    path = os.path.join(AI_DATA_DIR, f"{code}_init.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── 狀態重置 ─────────────────────────────────────────────────
    data["state"]["day"]             = 1
    data["state"]["today_actions"]   = []
    data["state"]["current_action"]  = "休息"
    data["state"]["current_location"] = data.get("residence", "公寓")

    # ── STM 清空 ─────────────────────────────────────────────────
    data["stm"]["turns"] = []

    # ── LTM 清空 ─────────────────────────────────────────────────
    data["ltm"]["propositions"] = []
    data["ltm"]["ltm_summary"]  = ""

    # ── 情緒重置 ─────────────────────────────────────────────────
    data["emotion"] = "平靜"

    # ── 時間表：移除動態時段、所有時段標為未完成 ─────────────────
    data["schedule"]["slots"] = [
        s for s in data["schedule"].get("slots", [])
        if s.get("type") != "dynamic"
    ]
    for slot in data["schedule"]["slots"]:
        slot["completed"] = False
        slot.pop("_remove", None)

    # ── 關係摘要清空（保留 initial）─────────────────────────────
    for rel in data["relationships"].values():
        rel["summary"] = ""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reset_all():
    """重置所有角色的可變狀態。"""
    for code in CHARACTER_NAMES:
        reset_character(code)
    print(f"已重置 {len(CHARACTER_NAMES)} 個角色：{list(CHARACTER_NAMES.keys())}")


if __name__ == "__main__":
    reset_all()
