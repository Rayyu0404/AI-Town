# ================================================================
# model/output_parser.py
# 解析模型輸出的三個區塊：[ACTION] [THOUGHT] [HAM]
#
# 模型輸出格式（prompts.py 定義）：
#   [ACTION]: 前往:咖啡廳
#   [THOUGHT]: Amy 的內心想法。
#   [HAM]: [{"subject":"Amy","relation":"遇見","object":"David"}]
#   [/HAM]
# ================================================================

import json
import re

from config.action_list import VALID_ACTIONS, VALID_LOCATIONS, ACTION_SEPARATOR


def parse_output(raw: str) -> dict:
    """
    一次解析模型的完整輸出。
    回傳：
    {
        "action":  str,   # 完整行動字串，例如 "前往:咖啡廳"
        "verb":    str,   # 行動動詞，例如 "前往"
        "target":  str,   # 行動目標（對話內容或地點），可為空字串
        "thought": str,
        "ham":     list,  # HAM 命題 list
        "raw":     str,   # 原始輸出（debug 用）
    }
    """
    action_raw = _parse_action(raw)
    verb, target = _split_action(action_raw)

    return {
        "action":  action_raw,
        "verb":    verb,
        "target":  target,
        "thought": _parse_thought(raw),
        "ham":     _parse_ham(raw),
        "raw":     raw,
    }


# ── 各區塊解析 ───────────────────────────────────────────────────

def _parse_action(raw: str) -> str:
    """
    從 [ACTION]: ... 行提取行動字串。
    無法解析時回傳 "休息"（最安全的 fallback）。
    """
    match = re.search(r"\[ACTION\]\s*:\s*(.+)", raw)
    if not match:
        return "休息"

    action = match.group(1).strip().split("\n")[0].strip()

    verb = action.split(ACTION_SEPARATOR)[0].strip()
    if verb in VALID_ACTIONS:
        return action

    # fallback：從輸出中找任何合法動詞
    for v in VALID_ACTIONS:
        if v in action:
            return v

    return "休息"


def _parse_thought(raw: str) -> str:
    """從 [THOUGHT]: ... 行提取內心想法。"""
    match = re.search(r"\[THOUGHT\]\s*:\s*(.+?)(?=\[|$)", raw, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().split("\n")[0].strip()


def _parse_ham(raw: str) -> list:
    """
    從 [HAM]: ... [/HAM] 或 [HAM]: ... 提取 JSON 命題列表。
    解析失敗時回傳空 list。
    """
    block_match = re.search(r"\[HAM\]\s*:?\s*(.*?)\[/HAM\]", raw, re.DOTALL)
    if block_match:
        return _try_parse_json_list(block_match.group(1))

    inline_match = re.search(r"\[HAM\]\s*:\s*(\[.+?\])", raw)
    if inline_match:
        return _try_parse_json_list(inline_match.group(1))

    return []


# ── 工具函式 ─────────────────────────────────────────────────────

def _split_action(action: str) -> tuple:
    """
    把 "前往:咖啡廳" 拆成 ("前往", "咖啡廳")。
    無分隔符時回傳 (action, "")。
    """
    if ACTION_SEPARATOR in action:
        parts = action.split(ACTION_SEPARATOR, 1)
        return parts[0].strip(), parts[1].strip()
    return action.strip(), ""


def _try_parse_json_list(text: str) -> list:
    """嘗試解析 JSON list，失敗回傳空 list。"""
    try:
        text = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return []
