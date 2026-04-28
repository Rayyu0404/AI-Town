# ================================================================
# core/markov.py
# 馬可夫鏈決策引擎（簡單思考路徑專用）
#
# 原理：根據 STM 歷史行動序列，計算各行動的轉移機率，
#       選出機率最高的行動作為下一步決策。
#
# 不需要模型呼叫，純統計計算，速度快。
# ================================================================

import random

from config.action_list import VALID_ACTIONS, ACTION_SEPARATOR


def compute_action_probs(stm_turns: list,
                         valid_actions: list = None) -> dict:
    """
    根據 STM 歷史計算各行動動詞的轉移機率。

    stm_turns     : STM 的所有 turn 紀錄（stm.get_all() 的結果）
    valid_actions : 合法行動清單，None 時使用全域 VALID_ACTIONS

    回傳 {action_verb: probability} dict，機率加總為 1.0。
    """
    if valid_actions is None:
        valid_actions = VALID_ACTIONS

    # 從 STM 取出行動動詞序列
    verbs = _extract_verbs(stm_turns)

    if len(verbs) < 2:
        # 歷史不足，回傳均勻分布
        prob = round(1.0 / len(valid_actions), 4)
        return {a: prob for a in valid_actions}

    current_verb = verbs[-1]

    # 計算從 current_verb 出發的一步轉移計數
    transitions = _count_transitions(verbs, current_verb)

    if not transitions:
        # 沒有觀察到此動詞的轉移，退回用頻率分布
        transitions = _count_frequency(verbs)

    # 加入 Laplace 平滑，確保每個合法行動都有非零機率
    total = sum(transitions.values()) + len(valid_actions)
    probs = {
        a: round((transitions.get(a, 0) + 1) / total, 4)
        for a in valid_actions
    }

    # 正規化（消除浮點誤差）
    s = sum(probs.values())
    probs = {a: round(p / s, 4) for a, p in probs.items()}

    return probs


def select_best_action(probs: dict,
                       exclude: list = None) -> str:
    """
    按機率加權隨機採樣行動（而非 argmax）。
    相同機率時自然隨機化，避免永遠卡在清單第一個動作。

    exclude : 要排除的行動列表（例如已確定無法執行的行動）
    """
    if not probs:
        return "休息"

    candidates = {
        a: p for a, p in probs.items()
        if not exclude or a not in exclude
    }

    if not candidates:
        return "休息"

    actions = list(candidates.keys())
    weights = [candidates[a] for a in actions]
    return random.choices(actions, weights=weights, k=1)[0]


def resolve_dialogue_target(probs: dict,
                              co_located_names: list) -> tuple:
    """
    當 Markov 選出「對話」時，決定對話目標。

    co_located_names : 同地點的其他角色名字列表

    回傳 (best_verb, target_name)：
      - best_verb  : 最終選定的行動動詞
      - target_name: 對話目標名字，或空字串
    """
    if co_located_names and "對話" in probs:
        # 同地點有人時，對話機率 ×4 再採樣
        boosted = dict(probs)
        boosted["對話"] = probs["對話"] * 4
        s = sum(boosted.values())
        boosted = {a: p / s for a, p in boosted.items()}
        best = select_best_action(boosted)
    else:
        best = select_best_action(probs)

    if best == "對話":
        if co_located_names:
            # 選擇第一個同地點的角色（未來可改成依關係強度選）
            return "對話", co_located_names[0]
        else:
            # 沒有可對話的對象，退選第二高機率行動
            best = select_best_action(probs, exclude=["對話"])
            return best, ""

    if best == "前往":
        # 目標地點由排程決定，Markov 只決定動詞
        return "前往", ""

    return best, ""


def format_probs_display(probs: dict, top_n: int = 5) -> str:
    """
    格式化機率分布，方便日誌與報表顯示。
    例如：工作:0.4200, 散步:0.1500, 休息:0.1200
    """
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{a}:{p:.4f}" for a, p in sorted_items[:top_n]]
    return ", ".join(parts)


# ── 內部工具函式 ─────────────────────────────────────────────────

def _extract_verbs(stm_turns: list) -> list:
    """從 STM turns 提取行動動詞列表（去掉冒號後的目標）。"""
    verbs = []
    for t in stm_turns:
        action = t.get("action", "")
        if action:
            verb = action.split(ACTION_SEPARATOR)[0].strip()
            if verb:
                verbs.append(verb)
    return verbs


def _count_transitions(verbs: list, from_verb: str) -> dict:
    """計算從 from_verb 到下一個動詞的轉移計數。"""
    counts = {}
    for i in range(len(verbs) - 1):
        if verbs[i] == from_verb:
            next_v = verbs[i + 1]
            counts[next_v] = counts.get(next_v, 0) + 1
    return counts


def _count_frequency(verbs: list) -> dict:
    """計算所有動詞的頻率（fallback 用）。"""
    counts = {}
    for v in verbs:
        counts[v] = counts.get(v, 0) + 1
    return counts
