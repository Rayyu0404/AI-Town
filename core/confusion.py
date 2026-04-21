# ================================================================
# core/confusion.py
# 困惑指數計算：C = w1*U + w2*K + w3*S
# 決定本輪走直覺路徑還是思考路徑
#
# U (Uncertainty) : 模型對自身判斷的不確定性
# K (Conflict)    : 輸入之間的矛盾程度
# S (Surprise)    : 當前狀態與 LTM 預期的差異
#
# 參考：Shenhav, Botvinick & Cohen (2013)
#   The Expected Value of Control: ACC Function
# ================================================================


def compute_U(action_candidates: list) -> float:
    """
    計算不確定性 U。
    action_candidates : 模型輸出的候選行動清單，
                        每筆為 {"action": str, "score": float 0~1}
    分數越分散（最高分與第二高分差距越小）→ U 越高。
    只有一個候選時 U=0（完全確定）。
    """
    if not action_candidates or len(action_candidates) < 2:
        return 0.0

    scores = sorted([c.get("score", 0.0)
                     for c in action_candidates], reverse=True)
    top1, top2 = scores[0], scores[1]

    # 差距越小代表越不確定
    gap = top1 - top2
    U = max(0.0, 1.0 - gap * 2)
    return round(min(U, 1.0), 4)


def compute_K(image_desc: str, input_text: str,
              current_action: str) -> float:
    """
    計算衝突程度 K。
    簡化版：比較視覺描述、輸入文字、當前行動之間是否有明顯矛盾。
    目前用關鍵字比對，之後可升級為模型計算。

    image_desc     : YOLO 轉語意的圖片描述
    input_text     : 接收到的對話或事件
    current_action : 角色當前正在做的事
    """
    conflict_score = 0.0

    # 簡單規則：輸入要求行動，但當前行動是相反的
    CONFLICT_PAIRS = [
        (["離開", "走了", "再見"], ["等待", "休息", "工作"]),
        (["緊急", "快", "危險"],  ["休息", "睡覺", "滑手機"]),
        (["來了", "到了", "進來"], ["工作", "整理"]),
    ]

    combined = f"{image_desc} {input_text}".lower()
    action_lower = current_action.lower()

    for trigger_words, conflict_actions in CONFLICT_PAIRS:
        has_trigger  = any(w in combined     for w in trigger_words)
        has_conflict = any(w in action_lower for w in conflict_actions)
        if has_trigger and has_conflict:
            conflict_score += 0.6

    return round(min(conflict_score, 1.0), 4)


def compute_S(current_scene: str, ltm_summary: str,
              today_actions: list) -> float:
    """
    計算驚訝程度 S。
    比較當前場景與 LTM 摘要的差異。

    current_scene  : 當前場景描述
    ltm_summary    : LTM 壓縮摘要（代表角色的「預期世界」）
    today_actions  : 今天已執行的行動（用來偵測偏離時間表）
    """
    # LTM 是空的（剛開始第一天）→ 無法比較，S=0
    if not ltm_summary:
        return 0.0

    surprise_score = 0.0

    # 規則 1：場景中出現 LTM 從未提過的人名或地點
    SURPRISE_KEYWORDS = ["陌生人", "從沒見過", "第一次", "意外", "突然"]
    if any(kw in current_scene for kw in SURPRISE_KEYWORDS):
        surprise_score += 0.7

    # 規則 2：今天完全沒有執行任何行動（卡住了）
    if not today_actions:
        surprise_score += 0.2

    return round(min(surprise_score, 1.0), 4)


def compute_confusion(U: float, K: float, S: float,
                      weights: dict) -> float:
    """
    計算最終困惑指數 C = w1*U + w2*K + w3*S。
    weights : {"w1": float, "w2": float, "w3": float}
    """
    w1 = weights.get("w1", 0.4)
    w2 = weights.get("w2", 0.3)
    w3 = weights.get("w3", 0.3)
    C = w1 * U + w2 * K + w3 * S
    return round(min(C, 1.0), 4)


def decide_mode(C: float, threshold: float) -> str:
    """
    依照 C 值與閾值決定思考路徑。
    回傳 "intuitive" 或 "deliberate"。
    """
    return "deliberate" if C >= threshold else "intuitive"


def evaluate(image_desc: str, input_text: str,
             current_action: str, current_scene: str,
             ltm_summary: str, today_actions: list,
             weights: dict,
             action_candidates: list = None) -> dict:
    """
    一次計算所有指數並決定模式，回傳完整結果 dict。
    這是主要對外介面，agent.py 直接呼叫這個。

    回傳格式：
    {
        "U": float, "K": float, "S": float,
        "C": float, "mode": "intuitive" | "deliberate"
    }
    """
    U = compute_U(action_candidates or [])
    K = compute_K(image_desc, input_text, current_action)
    S = compute_S(current_scene, ltm_summary, today_actions)
    C = compute_confusion(U, K, S, weights)
    mode = decide_mode(C, weights.get("threshold", 0.5))

    return {"U": U, "K": K, "S": S, "C": C, "mode": mode}