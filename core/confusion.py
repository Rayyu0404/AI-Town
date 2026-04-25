# ================================================================
# core/confusion.py
# 困惑指數計算：C = w1*U + w2*K + w3*S
# 決定本輪走直覺路徑（Markov）還是思考路徑（LTM 推理）
#
# U (Uncertainty) : 模型對自身判斷的不確定性
#   候選行動最高分與第二高分差距越小 → U 越高
#
# K (Conflict)    : 輸入之間的矛盾程度（邏輯衝突 + 情緒複雜性）
#   邏輯衝突偵測：CONFLICT_PAIRS 觸發 +0.6
#   強情緒（不知所措/思緒紊亂）：+0.45
#   一般情緒（心跳/緊張/告白）：+0.25
#   K ≥ 0.4 → 強制 deliberate（不論 C 是否超過閾值）
#
# S (Surprise)    : 當前場景與 LTM 預期的差異
#   高驚訝（陌生人/從沒見過）：+0.6
#   中高（第一次）：+0.5
#   中（意外/沒想到）：+0.35
#   低（突然）：+0.25
#   LTM 為空（第一天）→ S=0（無可比較的預期）
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
              current_action: str,
              current_scene: str = "") -> float:
    """
    計算衝突程度 K。
    同時偵測兩類衝突：
      1. 邏輯衝突：輸入要求行動與當前狀態矛盾（緊急 + 休息）
      2. 情緒複雜性：場景/輸入含有高情緒負荷關鍵字

    image_desc     : YOLO 轉語意的圖片描述
    input_text     : 接收到的對話或事件
    current_action : 角色當前正在做的事
    current_scene  : 當前場景描述（供情緒偵測用）
    """
    conflict_score = 0.0

    # ── 邏輯衝突偵測 ─────────────────────────────────────────────
    CONFLICT_PAIRS = [
        (["離開", "走了", "再見"],   ["等待", "休息", "工作"]),
        (["緊急", "快", "危險"],     ["休息", "睡覺", "滑手機"]),
        (["來了", "到了", "進來"],   ["工作", "整理"]),
    ]

    combined    = f"{image_desc} {input_text}".lower()
    action_lower = current_action.lower()

    for trigger_words, conflict_actions in CONFLICT_PAIRS:
        has_trigger  = any(w in combined      for w in trigger_words)
        has_conflict = any(w in action_lower  for w in conflict_actions)
        if has_trigger and has_conflict:
            conflict_score += 0.6

    # ── 情緒複雜性偵測（場景 + 輸入均納入） ──────────────────────
    combined_all = f"{combined} {current_scene}".lower()

    # 強情緒複雜性：思緒混亂、完全不知所措
    STRONG_EMOTION = [
        "混亂", "一片混亂", "說不清楚", "無法理解",
        "心亂", "不知所措", "不知道如何",
        "思緒很亂", "思緒紊亂",
    ]
    # 一般情緒/關係複雜性
    MOD_EMOTION = [
        "心跳加速", "心跳", "緊張", "期待", "在意",
        "感情", "告白", "表白", "暗戀", "心動",
        "情緒", "心情複雜", "心情有些",
        "思緒", "心裡有些",
    ]

    if any(kw in combined_all for kw in STRONG_EMOTION):
        conflict_score += 0.45   # 強烈情緒複雜性
    elif any(kw in combined_all for kw in MOD_EMOTION):
        conflict_score += 0.25   # 一般情緒複雜性

    return round(min(conflict_score, 1.0), 4)


def compute_S(current_scene: str, ltm_summary: str,
              today_actions: list) -> float:
    """
    計算驚訝程度 S。
    比較當前場景與 LTM 摘要的差異。

    current_scene  : 當前場景描述
    ltm_summary    : LTM 壓縮摘要（代表角色的「預期世界」）
    today_actions  : 今天已執行的行動（備用，目前未使用）

    驚訝分級：
      高：完全陌生的人/從沒見過          → 0.6
      中高：第一次發生的事               → 0.5
      中：一般「意外」                    → 0.35
      低：「突然」（常見口語，不必然驚訝）→ 0.25
    """
    # LTM 是空的（第一天）→ 無法比較，S=0
    if not ltm_summary:
        return 0.0

    surprise_score = 0.0

    # 高驚訝：真正遇到陌生或未預期的人/事
    HIGH_SURPRISE = ["陌生人", "從沒見過", "完全不認識", "第一次見到", "突然闖入"]
    # 中高驚訝：第一次發生
    MID_HIGH_SURPRISE = ["第一次", "從未"]
    # 中驚訝：一般意外
    MID_SURPRISE = ["意外", "沒想到", "不可思議"]
    # 低驚訝：「突然」這個口語詞（過度使用不代表真正驚訝）
    LOW_SURPRISE = ["突然"]

    if any(kw in current_scene for kw in HIGH_SURPRISE):
        surprise_score += 0.6
    elif any(kw in current_scene for kw in MID_HIGH_SURPRISE):
        surprise_score += 0.5
    elif any(kw in current_scene for kw in MID_SURPRISE):
        surprise_score += 0.35
    elif any(kw in current_scene for kw in LOW_SURPRISE):
        surprise_score += 0.25

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

    情緒強度 override：
      K >= 0.4 → 強制 deliberate
      （高情緒複雜度場景需要深度思考，不管 C 是否超過閾值或是否有 LTM）

    回傳格式：
    {
        "U": float, "K": float, "S": float,
        "C": float, "mode": "intuitive" | "deliberate"
    }
    """
    U = compute_U(action_candidates or [])
    K = compute_K(image_desc, input_text, current_action, current_scene)
    S = compute_S(current_scene, ltm_summary, today_actions)
    C = compute_confusion(U, K, S, weights)

    # 情緒強度 override：K >= 0.4 → 強制深度思考（不論是否有 LTM）
    if K >= 0.4:
        mode = "deliberate"
    else:
        mode = decide_mode(C, weights.get("threshold", 0.5))

    return {"U": U, "K": K, "S": S, "C": C, "mode": mode}
