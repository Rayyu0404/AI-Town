# ================================================================
# config/model_config.py
# Phi-3.5-Vision 模型相關設定
# ================================================================

# ── 模型設定 ─────────────────────────────────────────────────────
MODEL_ID              = "microsoft/Phi-3.5-vision-instruct"
MODEL_NUM_CROPS       = 4

# ── 推論參數 ─────────────────────────────────────────────────────
# 直覺路徑（快速、省 token）
INTUITIVE_MAX_TOKENS  = 150
INTUITIVE_TEMPERATURE = 0.0

# 思考路徑（完整推理）
DELIBERATE_MAX_TOKENS = 300
DELIBERATE_TEMPERATURE = 0.0

# ── 睡眠濃縮各步驟 token 上限（任務單純，不需要太多 token）───────
CONSOLIDATE_SELECT_MAX_TOKENS   = 100   # select_ltm：JSON list
CONSOLIDATE_SUMMARY_MAX_TOKENS  = 60    # ltm_summary：1-2 句話
CONSOLIDATE_RELATION_MAX_TOKENS = 60    # update_relationship：1-2 句話
CONSOLIDATE_EMOTION_MAX_TOKENS  = 10    # infer_emotion：單一詞彙

# ── Prompt token 預算（避免爆 context window）────────────────────
# Phi-3.5 context window = 128K tokens
# 保守預留：模型輸出 300 + prompt 本身結構 200 + 圖片描述 200
STM_TOKEN_BUDGET      = 1500    # 整個 STM 區塊最多幾個 token
LTM_TOKEN_BUDGET      = 500     # LTM 注入區塊最多幾個 token