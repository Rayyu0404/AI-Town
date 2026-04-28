# ================================================================
# config/world_config.py
# 世界全域設定，修改參數不需動其他程式碼
# ================================================================

import os

# ── 路徑 ────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_DATA_DIR   = os.path.join(BASE_DIR, "AI_Data")
LOG_DIR       = os.path.join(BASE_DIR, "logs")

# ── 角色代號 <-> 名字對應表 ──────────────────────────────────────
# 程式內部全部用代號，名字只在組 prompt 時透過這裡轉換
CHARACTER_NAMES = {
    "A": "Amy",
    "B": "Ben",
    "C": "Claire",
    "D": "David",
    "E": "Emma",
}

# 反查用（名字 -> 代號）
CHARACTER_CODES = {v: k for k, v in CHARACTER_NAMES.items()}

# ── STM 設定 ─────────────────────────────────────────────────────
STM_DEFAULT_CAPACITY = 15       # 一天最多幾筆 STM（超過觸發中途濃縮）

# ── LTM 衰減設定 ─────────────────────────────────────────────────
LTM_DECAY_RATE       = 0.05     # 每天睡覺時 strength 減少的比例
LTM_FORGET_THRESHOLD = 0.2      # strength 低於此值時在 prune 中刪除
LTM_FALLBACK_COUNT   = 3        # 模型解析失敗時，fallback 取前幾筆命題

# ── 情緒選項 ─────────────────────────────────────────────────────
# prompts.py 和 memory_consolidation.py 都從這裡取，統一管理
VALID_EMOTIONS = ["平靜", "開心", "緊張", "不安", "難過", "興奮", "困惑", "疲憊"]

# ── 世界時間設定 ─────────────────────────────────────────────────
SLEEP_ACTION       = "睡覺"     # 代表睡覺的 action 字串
DAY_START          = "06:00"    # 每天模擬起始時間
MINUTES_PER_TICK   = 60         # 每個時間段的分鐘數（1 小時）
FORCED_SLEEP_HOUR  = 4          # 凌晨 N 點強制入睡（04:00）
MAX_TICKS_PER_DAY  = 22         # 06:00 到 04:00 共 22 個小時

# ── 對話設定 ─────────────────────────────────────────────────────
MAX_CONVERSATION_TURNS_PER_SLOT = 5   # 每個時間段內對話輪次上限