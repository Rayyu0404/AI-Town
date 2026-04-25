# AI 模擬生活系統

多個 AI 角色在 Unreal Engine 虛擬環境中自主生活與互動，結合 HAM 認知記憶模型與 Phi-3.5-Vision 多模態模型，實現具備感知、思考、記憶與遺忘能力的 AI 代理人。

---

## 快速開始

```bash
pip install -r requirements.txt

# 快速驗證（不需 GPU）
python simulate.py --no-model

# 完整模擬（需要 GPU 與 Phi-3.5 模型）
python simulate.py

# 啟動 WebSocket 伺服器（與 UE 連接）
python main.py
```

---

## 核心設計（最新版）

### 兩段式時間軸

一天分為多個時間段（每 30 分鐘或 1 小時一個）。每個時間段分兩個階段：

```
Phase 1 — 執行：執行上一個時間段的決策（pending_action）
  - 對話行動：雙方配對後執行 N 輪對話（最多 MAX_CONVERSATION_TURNS_PER_SLOT 輪）
  - 睡覺行動：觸發 STM → LTM 濃縮
  - 其他行動：更新角色狀態

Phase 2 — 決策：所有角色針對下一時段做決策
  - 簡單思考（STM 路徑）：馬可夫鏈機率分布 → 選最高機率行動
  - 複雜思考（LTM 路徑）：完整記憶注入 → 模型自主推理

同步點：所有角色完成決策後 → 時鐘推進
```

### 簡單思考 vs 複雜思考

| | 簡單思考（STM 路徑）| 複雜思考（LTM 路徑）|
|---|---|---|
| 觸發條件 | 困惑指數 C < 0.5 | 困惑指數 C ≥ 0.5 |
| 決策方式 | **馬可夫鏈**：根據 STM 歷史行動計算轉移機率 → 選最高機率 | **模型自主推理**：注入完整長短期記憶，讓模型自己決定 |
| 輸出 | 行動 + 機率分布（無 THOUGHT/HAM）| 行動 + 內心想法 + HAM 命題 |
| 模型呼叫 | 不需要（純統計）| 需要（完整推理） |

### 記憶儲存（不變）

**STM（短期記憶）**：每輪 turn 紀錄，每天上限 15 筆，睡覺時濃縮
- 簡單思考路徑：記錄行動與機率分布（無 HAM 命題）
- 複雜思考路徑：完整紀錄（含 HAM 命題）

**LTM（長期記憶）**：HAM 命題二元樹
- 衰減：`actual_decay = DECAY_RATE / (1 + access_count × 0.5)`
- strength < 0.2 時自動刪除

### 對話機制

```
12:30 時間段（決策階段）：
  A 和 B 在同地點，各自決策 → 都選「對話」→ 配對
  → pending_action = "對話:Amy" / "對話:Ben"

13:00 時間段（執行階段）：
  A 和 B 開始對話，每輪一來一往，最多 N 輪
  執行完後進入決策階段：
  → A 決定繼續 / 離開
  → B 決定繼續 / 其他

14:00 時間段：
  若 A 離開 → 對話結束，B 重新決策
```

---

## 專案結構

```
NEW/
├── AI_Data/                    角色 JSON 檔案（A-E）
├── config/
│   ├── prompts.py              所有 prompt 模板（直覺/思考/睡眠濃縮）
│   ├── world_config.py         角色對應表、記憶參數、時間與對話設定
│   ├── model_config.py         模型 ID、token 預算
│   └── action_list.py          合法行動與地點清單
├── core/
│   ├── character.py            角色物件（含 pending_action 等執行期狀態）
│   ├── stm.py                  STM 短期記憶（turn 紀錄）
│   ├── ltm.py                  LTM 長期記憶（HAM 命題）
│   ├── confusion.py            困惑指數 C = w1U + w2K + w3S
│   ├── markov.py               馬可夫鏈決策引擎（簡單思考路徑）
│   └── memory_consolidation.py 睡眠濃縮流程（STM → LTM）
├── model/
│   ├── model_loader.py         Phi-3.5-Vision 單例載入
│   ├── vision_encoder.py       圖片前處理
│   ├── text_encoder.py         prompt 組合
│   ├── fusion_decoder.py       圖文融合推論
│   ├── prompt_builder.py       直覺／思考路徑 prompt 組裝
│   └── output_parser.py        解析 [ACTION][THOUGHT][HAM] 輸出
├── agent/
│   ├── agent.py                decide()（決策）+ step()（對話執行）+ sleep()
│   └── agent_manager.py        兩段式時間軸管理 + 對話配對 + 睡眠協調
├── world/
│   └── world_clock.py          世界時鐘、時間段推進、時間表判斷
├── perception/
│   └── yolo_handler.py         YOLO 偵測 + 語意轉換
├── server/
│   └── ws_server.py            WebSocket 伺服器（對接 UE）
├── utils/
│   ├── file_io.py              角色 JSON 讀寫
│   └── logger.py               每日 log 檔
├── main.py                     程式進入點（伺服器 / 離線模擬）
├── simulate.py                 完整測試 + HTML 報告生成
└── simulate_report.html        最近一次測試的視覺化報告
```

---

## 角色設定

| 代號 | 名字 | 職業 | 困惑閾值 | 感情狀態 |
|------|------|------|---------|---------|
| A | Amy | 咖啡師 | 0.45 | 暗戀 David，被 Ben 追求 |
| B | Ben | 超市員工 | 0.60 | 喜歡 Amy（Amy 在迴避） |
| C | Claire | 辦公室員工 | 0.50 | 喜歡 Emma，未表白 |
| D | David | 公司老闆 | 0.55 | 對 Emma 有好感，未深究 |
| E | Emma | 餐廳員工 | 0.50 | 喜歡 Claire，還在觀察 |

---

## 困惑指數

```
C = w1 × U（不確定性）+ w2 × K（衝突程度）+ w3 × S（驚訝程度）

C < threshold  →  簡單思考（馬可夫鏈，不呼叫模型）
C ≥ threshold  →  複雜思考（完整 LTM 推理，呼叫 Phi 模型）
```

---

## 主要設定參數（config/world_config.py）

| 參數 | 預設值 | 說明 |
|------|-------|------|
| STM_DEFAULT_CAPACITY | 15 | 短期記憶一天上限筆數 |
| LTM_DECAY_RATE | 0.05 | 每天衰減率 |
| LTM_FORGET_THRESHOLD | 0.2 | 遺忘閾值 |
| MINUTES_PER_TICK | 30 | 每個時間段分鐘數 |
| MAX_CONVERSATION_TURNS_PER_SLOT | 5 | 每時間段對話輪次上限 |
| CONFUSION_THRESHOLD | 0.5 | 觸發複雜思考的困惑指數閾值 |

---

## 測試指令

```bash
# 快速流程驗證（假模型，無 GPU）
python simulate.py --no-model

# 指定時間段數量
python simulate.py --no-model --slots 6

# 只驗證 Phi 模型是否可下載
python simulate.py --check-only

# 完整測試（Phi 模型 + 兩段式模擬）
python simulate.py
```

測試完成後開啟 `simulate_report.html` 查看視覺化報告。

---

## 參考文獻

- Anderson & Bower (1973). Human Associative Memory.
- Atkinson & Shiffrin (1968). Human memory: A proposed system.
- Ebbinghaus (1885). Über das Gedächtnis.
- Shenhav, Botvinick & Cohen (2013). The expected value of control. Neuron.
- Hardt, Nader & Nadel (2013). Decay happens. Trends in Cognitive Sciences.
