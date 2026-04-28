# 專題描述文件

---

## 專題名稱

**基於 HAM 認知模型與多模態大語言模型的 AI 角色自主生活模擬系統**

---

## 一句話摘要

本專題建構一套讓多個 AI 角色在 Unreal Engine 虛擬環境中自主生活、互動並形成長期記憶的完整系統，結合神經科學的人類記憶理論與現代多模態模型，實現具備感知、思考、記憶與遺忘能力的 AI 代理人。

---

## 系統概覽

本系統讓五個具備獨立個性、職業與人際關係的 AI 角色，在共同的虛擬世界中自主運作。每個角色透過 Unreal Engine 接收來自虛擬環境的畫面與事件，經由多模態模型做出決策，並將重要的生活經歷逐漸累積為長期記憶，同時會因時間流逝與使用頻率的差異而產生遺忘，模擬真實人腦的記憶機制。

---

## 理論基礎

### Human Associative Memory（HAM）模型

Anderson & Bower（1973）提出的 HAM 模型認為，人類長期記憶以「命題」的方式組織，每個記憶單元由**主詞（Subject）、關係（Relation）、受詞（Object）**三個節點構成，並可附加地點與時間資訊。本系統以此結構作為角色長期記憶（LTM）的儲存格式：

```
示例命題：Amy 遇見 David（咖啡廳，早上）
主詞: Amy  |  關係: 遇見  |  受詞: David  |  地點: 咖啡廳  |  時間: 早上
```

每筆命題都帶有 `strength`（強度）值，反映該記憶的重要程度與鮮明程度。

### Atkinson-Shiffrin 多重記憶模型

記憶分為兩層：
- **STM（短期記憶）**：每天最多 15 筆，記錄當天的場景、事件、決策
- **LTM（長期記憶）**：從 STM 精選後長期保存，具有衰減與遺忘機制

每天角色「睡覺」時，由 AI 模型判斷今日 STM 中哪些事件值得永久記憶，模擬人腦在睡眠期間進行的記憶鞏固過程（Memory Consolidation）。

### Ebbinghaus 遺忘曲線與主動遺忘

LTM 中每筆命題的強度每天衰減，但越常被「想起來」的記憶衰減越慢：

```
actual_decay = DECAY_RATE / (1 + access_count × 0.5)

access_count = 0（從未被想起）→ decay = 0.05（快速遺忘）
access_count = 2（被想起兩次）→ decay = 0.025（慢速遺忘）
access_count = 10（常被想起）→ decay = 0.008（幾乎不忘）
```

強度低於 0.2 的記憶會被自動刪除，模擬「徹底遺忘」。

### Expected Value of Control（EVC）困惑指數

參考 Shenhav、Botvinick & Cohen（2013）的前扣帶皮層（ACC）功能理論。人腦會根據當前情境的複雜程度，動態分配認知資源——簡單情境走直覺，複雜情境才深思。

本系統計算**困惑指數 C = w₁U + w₂K + w₃S**：

| 分量 | 意義 | 計算方式 |
|------|------|---------|
| **U**（不確定性） | 角色不知道該做什麼 | 根據候選行動的評分差距計算 |
| **K**（衝突） | 場景資訊相互矛盾 | 關鍵字對衝突偵測（如「陌生人」vs「認識的人」）|
| **S**（驚訝） | 意料之外的事發生 | 場景新穎度 + 行動停滯懲罰 |

C 值超過閾值 → 深度思考路徑（deliberate）；低於閾值 → 直覺快速決策（intuitive）。

---

## 五個角色設計

| 代號 | 名字 | 職業 | 個性關鍵字 | 困惑閾值 | 個性說明 |
|------|------|------|-----------|---------|---------|
| A | Amy | 咖啡師 | 溫柔、壓抑、細心 | 0.45（敏感） | 容易陷入深思，對細節很在意 |
| B | Ben | 超市員工 | 開朗、粗神經、不安全感 | 0.60（大條） | 大多走直覺，遲鈍不易察覺微妙變化 |
| C | Claire | 辦公室員工 | 直率、謹慎、可靠 | 0.50 | 均衡型，直接表達自己的想法 |
| D | David | 公司老闆 | 穩重、內斂、投入工作 | 0.55 | 不輕易改變計畫，對工作高度專注 |
| E | Emma | 餐廳員工 | 活潑、敏感、隨性 | 0.50 | 情緒豐富，容易被氛圍影響 |

**人際關係網路（設計有戲劇性張力）：**
- Ben → Amy（Ben 追求，Amy 迴避）
- Amy ↔ Emma（閨蜜，互相支持）
- Claire → Emma（單戀，Emma 尚未察覺）
- David → Emma（暗中有好感，不公開）
- Claire ↔ David（職場關係，直來直往）

---

## 技術架構

### 整體架構分層

```
┌─────────────────────────────────────────────────────┐
│  Unreal Engine（虛擬世界）                            │
└──────────────────┬──────────────────────────────────┘
                   │  WebSocket（畫面 + 事件）
┌──────────────────▼──────────────────────────────────┐
│  server / perception 層                               │
│  ws_server.py + yolo_handler.py                      │
│  接收畫面 → YOLO 偵測 → 轉換為中文語意描述              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  agent 層                                             │
│  agent_manager.py（協調 5 個角色）                    │
│  agent.py（單一角色推論流程）                          │
└──────┬───────────────────────────────────────────────┘
       │
  ┌────▼──────────┐        ┌─────────────────────────┐
  │  core 層       │        │  model 層                │
  │  character.py │◄──────►│  model_loader.py         │
  │  stm.py       │        │  text_encoder.py         │
  │  ltm.py       │        │  vision_encoder.py       │
  │  confusion.py │        │  fusion_decoder.py       │
  │  consolidation│        │  prompt_builder.py       │
  └───────────────┘        │  output_parser.py        │
                           └─────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│  config 層（全域設定、Prompt 模板、模型參數）           │
│  utils 層（檔案讀寫、log）                             │
│  world 層（虛擬時鐘）                                 │
└─────────────────────────────────────────────────────┘
```

**核心設計原則：下層不依賴上層。**`core/` 完全不碰模型，只透過 callback 接收結果，這讓核心邏輯可以獨立用 mock 測試，不需要每次都跑 GPU。

---

## 資料流程（一輪推論）

```
Unreal Engine 傳入：場景描述 + 畫面 + 事件文字
         ↓
YOLO 偵測畫面 → 轉換為中文語意（"咖啡廳裡有2個人、1個杯子。"）
         ↓
【困惑指數計算】
  U（不確定）+ K（衝突）+ S（驚訝）→ C 值
  C < 閾值 → 直覺路徑
  C ≥ 閾值 → 思考路徑
         ↓
【Prompt 組裝】
  直覺路徑：一句話個性 + 最近 3 筆 STM + LTM 摘要
  思考路徑：完整個性 + 習慣 + 全部 STM + 相關 LTM + 完整關係描述
         ↓
Phi-3.5-Vision 模型推論（本地 GPU）
         ↓
【解析輸出】
  [ACTION] 前往:咖啡廳     ← 角色行動
  [THOUGHT] 需要去補貨...  ← 內心想法
  [HAM] ["Amy 前往 咖啡廳（早上）", ...]  ← 本輪記憶命題
         ↓
行動 → WebSocket → Unreal Engine 執行動畫
HAM 命題 → 寫入 STM
         ↓
（角色說出「睡覺」或 STM 滿 15 筆）
【睡眠濃縮】
  模型篩選 STM → 重要命題寫入 LTM
  更新 LTM 摘要、關係描述、今日情緒
  衰減 LTM strength → 低於 0.2 的命題自動刪除
  清空 STM → day + 1
  寫回磁碟
```

---

## 模組詳解

### Config 層（`config/`）— 全局設定中心

所有可調整的參數集中在此，避免散落在程式各處。

**`world_config.py`** — 世界規則

| 常數 | 值 | 用途說明 |
|------|-----|---------|
| `CHARACTER_NAMES` | A→Amy, B→Ben... | 用代碼不用全名，推論時省 token |
| `STM_CAPACITY` | 15 筆 | 約等於真實一天的互動量 |
| `LTM_DECAY_RATE` | 0.05（5%） | 每天睡眠衰減，模仿遺忘曲線 |
| `LTM_FORGET_THRESHOLD` | 0.2 | 低於此值視為徹底遺忘 |
| `VALID_EMOTIONS` | 8 種中文情緒 | 限定詞彙，防止模型造出不合格的詞 |
| `SLEEP_ACTION` | "睡覺" | 偵測到此動詞就觸發睡眠濃縮流程 |

**`model_config.py`** — 推論參數

```
模型：microsoft/Phi-3.5-vision-instruct（4.15B，支援圖片+文字）
```

| 路徑 | 最多 token | 溫度 | 說明 |
|------|-----------|------|------|
| 直覺 | 150 | 0.0 | 快速，節省 GPU；確定性輸出 |
| 思考 | 300 | 0.0 | 完整推理空間；仍需格式穩定 |

溫度設為 0.0（greedy decoding），確保每次相同輸入得到相同格式的輸出，parser 不易出錯。

睡眠濃縮各步驟的 token 預算（針對 RTX 4060 Laptop 的效能優化）：

| 步驟 | Token 上限 | 說明 |
|------|-----------|------|
| 篩選重要命題 | 100 | 輸出 JSON list，不需長文 |
| 生成 LTM 摘要 | 60 | 1-2 句話 |
| 更新關係描述 | 60 | 1-2 句話 |
| 推斷今日情緒 | 10 | 單一詞彙，如「開心」|

> **為什麼要這樣限制**：RTX 4060 Laptop 跑 Phi-3.5 約 1.8 tokens/秒。若每步用預設 256 token，5 個角色睡眠需要約 47 分鐘；改用以上預算後縮短至 5-8 分鐘。

**`prompts.py`** — 所有 Prompt 模板集中管理

全部用繁體中文撰寫，7 個模板函式：
- `prompt_intuitive()` — 直覺路徑推論
- `prompt_deliberate()` — 思考路徑推論
- `prompt_select_ltm()` — 請模型判斷哪些記憶值得長期保存
- `prompt_ltm_summary()` — 請模型壓縮所有 LTM 為摘要
- `prompt_update_relationship()` — 根據今天對話更新關係描述
- `prompt_infer_emotion()` — 從今天事件推斷當前情緒
- `prompt_generate_schedule()` — 產生排程計畫

**`action_list.py`** — 合法行動與地點

約束模型只能輸出已定義的動詞與地點，避免亂造詞。
- 動詞：前往、回家、對話、工作、休息、吃飯、購物、散步、睡覺等 12 種
- 地點：公寓、咖啡廳、超市、辦公室、餐廳、公園等 16 個

---

### Core 層（`core/`）— 純邏輯，無模型依賴

這層完全不 import 任何模型相關的東西，所有功能都可以用假資料（mock）獨立測試。

**`character.py`** — 角色狀態容器

從 JSON 載入後封裝成物件，分為靜態屬性和動態屬性：

```
靜態（不會改變）：code, name, role, gender, age, personality, habit
動態（會隨時間更新）：emotion, day, current_location, current_action
每日清零：today_actions（今天做了什麼）
```

關係設計有兩層：
- `initial`（初始描述）：角色「天生設定」，不可改變
- `summary`（動態摘要）：根據實際互動不斷更新，記錄關係的演變

**`stm.py`** — 短期記憶

每筆 STM turn 的結構：
```
turn_id: "D001_T003"（第1天第3輪）
scene:      場景描述（"公園，下午，晴天"）
image_desc: YOLO 語意化結果（"公園裡有2個人"）
input_text: 接收到的對話或事件
action:     本輪決策行動（"前往:咖啡廳"）
ham_propositions: 本輪抽取的記憶命題 list
```

**`ltm.py`** — 長期記憶（HAM 命題）

每筆 LTM 命題的結構：
```
id:           "L001"（唯一 ID，即使 prune 後也不重複）
subject:      "Amy"
relation:     "遇見"
object:       "David"
location:     "咖啡廳"（可選）
time:         "早上"（可選）
strength:     0.9（0.0-1.0，越高越難忘）
access_count: 2（被讀取次數，影響衰減速度）
encoded_day:  3（第幾天存入）
```

文字輸出格式（注入 prompt 時）：`"- Amy 遇見 David（咖啡廳，早上）"`

**`confusion.py`** — 困惑指數計算

```python
C = w1 * U + w2 * K + w3 * S

U = 1.0 - (最高分行動 - 次高分行動) * 2
    # 越難選擇，U 越高
K = 關鍵字衝突偵測（0 到 0.6）
    # 如同時出現「陌生人」和「老朋友」
S = 場景新穎度（0.7）+ 行動停滯懲罰（0.2）
    # 一直重複同樣行動會被懲罰
```

每個角色的 w1、w2、w3 不同，體現個性差異。

**`memory_consolidation.py`** — 睡眠濃縮流程（7 步驟）

```
1. 模型從今日 STM 命題中篩選重要的
2. 重要命題寫入 LTM
3. 模型生成 LTM 壓縮摘要（1-2 句話）
4. 對今天互動過的每個角色，模型更新關係描述
5. 模型從今天發生的事推斷當前情緒
6. 所有 LTM 命題依公式衰減，低分者刪除
7. 清空 STM，day + 1
```

這個模組接受 `make_model_fn`（函式工廠）而非直接的模型物件，原因是保持 core 層獨立，不直接依賴 model 層。

---

### Model 層（`model/`）— Phi-3.5-Vision 封裝

**`model_loader.py`** — 模型載入器

- 自動選擇最佳裝置：CUDA（NVIDIA GPU）> MPS（Apple Silicon）> CPU
- Singleton 設計：整個系統只載入一次，不重複佔用記憶體
- 提供 `make_model_fn(max_new_tokens=N)`：回傳一個純文字推論的 callable，供 consolidation 使用

**`text_encoder.py`** — 文字 Prompt 轉換

將 prompt 文字轉換為 Phi-3.5 的 chat 格式，並在有圖片時插入 `<|image_1|>` 等佔位符。

**`vision_encoder.py`** — 圖片前處理

將 PIL 圖片轉換為模型所需的 `pixel_values` 張量（tensor）格式。

**`fusion_decoder.py`** — 圖文融合推論

把文字和圖片 tensor 合併後，呼叫 `model.generate()`，解碼後回傳純文字（只回傳新生成的部分，不包含 prompt）。

**`prompt_builder.py`** — 依路徑組裝 Prompt

直覺路徑（輸入資訊少，速度快）：
```
系統說明 + 一句話個性 + 最近 3 筆 STM + LTM 摘要 + 初始關係
```

思考路徑（輸入資訊完整，較慢）：
```
系統說明 + 完整個性 + 習慣 + 全部 STM（超過 1500 token 從最舊的截掉）
+ 相關 LTM 命題（關鍵字比對）+ 完整關係摘要 + 今日行動清單
```

**`output_parser.py`** — 解析模型輸出

模型預期輸出格式：
```
[ACTION] 前往:咖啡廳
[THOUGHT] 需要去補貨，趁現在比較空閒。
[HAM] ["Amy 前往 咖啡廳（早上）", "Amy 需要 補貨"]
[/HAM]
```

Parser 用 regex 提取三個區塊，並將 "前往:咖啡廳" 拆成 `verb="前往"`、`target="咖啡廳"`。

---

### Agent 層（`agent/`）— 推論協調

**`agent.py`** — 單角色一輪完整推論

整合所有模組，按順序執行 8 個步驟：
```
1. 計算困惑指數 → 決定推論路徑
2. 組裝 Prompt
3. 圖片前處理（有圖才跑）
4. 圖文融合
5. 模型推論
6. 解析輸出
7. 更新角色狀態（位置、行動）
8. 寫入 STM
```

回傳結果包含：action（行動指令）、thought（內心想法）、ham（命題）、mode（推論路徑）、should_sleep（是否該睡覺）。

**`agent_manager.py`** — 5 角色協調器

管理所有角色的推論排程，並處理兩個重要情境：

1. **AI 間對話傳遞**：當 A 說話對象是 B，manager 會自動把 A 說的話傳給 B，讓 B 也執行一輪推論並回應。用 `_forwarded=True` 旗標防止 A→B→A 無限循環。

2. **睡眠觸發**：偵測到 `should_sleep=True` 或 STM 滿 15 筆時，自動執行睡眠濃縮，濃縮完儲存到磁碟。全部角色都睡覺後，虛擬時鐘推進到下一天。

---

### Perception 層（`perception/`）— 視覺感知

**`yolo_handler.py`** — YOLO 物件偵測

YOLO v8n（最輕量版本）偵測畫面後，只保留 10 類「有意義的物件」：人、椅子、杯子、筆電、瓶子、手提包、書、手機、長椅、桌子。

偵測結果不是傳數字給模型，而是先轉換成**中文語意描述**再注入 prompt：
```
原始偵測：[{class:"person", conf:0.9}, {class:"cup", conf:0.8}]
語意描述："咖啡廳裡有2個人、1個杯子。"
```

> 這樣做的原因：語言模型理解「2個人、1個杯子」比理解 bounding box 座標容易得多，推論品質更高。

---

### Server 層（`server/`）— UE 通訊橋接

**`ws_server.py`** — WebSocket 伺服器

UE 傳來的訊息格式：
```json
{
  "character": "A",
  "scene": "咖啡廳，早上，晴天",
  "image_b64": "（Base64 編碼的截圖）",
  "input_text": "Ben說：早安！",
  "target": "B"
}
```

Python 回傳的格式：
```json
{
  "character": "A",
  "action": "對話:早安，你今天看起來很精神",
  "thought": "Ben 似乎心情不錯...",
  "mode": "intuitive"
}
```

圖片接收後立刻做 Base64 解碼 → PIL 格式 → YOLO 偵測 → 語意文字，然後才傳給 agent。

---

### World 層（`world/`）— 虛擬時間

**`world_clock.py`** — 世界時鐘

- 時間以「午夜後分鐘數」儲存（整數）
- 每 tick 前進 30 分鐘（可設定）
- 負責觸發排程行動（起床 07:00、吃飯 12:00 等固定時間點）
- 所有角色都睡覺後才推進到隔天

---

### Utils 層（`utils/`）— 工具

**`file_io.py`** — 角色狀態的讀寫

設計策略：**啟動時全部載入記憶體，只有睡眠時才寫回磁碟**。

好處：推論過程中不需要讀寫磁碟，速度快；一天內最多只損失一天的進度。

**`logger.py`** — 每日 log

每天產生一個 log 檔（`logs/YYYYMMDD.log`），記錄每輪的 C 值、行動決策、睡眠濃縮結果。

---

## 角色資料結構（`AI_Data/A_init.json` 為例）

```json
{
  "name": "Amy",
  "name_code": "A",
  "role": "咖啡師",
  "personality_short": "（一句話版，直覺路徑 prompt 用）",
  "personality": "（完整版，思考路徑 prompt 用）",
  "habit": "每天早上先做一杯手沖咖啡...",
  "relationships": {
    "B": {
      "initial": "認識但關係微妙，對方似乎有好感",
      "summary": "（每次睡眠後由模型更新）"
    }
  },
  "emotion": "平靜",
  "stm": { "capacity": 15, "turns": [] },
  "ltm": { "ltm_summary": "", "propositions": [] },
  "state": {
    "day": 1,
    "current_location": "公寓",
    "current_action": "休息",
    "today_actions": []
  },
  "schedule": {
    "slots": [
      { "time": "07:00", "action": "起床", "location": "公寓", "type": "fixed" },
      { "time": "08:00", "action": "前往", "location": "咖啡廳", "type": "fixed" }
    ]
  },
  "confusion_weights": { "w1": 0.4, "w2": 0.3, "w3": 0.3, "threshold": 0.45 }
}
```

`initial` 是角色「天生設定」，`summary` 是「活生生的記憶」，兩者共存讓角色既有基礎個性，又能因互動而改變對他人的看法。

---

## 模組結構

```
project/
├── AI_Data/                    # 角色 JSON（A-E 代號）
├── config/
│   ├── prompts.py              # 所有 prompt 模板（繁體中文）
│   ├── world_config.py         # 世界規則、情緒清單、衰減參數
│   ├── model_config.py         # 模型參數、token 預算
│   └── action_list.py          # 合法行動動詞與地點
├── core/                       # 純邏輯，不依賴模型，可獨立測試
│   ├── character.py            # 角色狀態封裝
│   ├── stm.py                  # STM 短期記憶管理
│   ├── ltm.py                  # LTM 長期記憶（HAM 命題 + 衰減）
│   ├── confusion.py            # 困惑指數 C = w₁U + w₂K + w₃S
│   └── memory_consolidation.py # 睡眠濃縮 7 步驟流水線
├── model/
│   ├── model_loader.py         # Phi-3.5 載入（Singleton）
│   ├── vision_encoder.py       # 圖片 → 張量
│   ├── text_encoder.py         # Prompt → chat 格式
│   ├── fusion_decoder.py       # 圖文融合 + 生成
│   ├── prompt_builder.py       # 組裝直覺/思考路徑 prompt
│   └── output_parser.py        # 解析 ACTION / THOUGHT / HAM
├── agent/
│   ├── agent.py                # 單一角色完整一輪推論
│   └── agent_manager.py        # 多角色協調 + AI 間對話傳遞
├── perception/
│   └── yolo_handler.py         # YOLO 偵測 + 語意轉換
├── server/
│   └── ws_server.py            # WebSocket 橋接（UE ↔ Python）
├── world/
│   └── world_clock.py          # 虛擬時鐘 + 排程觸發
├── utils/
│   ├── file_io.py              # JSON 讀寫（睡眠時才寫磁碟）
│   ├── logger.py               # 每日 log
│   └── test_reset.py           # 測試用狀態重置
├── logs/                       # 日誌目錄
├── simulate.py                 # 整合測試（含 MockLoader，不需 GPU）
├── test.ipynb                  # 模組單元測試
└── main.py                     # 程式進入點（正常模式 / demo 模式）
```

---

## 關鍵設計決策

| 決策 | 為什麼這樣做 |
|------|------------|
| HAM 三元組記憶（不用向量） | 不需要額外的 embedding 模型；命題可直接注入 context；人類也看得懂 |
| temperature = 0.0 | 輸出格式穩定，parser 不易出錯；同樣輸入可重現 |
| core 層不依賴 model 層 | 可用 mock 測試所有記憶邏輯，不需要 GPU |
| YOLO 結果先語意化再給模型 | 模型理解「2個人」比理解座標數字容易得多 |
| 狀態在 RAM，睡眠才存磁碟 | 推論速度快；一天內崩潰最多損失一天 |
| 每步驟獨立 token 預算 | 情緒推斷只需 10 tokens，省 GPU 時間（5 角色睡眠從 47 分鐘降至 8 分鐘）|
| 對話轉發用 `_forwarded` 旗標 | 防止 A→B→A 無限對話迴圈 |
| 兩路徑推論（直覺/思考） | 模仿人腦的認知資源分配，平衡速度與品質 |
