# AI 模擬生活系統

多個 AI 角色在 Unreal Engine 虛擬環境中自主生活與互動，結合 HAM 認知記憶模型與 Phi-3.5-Vision 多模態模型，實現具備感知、思考、記憶與遺忘能力的 AI 代理人。

---

## 快速開始

```bash
pip install -r requirements.txt
python main.py
```

---

## 專案結構

```
project/
├── AI_Data/                    角色 JSON 檔案（A-E）
├── config/
│   ├── prompts.py              所有 prompt 集中管理
│   ├── world_config.py         角色對應表、情緒清單、記憶參數
│   ├── model_config.py         模型 ID、token 預算
│   └── action_list.py          合法行動與地點清單
├── core/                       核心邏輯（不依賴模型，可獨立測試）
│   ├── character.py            角色物件：屬性存取、時間表、關係管理
│   ├── stm.py                  STM 短期記憶
│   ├── ltm.py                  LTM 長期記憶（HAM 命題）
│   ├── confusion.py            困惑指數 C = w1U + w2K + w3S
│   └── memory_consolidation.py 睡眠濃縮流程
├── model/
│   ├── model_loader.py         模型單例載入
│   ├── vision_encoder.py       圖片前處理
│   ├── text_encoder.py         prompt 組合
│   ├── fusion_decoder.py       圖文融合推論
│   ├── prompt_builder.py       直覺／思考路徑 prompt 組裝
│   └── output_parser.py        解析 [ACTION][THOUGHT][HAM] 輸出
├── agent/
│   ├── agent.py                單一角色完整一輪流程
│   └── agent_manager.py        多角色排程 + AI 間對話傳遞
├── perception/
│   └── yolo_handler.py         YOLO 偵測 + 語意轉換
├── server/
│   └── ws_server.py            WebSocket 伺服器（對接 UE）
├── world/
│   └── world_clock.py          世界時間推進、睡眠觸發
├── utils/
│   ├── file_io.py              角色 JSON 讀寫
│   └── logger.py               每日 log 檔
├── logs/
├── main.py
├── requirements.txt
├── test.ipynb                  模組測試 notebook
└── PROJECT_DESCRIPTION.md      專題完整說明
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

## 核心機制

### 記憶雙層架構

**STM**：每輪對話一筆，每天上限 15 筆，睡覺時濃縮進 LTM

**LTM**：HAM 命題二元樹，每筆帶 strength 衰減值
- 衰減：`actual_decay = DECAY_RATE / (1 + access_count × 0.5)`
- strength < 0.2 時自動刪除

### 困惑指數

```
C = w1 × U + w2 × K + w3 × S
C < threshold → 直覺路徑（精簡 prompt，150 tokens）
C ≥ threshold → 思考路徑（完整 prompt，300 tokens）
```

### 模型輸出格式

```
[ACTION]: 對話:早安，David。
[THOUGHT]: David又來了，Amy保持平靜。
[HAM]
[{"subject":"Amy","relation":"服務","object":"David","location":"咖啡廳"}]
[/HAM]
```

---

## 開發進度

### 已完成
- [x] config/ — prompts、world_config、model_config、action_list
- [x] utils/ — file_io、logger
- [x] core/ — character、stm、ltm、confusion、memory_consolidation
- [x] model/ — model_loader、vision_encoder、text_encoder、fusion_decoder、prompt_builder
- [x] 角色 JSON（A-E 五個角色）

### 待完成
- [ ] model/output_parser.py
- [ ] agent/agent.py
- [ ] agent/agent_manager.py
- [ ] perception/yolo_handler.py
- [ ] world/world_clock.py
- [ ] server/ws_server.py
- [ ] main.py

---

## 參考文獻

- Anderson & Bower (1973). Human Associative Memory.
- Atkinson & Shiffrin (1968). Human memory: A proposed system.
- Ebbinghaus (1885). Über das Gedächtnis.
- Shenhav, Botvinick & Cohen (2013). The expected value of control. Neuron.
- Hardt, Nader & Nadel (2013). Decay happens. Trends in Cognitive Sciences.