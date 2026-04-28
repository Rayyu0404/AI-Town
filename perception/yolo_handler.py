# ================================================================
# perception/yolo_handler.py
# YOLO 偵測 + 語意轉換
# 偵測 Unreal Engine 傳來的畫面，判斷場景變化是否值得觸發推論
# ================================================================

from PIL import Image
from typing import Optional

# YOLO 為可選依賴，未安裝時使用 fallback 模式
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# 偵測到此類物件時視為「有意義的場景變化」
_MEANINGFUL_CLASSES = {
    "person", "chair", "cup", "laptop", "bottle",
    "handbag", "book", "cell phone", "bench", "dining table",
    # 可依 UE 場景擴充
}

# 中文物件名稱對照
_CLASS_ZH = {
    "person":       "人",
    "chair":        "椅子",
    "cup":          "杯子",
    "laptop":       "筆電",
    "bottle":       "瓶子",
    "handbag":      "手提包",
    "book":         "書",
    "cell phone":   "手機",
    "bench":        "長椅",
    "dining table": "桌子",
}


class YoloHandler:
    """
    YOLO 偵測包裝器。
    若 ultralytics 未安裝，所有方法仍可正常呼叫（回傳空結果）。
    """

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.4):
        """
        model_path : YOLO 模型權重路徑（預設使用 YOLOv8n）
        confidence : 偵測信心度閾值
        """
        self._model      = None
        self._confidence = confidence

        if _YOLO_AVAILABLE:
            try:
                self._model = _YOLO(model_path)
            except Exception as e:
                print(f"[YoloHandler] 模型載入失敗：{e}，使用 fallback 模式")

    # ── 主要介面 ─────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> list:
        """
        對圖片執行物件偵測。
        回傳：[{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
        YOLO 未安裝時回傳空 list。
        """
        if self._model is None:
            return []

        results = self._model(image, conf=self._confidence, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])]
                detections.append({
                    "class":      cls_name,
                    "confidence": float(box.conf[0]),
                    "bbox":       box.xyxy[0].tolist(),
                })
        return detections

    def is_meaningful(self, detections: list) -> bool:
        """
        判斷偵測結果是否包含值得觸發推論的物件。
        空偵測或無意義物件時回傳 False（跳過本幀）。
        """
        if not detections:
            return False
        return any(d["class"] in _MEANINGFUL_CLASSES for d in detections)

    def to_description(self, detections: list,
                       location: str = "") -> str:
        """
        將偵測結果轉為自然語言描述，供注入 STM 的 image_desc 欄位。
        location : 當前場景地點（增加語境）
        """
        if not detections:
            return ""

        meaningful = [
            d for d in detections
            if d["class"] in _MEANINGFUL_CLASSES
        ]
        if not meaningful:
            return ""

        # 計算各物件出現次數
        counts: dict = {}
        for d in meaningful:
            zh = _CLASS_ZH.get(d["class"], d["class"])
            counts[zh] = counts.get(zh, 0) + 1

        parts = []
        for name, cnt in counts.items():
            parts.append(f"{cnt}個{name}" if cnt > 1 else name)

        desc = "、".join(parts)
        if location:
            return f"{location}裡有{desc}。"
        return f"畫面中有{desc}。"

    def process(self, image: Image.Image,
                location: str = "") -> tuple:
        """
        一次完成偵測 + 判斷 + 語意轉換。
        回傳 (should_trigger: bool, description: str)
        """
        detections   = self.detect(image)
        meaningful   = self.is_meaningful(detections)
        description  = self.to_description(detections, location) if meaningful else ""
        return meaningful, description
