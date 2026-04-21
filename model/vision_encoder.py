# ================================================================
# model/vision_encoder.py
# 圖片前處理：將圖片轉為 Phi-3.5-Vision 所需的 tensor 格式
# ================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union
from PIL import Image

ImageLike = Union[str, Image.Image]


@dataclass
class VisionBatch:
    """Phi-3.5-Vision image_processor 的輸出封裝。"""
    pixel_values:   Any
    image_sizes:    Any = None
    num_img_tokens: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"pixel_values": self.pixel_values}
        if self.image_sizes    is not None: d["image_sizes"]    = self.image_sizes
        if self.num_img_tokens is not None: d["num_img_tokens"] = self.num_img_tokens
        return d


class VisionEncoder:
    """包裝 Phi-3.5-Vision 的 image_processor，準備圖片 tensor。"""

    def __init__(self, processor):
        self.processor       = processor
        self.image_processor = processor.image_processor

    @staticmethod
    def _load(img: ImageLike) -> Image.Image:
        return img.convert("RGB") if isinstance(img, Image.Image) \
               else Image.open(img).convert("RGB")

    def encode(self, images: Sequence[ImageLike],
               return_tensors: str = "pt") -> VisionBatch:
        pil = [self._load(im) for im in images]
        out = self.image_processor(pil, return_tensors=return_tensors)
        return VisionBatch(
            pixel_values   = out.get("pixel_values"),
            image_sizes    = out.get("image_sizes"),
            num_img_tokens = out.get("num_img_tokens"),
        )