# ================================================================
# model/fusion_decoder.py
# 圖文融合推論：將圖片 tensor 與文字 prompt 合併後送入模型
# ================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from config.model_config import (
    INTUITIVE_MAX_TOKENS, INTUITIVE_TEMPERATURE,
    DELIBERATE_MAX_TOKENS, DELIBERATE_TEMPERATURE,
)


@dataclass
class GenerationConfig:
    max_new_tokens: int   = 256
    temperature:    float = 0.0
    do_sample:      bool  = False

    @classmethod
    def intuitive(cls) -> "GenerationConfig":
        """直覺路徑的推論設定。"""
        return cls(
            max_new_tokens = INTUITIVE_MAX_TOKENS,
            temperature    = INTUITIVE_TEMPERATURE,
            do_sample      = False,
        )

    @classmethod
    def deliberate(cls) -> "GenerationConfig":
        """思考路徑的推論設定。"""
        return cls(
            max_new_tokens = DELIBERATE_MAX_TOKENS,
            temperature    = DELIBERATE_TEMPERATURE,
            do_sample      = False,
        )


class FusionDecoder:
    """圖文融合 + 文字生成。"""

    def __init__(self, model, processor, device: torch.device):
        self.model     = model
        self.processor = processor
        self.device    = device

    def fuse_inputs(self, text: str,
                    image_inputs: Dict[str, Any],
                    return_tensors: str = "pt") -> Dict[str, Any]:
        """將圖片 tensor 與文字 prompt 合併為模型輸入。"""
        return self.processor._convert_images_texts_to_inputs(
            image_inputs, text, return_tensors=return_tensors
        )

    @torch.inference_mode()
    def generate(self, fused_inputs: Dict[str, Any],
                 gen_cfg: Optional[GenerationConfig] = None) -> str:
        """
        執行推論，回傳純文字輸出（已去除 prompt 部分）。
        """
        if gen_cfg is None:
            gen_cfg = GenerationConfig()

        fused_inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in fused_inputs.items()
        }

        ids = self.model.generate(
            **fused_inputs,
            eos_token_id   = self.processor.tokenizer.eos_token_id,
            max_new_tokens = gen_cfg.max_new_tokens,
            temperature    = gen_cfg.temperature,
            do_sample      = gen_cfg.do_sample,
        )

        ids = ids[:, fused_inputs["input_ids"].shape[1]:]

        return self.processor.batch_decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]