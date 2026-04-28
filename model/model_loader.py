# ================================================================
# model/model_loader.py
# Phi-3.5-Vision 模型載入
# 模型只載入一次，所有角色的 agent 共用同一個實例
# ================================================================

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from config.model_config import MODEL_ID, MODEL_NUM_CROPS
from model.vision_encoder import VisionEncoder
from model.text_encoder import TextEncoder
from model.fusion_decoder import FusionDecoder, GenerationConfig


def pick_device() -> torch.device:
    """選擇最佳運算裝置。"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelLoader:
    """
    Phi-3.5-Vision 模型的單例載入器。
    使用方式：
        loader = ModelLoader()
        loader.load()
        # 之後透過 loader.vision / loader.text / loader.fusion 使用
    """

    def __init__(self):
        self.device    = pick_device()
        self.model     = None
        self.processor = None
        self.vision    = None   # VisionEncoder
        self.text      = None   # TextEncoder
        self.fusion    = None   # FusionDecoder
        self._loaded   = False

    def load(self):
        """
        載入模型與 processor，初始化三個功能元件。
        初次執行會下載模型（約 8GB），之後從快取載入。
        """
        if self._loaded:
            return

        print(f"[ModelLoader] 載入模型：{MODEL_ID}")
        print(f"[ModelLoader] 裝置：{self.device}")

        # 載入模型
        # _attn_implementation="eager" 確保非 CUDA 環境也能執行
        # 不使用 device_map="auto"，改為統一用 .to(self.device) 管理裝置
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
        )
        self.model = self.model.to(self.device)

        # 載入 processor
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            num_crops=MODEL_NUM_CROPS,
        )

        # 初始化三個功能元件
        self.vision  = VisionEncoder(self.processor)
        self.text    = TextEncoder(self.processor)
        self.fusion  = FusionDecoder(self.model, self.processor, self.device)

        self._loaded = True
        print("[ModelLoader] 模型載入完成。")

    def is_loaded(self) -> bool:
        return self._loaded

    def make_model_fn(self, max_new_tokens: int = 256,
                      temperature: float = 0.0) -> callable:
        """
        回傳一個純文字推論函式（無圖片）。
        供 memory_consolidation 等只需要文字推論的場合使用。
        signature: model_fn(prompt: str) -> str
        """
        if not self._loaded:
            raise RuntimeError("請先呼叫 load() 載入模型。")

        def model_fn(prompt: str) -> str:
            text_prompt = self.text.build_prompt(prompt, num_images=0)
            fused = self.fusion.fuse_inputs(
                text=text_prompt.prompt,
                image_inputs={},
            )
            fused["use_cache"] = False  # 睡眠濃縮不需要 KV cache，節省記憶體
            return self.fusion.generate(
                fused_inputs=fused,
                gen_cfg=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,
                ),
            )

        return model_fn