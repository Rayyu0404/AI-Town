# ================================================================
# model/text_encoder.py
# 對話 prompt 建構：使用 apply_chat_template 組合 prompt
# 並在需要時插入圖片佔位符 <|image_N|>
# ================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TextPrompt:
    prompt: str


class TextEncoder:
    """建立含圖片佔位符的 chat prompt。"""

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def build_prompt(self, user_text: str,
                     num_images: int = 0,
                     system_text: Optional[str] = None) -> TextPrompt:
        """
        組合完整的 chat prompt。
        user_text  : 主要 prompt 文字
        num_images : 圖片數量，每張插入一個 <|image_N|> 佔位符
        system_text: 可選的 system prompt
        """
        # 每張圖插入一個佔位符
        placeholder = "".join(
            f"<|image_{i}|>\n" for i in range(1, num_images + 1)
        )

        messages: List[dict] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({
            "role": "user",
            "content": placeholder + user_text
        })

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return TextPrompt(prompt=prompt)