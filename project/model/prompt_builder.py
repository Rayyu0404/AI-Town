# ================================================================
# model/prompt_builder.py
# 依照思考路徑組裝完整 prompt
# 直覺路徑：精簡版，省 token
# 思考路徑：完整版，帶入所有記憶與個性細節
# ================================================================

from config.prompts import prompt_intuitive, prompt_deliberate
from config.model_config import STM_TOKEN_BUDGET, LTM_TOKEN_BUDGET


class PromptBuilder:
    """
    根據角色狀態與思考模式組裝 prompt。
    character : Character 物件
    stm       : STM 物件
    ltm       : LTM 物件
    """

    def __init__(self, character, stm, ltm):
        self.character = character
        self.stm       = stm
        self.ltm       = ltm

    def build(self, scene: str, mode: str,
              target_code: str = None) -> str:
        """
        組裝完整 prompt。
        scene       : 當前場景描述
        mode        : "intuitive" 或 "deliberate"
        target_code : 當前互動對象的代號（可為 None）
        """
        if mode == "intuitive":
            return self._build_intuitive(scene, target_code)
        return self._build_deliberate(scene, target_code)

    # ── 直覺路徑 ─────────────────────────────────────────────────

    def _build_intuitive(self, scene: str,
                          target_code: str = None) -> str:
        """
        直覺路徑：
        - personality_short（一句話）
        - 最近 3 筆 STM
        - LTM 壓縮摘要（一句話）
        - 當前對象的關係 initial 描述
        """
        char = self.character

        # 只取最近 3 筆 STM
        stm_text = _format_stm(self.stm.get_recent(3))

        # LTM 摘要
        ltm_summary = self.ltm.get_summary()

        # 關係描述（只帶 initial，不帶 summary）
        rel_text = ""
        if target_code:
            rel_text = char.get_relationship_text(
                target_code, include_summary=False
            )

        return prompt_intuitive(
            character_name    = char.name,
            personality_short = char.get_personality(short=True),
            emotion           = char.emotion,
            relationship_text = rel_text,
            stm_text          = stm_text,
            ltm_summary       = ltm_summary,
            scene             = scene,
        )

    # ── 思考路徑 ─────────────────────────────────────────────────

    def _build_deliberate(self, scene: str,
                           target_code: str = None) -> str:
        """
        思考路徑：
        - personality 完整版 + habit
        - 全部 STM（有 token 預算限制）
        - LTM 相關命題（用 retrieve 取最相關的）
        - 當前對象的完整關係描述（initial + summary）
        """
        char = self.character

        # 全部 STM，但受 token 預算限制
        stm_text = _format_stm(
            self.stm.get_all(),
            max_chars=STM_TOKEN_BUDGET * 4  # 粗估 1 token ≈ 4 字元
        )

        # LTM：優先取與當前對象相關的命題
        if target_code:
            from config.world_config import CHARACTER_NAMES
            target_name = CHARACTER_NAMES.get(target_code, target_code)
            props1 = self.ltm.retrieve(query_subject=char.name, top_k=5)
            props1_ids = {p["id"] for p in props1}
            # update_access=False 避免已在 props1 命中的命題被重複計數
            props2 = self.ltm.retrieve(
                query_object=target_name, top_k=5, update_access=False
            )
            for p in props2:
                if p["id"] not in props1_ids:
                    p["access_count"] += 1
                    p["strength"] = 1.0
            # 去重
            seen = set()
            unique_props = []
            for p in props1 + props2:
                if p["id"] not in seen:
                    seen.add(p["id"])
                    unique_props.append(p)
            ltm_props_text = self.ltm.to_text(unique_props)
        else:
            # 沒有特定對象，取全部 LTM（受 token 預算限制）
            all_props = self.ltm.get_all()
            ltm_props_text = _truncate(
                self.ltm.to_text(all_props),
                max_chars=LTM_TOKEN_BUDGET * 4
            )

        # 關係描述（完整版，含 summary）
        rel_text = ""
        if target_code:
            rel_text = char.get_relationship_text(
                target_code, include_summary=True
            )

        return prompt_deliberate(
            character_name  = char.name,
            personality     = char.get_personality(short=False),
            habit           = char.get_habit(),
            emotion         = char.emotion,
            relationship_text = rel_text,
            stm_text        = stm_text,
            ltm_props_text  = ltm_props_text,
            scene           = scene,
        )


# ── 工具函式 ─────────────────────────────────────────────────────

def _format_stm(turns: list, max_chars: int = None) -> str:
    """把 STM turns 格式化為 prompt 可用的文字。"""
    if not turns:
        return "（目前沒有近期記憶）"

    lines = []
    for t in turns:
        lines.append(f"[{t['turn_id']}] {t['scene']}")
        if t.get("image_desc"):
            lines.append(f"  看到：{t['image_desc']}")
        if t.get("input_text"):
            lines.append(f"  收到：{t['input_text']}")
        lines.append(f"  行動：{t['action']}")

    text = "\n".join(lines)
    if max_chars:
        text = _truncate(text, max_chars)
    return text


def _truncate(text: str, max_chars: int) -> str:
    """超過 max_chars 時截斷並加上省略提示。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...（已截斷）"