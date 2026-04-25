# ================================================================
# core/memory_consolidation.py
# 睡眠濃縮：STM -> LTM
# 每天角色睡覺時觸發，由模型判斷哪些值得存進 LTM
# 同時更新 relationship_state、emotion、ltm_summary
# ================================================================

import json
import re
from config.prompts import (
    prompt_select_ltm,
    prompt_ltm_summary,
    prompt_update_relationship,
    prompt_infer_emotion,
)
from config.world_config import VALID_EMOTIONS, LTM_FALLBACK_COUNT
from config.model_config import (
    CONSOLIDATE_SELECT_MAX_TOKENS,
    CONSOLIDATE_SUMMARY_MAX_TOKENS,
    CONSOLIDATE_RELATION_MAX_TOKENS,
    CONSOLIDATE_EMOTION_MAX_TOKENS,
)


# ── 濃縮流程主函式（agent 呼叫這個）────────────────────────────

def consolidate(character, stm, ltm, make_model_fn) -> dict:
    """
    執行完整的睡眠濃縮流程。

    character    : Character 物件
    stm          : STM 物件
    ltm          : LTM 物件
    make_model_fn: 呼叫 loader.make_model_fn(max_new_tokens=N) 的 callable
                   由 agent.py 傳入，讓 core 不直接依賴 model 模組

    回傳 {"new_propositions": int, "summary": str,
           "ltm_total": int, "ltm_pruned": int}
    """
    turns = stm.get_all()

    if not turns:
        return {"new_propositions": 0, "summary": ltm.get_summary(),
                "ltm_total": ltm.count(), "ltm_pruned": 0}

    # ── Step 1：請模型判斷哪些命題值得存進 LTM ──────────────────
    all_props       = stm.get_all_propositions()
    important_props = _select_important(
        character, all_props, turns,
        make_model_fn(max_new_tokens=CONSOLIDATE_SELECT_MAX_TOKENS))

    # ── Step 2：寫入 LTM ─────────────────────────────────────────
    ltm.encode_batch(important_props, day=character.day)

    # ── Step 3：更新 LTM 摘要 ────────────────────────────────────
    new_summary = _generate_summary(
        character, ltm,
        make_model_fn(max_new_tokens=CONSOLIDATE_SUMMARY_MAX_TOKENS))
    ltm.set_summary(new_summary)

    # ── Step 4：更新關係摘要 ─────────────────────────────────────
    _update_relationships(
        character, turns,
        make_model_fn(max_new_tokens=CONSOLIDATE_RELATION_MAX_TOKENS))

    # ── Step 5：更新情緒 ─────────────────────────────────────────
    new_emotion = _infer_emotion(
        character, turns,
        make_model_fn(max_new_tokens=CONSOLIDATE_EMOTION_MAX_TOKENS))
    character.emotion = new_emotion

    # ── Step 6：套用 LTM 衰減 ────────────────────────────────────
    ltm.apply_decay()
    removed = ltm.prune()

    # ── Step 7：清空 STM，推進到下一天 ───────────────────────────
    stm.clear()
    character.advance_day()

    return {
        "new_propositions": len(important_props),
        "summary":          new_summary,
        "ltm_total":        ltm.count(),
        "ltm_pruned":       removed,
    }


# ── 私有函式 ─────────────────────────────────────────────────────

def _select_important(character, all_props: list,
                      turns: list, model_fn) -> list:
    """請模型從今天的 STM 命題中選出值得長期記憶的事件。"""
    if not all_props:
        if not turns:
            return []
        # Bootstrap：Markov 路徑不生成 HAM，從行動紀錄建立基本命題供模型篩選
        basic = []
        for t in turns:
            action = t.get('action', '')
            scene  = t.get('scene', '')[:40]
            if not action:
                continue
            if ':' in action:
                verb, obj = action.split(':', 1)
                basic.append({"subject": character.name,
                               "relation": verb, "object": obj[:40]})
            else:
                basic.append({"subject": character.name,
                               "relation": action,
                               "object": scene if scene else "日常"})
        all_props = basic[:5]

    if not all_props:
        return []

    turns_text = _format_turns(turns)
    props_text = json.dumps(all_props, ensure_ascii=False, indent=2)
    prompt     = prompt_select_ltm(character.name, turns_text, props_text)
    raw        = model_fn(prompt)

    return _parse_json_list(raw, fallback=all_props[:LTM_FALLBACK_COUNT])


def _generate_summary(character, ltm, model_fn) -> str:
    """請模型根據目前所有 LTM 命題產生新的壓縮摘要（1-2 句話）。"""
    if ltm.count() == 0:
        return ""

    prompt  = prompt_ltm_summary(character.name, ltm.to_text())
    summary = model_fn(prompt).strip()

    # 防止模型輸出過長
    if len(summary) > 200:
        summary = summary[:200] + "..."
    return summary


def _update_relationships(character, turns: list, model_fn):
    """請模型根據今天的對話，更新與互動角色的關係摘要。"""
    from config.world_config import CHARACTER_NAMES

    turns_text     = _format_turns(turns)
    involved_codes = [
        code for code, name in CHARACTER_NAMES.items()
        if code != character.code and name in turns_text
    ]

    for code in involved_codes:
        name    = CHARACTER_NAMES[code]
        rel     = character.get_relationship(code)
        initial = rel.get("initial", "")
        old_sum = rel.get("summary", "")

        prompt      = prompt_update_relationship(
            character.name, name, initial, old_sum, turns_text
        )
        new_summary = model_fn(prompt).strip()
        if new_summary:
            character.update_relationship_summary(code, new_summary)


def _infer_emotion(character, turns: list, model_fn) -> str:
    """請模型根據今天發生的事推斷角色的當前情緒。"""
    turns_text = _format_turns(turns)
    prompt     = prompt_infer_emotion(character.name, turns_text)
    emotion    = model_fn(prompt).strip()

    return emotion if emotion in VALID_EMOTIONS else VALID_EMOTIONS[0]


# ── 工具函式 ─────────────────────────────────────────────────────

def _format_turns(turns: list) -> str:
    """把 STM turns 格式化為可讀文字。"""
    lines = []
    for t in turns:
        lines.append(f"[{t['turn_id']}] 場景：{t['scene']}")
        if t.get("image_desc"):
            lines.append(f"  視覺：{t['image_desc']}")
        if t.get("input_text"):
            lines.append(f"  輸入：{t['input_text']}")
        lines.append(f"  行動：{t['action']}")
    return "\n".join(lines)


def _parse_json_list(raw: str, fallback: list) -> list:
    """從模型輸出中解析 JSON list，失敗時回傳 fallback。"""
    try:
        raw    = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return fallback