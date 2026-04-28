# ================================================================
# config/prompts.py
# 所有 prompt 集中管理
# 每個函式接受必要變數，回傳完整 prompt 字串
# 需要調整措辭或格式時，只改這個檔案
# ================================================================

from config.action_list import VALID_ACTIONS, VALID_LOCATIONS
from config.world_config import VALID_EMOTIONS


# ================================================================
# 睡眠濃縮相關 prompt
# ================================================================

def prompt_select_ltm(character_name: str,
                       turns_text: str,
                       props_text: str) -> str:
    """
    判斷今天的 STM 命題中，哪些值得存進 LTM。
    用於 memory_consolidation._select_important()
    """
    return f"""你正在幫 {character_name} 決定哪些今日記憶值得長期保存。

今天發生的事：
{turns_text}

今天抽取出的所有命題：
{props_text}

請從上面的命題中，選出值得長期記憶的項目。
規則：
- 保留涉及 {character_name} 認識的人或新認識的人的事件
- 保留改變了 {character_name} 感受或關係的事件
- 刪除瑣碎的日常行為（正常工作、吃飯等）
- 最多保留 5 筆

只回傳 JSON 陣列，不要其他文字。範例：
[{{"subject":"{character_name}","relation":"遇見","object":"David","location":"咖啡廳"}}]"""


def prompt_ltm_summary(character_name: str,
                        props_text: str) -> str:
    """
    根據目前所有 LTM 命題產生壓縮摘要（1-2 句話）。
    用於 memory_consolidation._generate_summary()
    """
    return f"""請用 1-2 句話總結 {character_name} 的長期記憶。

記憶命題：
{props_text}

以 {character_name} 的視角撰寫，聚焦在重要的人際關係與事件。
只回傳摘要文字，不要其他格式。"""


def prompt_update_relationship(char_name: str,
                                target_name: str,
                                initial: str,
                                old_summary: str,
                                turns_text: str) -> str:
    """
    根據今天的對話更新與特定角色的關係摘要。
    用於 memory_consolidation._update_relationships()
    """
    old_part = f"先前的關係摘要：{old_summary}" if old_summary else "先前沒有摘要紀錄。"
    return f"""請更新 {char_name} 與 {target_name} 之間的關係摘要。

初始關係設定：{initial}
{old_part}

今天相關的事件：
{turns_text}

用一句話描述目前兩人關係的現狀。
只有今天發生了重要變化才需要更新，否則維持先前摘要的核心內容。
只回傳一句話，不要其他格式。"""


def prompt_infer_emotion(character_name: str,
                          turns_text: str) -> str:
    """
    根據今天發生的事推斷角色目前的情緒。
    用於 memory_consolidation._infer_emotion()
    """
    emotions_str = "、".join(VALID_EMOTIONS)
    return f"""根據今天發生的事，{character_name} 目前的情緒是什麼？

今天的事件：
{turns_text}

從以下選項選一個最符合的情緒：
{emotions_str}

只回傳情緒詞，不要其他文字。"""


# ================================================================
# 主推論 prompt（直覺路徑 & 思考路徑）
# ================================================================

def prompt_intuitive(character_name: str,
                      personality_short: str,
                      emotion: str,
                      relationship_text: str,
                      stm_text: str,
                      ltm_summary: str,
                      scene: str) -> str:
    """直覺路徑 prompt：快速決策，省 token。"""
    actions_str   = "、".join(VALID_ACTIONS)
    locations_str = "、".join(VALID_LOCATIONS)

    ltm_part = f"\n【長期記憶摘要】\n{ltm_summary}" if ltm_summary else ""
    rel_part = f"\n【與對方的關係】\n{relationship_text}" if relationship_text else ""

    return f"""你正在扮演 {character_name}。

【個性】{personality_short}
【目前情緒】{emotion}{rel_part}
【最近發生的事】
{stm_text}{ltm_part}

【目前場景】
{scene}

【可執行的行動】{actions_str}
【可前往的地點】{locations_str}

請依照以下格式回答：
[ACTION]: （從行動清單選一個；移動用「前往:地點」；對話用「對話:」加上實際要說的完整一句話，例如「對話:你剛才說的那件事，我一直在想……」）
[THOUGHT]: （{character_name} 的內心想法，1-2 句）
[HAM]: （本場景抽取的命題，JSON 陣列，1-3 筆，格式：[{{"subject":"...","relation":"...","object":"..."}}]）
[/HAM]"""


def prompt_deliberate(character_name: str,
                       personality: str,
                       habit: str,
                       emotion: str,
                       relationship_text: str,
                       stm_text: str,
                       ltm_props_text: str,
                       scene: str,
                       current_event: str = "") -> str:
    """
    思考路徑 prompt（複雜思考）：完整記憶注入，模型自主決策。
    角色根據完整長短期記憶、個性、情緒，自己選擇最合適的行動。
    強調真實對話（不能只說「想說的話」）與情境考量（工作、時間、對方狀態）。
    """
    actions_str   = "、".join(VALID_ACTIONS)
    locations_str = "、".join(VALID_LOCATIONS)

    ltm_part   = f"\n【長期記憶（HAM 命題）】\n{ltm_props_text}" if ltm_props_text else ""
    rel_part   = f"\n【與對方的關係】\n{relationship_text}" if relationship_text else ""
    event_part = f"\n\n【當前事件】\n{current_event}" if current_event else ""

    return f"""你正在扮演 {character_name}，請根據你的完整記憶與個性，自主決定此刻最想做的事。

【完整個性】{personality}
【習慣】{habit}
【目前情緒】{emotion}{rel_part}
【今天的短期記憶（STM）】
{stm_text}{ltm_part}

【目前場景】
{scene}{event_part}

【可執行的行動】{actions_str}
【可前往的地點】{locations_str}

決策時請考慮：
- 現在幾點？是否需要上班、吃飯、休息或回家？
- 附近有誰？如果想說話，對方在嗎？
- 還有哪些重要的事沒做完？工作是否會受影響？
- 睡眠時機：感覺疲憊或已是深夜時可選「睡覺」

如果選擇「對話」，必須寫出你真正要說的完整一句話。
✗ 不要寫：「對話:想說的話」
✓ 正確範例：「對話:你今天工作順利嗎？我看你一直皺著眉頭。」
✓ 正確範例：「對話:我有件事一直想跟你說，但不知道從哪裡開口。」
✓ 正確範例：「對話:謝謝你今天陪我，真的讓我輕鬆很多。」

請依照以下格式回答：
[ACTION]: （從行動清單選一個；移動用「前往:地點」；對話用「對話:」加上實際說出的一整句話）
[THOUGHT]: （{character_name} 的內心想法，2-3 句，反映個性、記憶與當下情境的真實考量）
[HAM]: （本場景的關鍵命題，JSON 陣列，1-5 筆）
格式：[{{"subject":"...","relation":"...","object":"...","location":"...","time":"..."}}]
[/HAM]"""


# ================================================================
# 時間表生成 prompt（之後加入時間表機制時使用）
# ================================================================

def prompt_generate_schedule(character_name: str,
                               personality_short: str,
                               habit: str,
                               ltm_summary: str,
                               day: int) -> str:
    """
    每天起床時，根據角色個性與記憶生成當天的動態時段。
    用於之後的時間表機制。
    """
    return f"""你正在幫 {character_name} 規劃第 {day} 天的行程。

【個性】{personality_short}
【習慣】{habit}
【長期記憶摘要】{ltm_summary if ltm_summary else "（目前沒有記憶）"}

請在固定行程之外，加入 0-2 個今天可能發生的動態事件。
動態事件需符合角色個性，並與記憶中的人物或地點有關。

只回傳 JSON 陣列，格式：
[{{"time":"14:00","action":"對話","location":"咖啡廳"}}]
若沒有動態事件，回傳空陣列 []。"""