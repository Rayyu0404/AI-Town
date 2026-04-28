# ================================================================
# simulate.py
# 多天模擬（預設 10 天）+ 全功能驗證 + 詳細 HTML 報告
#
# 執行方式：
#   python simulate.py              → 完整模式（Phi 模型）
#   python simulate.py --no-model   → 假模型快速驗證（不需 GPU）
#   python simulate.py --slots N    → 每天時間段數（預設 4）
#   python simulate.py --check-only → 只驗證模型下載
#
# 模擬涵蓋（10 天）：
#   Day 1: 初始日（以 Markov 為主）→ 首次對話 → 睡眠濃縮
#   Day 2: 有 LTM + 情緒複雜觸發 → 複雜思考 → 對話
#   Day 3: LTM 增長 + 情緒變化 → 全功能運作
#   Day 4-10: 持續深化角色關係與思考路徑驗證
#
# 決策路徑：
#   簡單思考（Markov）: C < threshold  → 機率分布選行動
#   複雜思考（LTM）:    C ≥ threshold 或 K ≥ 0.4 → 模型自主推理
#
# 輸出：
#   simulate_report.html  — 含天分頁、對話展開的視覺化報告
# ================================================================

import argparse
import sys
import io
import json
import os
import time
import traceback
import numpy as np
from copy import deepcopy

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, '.')

from config.world_config import (CHARACTER_NAMES, CHARACTER_CODES,
                                  MAX_CONVERSATION_TURNS_PER_SLOT,
                                  VALID_EMOTIONS)
from config.action_list import VALID_ACTIONS, VALID_LOCATIONS


# ================================================================
# 輸出工具
# ================================================================

_results  = {'pass': 0, 'fail': 0}
_sim_data = {'days': [], 'anomalies': [], 'stats': {}}


def _check(label: str, cond: bool, detail: str = '') -> bool:
    sym = 'PASS' if cond else 'FAIL'
    sfx = f'  ({detail})' if detail else ''
    print(f'    [{sym}] {label}{sfx}')
    _results['pass' if cond else 'fail'] += 1
    return cond


def section(t):
    print(f'\n{"="*62}\n  {t}\n{"="*62}')


def subsection(t):
    print(f'\n  ── {t} ──')


# ================================================================
# 智慧假模型（--no-model 模式）
# ================================================================

# ── 工具：從 prompt 擷取角色名字 ────────────────────────────────
def _extract_char_from_prompt(text: str) -> str:
    """從 prompt 文字擷取當前角色名字（涵蓋所有 prompt 模板格式）。"""
    for name in ['Amy', 'Ben', 'Claire', 'David', 'Emma']:
        if (f'你正在扮演 {name}' in text or
                f'幫 {name} 決定' in text or
                f'幫 {name} 規劃' in text or
                f'{name} 目前的情緒' in text or
                f'總結 {name} 的' in text or
                f'更新 {name} 與' in text or
                f'幫 {name} ' in text):
            return name
    return 'Amy'


def _extract_target_from_prompt(text: str, char_name: str) -> str:
    """從關係區塊擷取對話目標角色名字。"""
    idx = text.find('【與對方的關係】')
    if idx == -1:
        return ''
    section = text[idx:idx + 300]
    for name in ['Amy', 'Ben', 'Claire', 'David', 'Emma']:
        if name != char_name and name in section:
            return name
    return ''


# ── 各角色決策輸出池（性格符合，含部分對話選項） ────────────────
# 每個角色 6 項：4 個日常行動 + 2 個主動對話（來回輪換）
_CHAR_OUTPUTS = {
    'Amy': [
        ('[ACTION]: 工作\n'
         '[THOUGHT]: Amy認真泡著每一杯咖啡，這是她最擅長的事，讓她感到安心。\n'
         '[HAM]: [{"subject":"Amy","relation":"工作","object":"咖啡廳","time":"上午"}]\n[/HAM]'),
        ('[ACTION]: 對話:今天的咖啡調好了，你要嗎？\n'
         '[THOUGHT]: Amy鼓起勇氣先開口，心跳有點快但表面維持自然。\n'
         '[HAM]: [{"subject":"Amy","relation":"關心","object":"David","location":"咖啡廳"}]\n[/HAM]'),
        ('[ACTION]: 工作\n'
         '[THOUGHT]: 下午時段也要維持水準，Amy提醒自己專心。\n'
         '[HAM]: [{"subject":"Amy","relation":"工作","object":"咖啡","time":"下午"}]\n[/HAM]'),
        ('[ACTION]: 整理\n'
         '[THOUGHT]: Amy把吧台整理得乾乾淨淨，心情也跟著平靜下來。\n'
         '[HAM]: [{"subject":"Amy","relation":"整理","object":"吧台","location":"咖啡廳"}]\n[/HAM]'),
        ('[ACTION]: 對話:你今天看起來有些不一樣呢。\n'
         '[THOUGHT]: Amy說完立刻後悔，不知道這樣說合不合適。\n'
         '[HAM]: [{"subject":"Amy","relation":"注意","object":"David的表情","location":"咖啡廳"}]\n[/HAM]'),
        ('[ACTION]: 休息\n'
         '[THOUGHT]: 趁空檔喘口氣，腦子裡卻還是轉個不停。\n'
         '[HAM]: [{"subject":"Amy","relation":"休息","object":"後場","time":"午休"}]\n[/HAM]'),
    ],
    'Ben': [
        ('[ACTION]: 工作\n'
         '[THOUGHT]: Ben整理著貨架，心裡偶爾飄到Amy的臉，但還是把工作做完。\n'
         '[HAM]: [{"subject":"Ben","relation":"工作","object":"超市","time":"上午"}]\n[/HAM]'),
        ('[ACTION]: 對話:妳今天……有空聊一下嗎？\n'
         '[THOUGHT]: Ben終於鼓起勇氣，但說完就開始緊張地等對方反應。\n'
         '[HAM]: [{"subject":"Ben","relation":"嘗試","object":"找Amy說話","location":"咖啡廳"}]\n[/HAM]'),
        ('[ACTION]: 工作\n'
         '[THOUGHT]: 下午繼續，Ben讓自己專心工作，別想太多。\n'
         '[HAM]: [{"subject":"Ben","relation":"工作","object":"超市","time":"下午"}]\n[/HAM]'),
        ('[ACTION]: 散步\n'
         '[THOUGHT]: Ben在附近晃了晃，說不定能巧遇Amy，但也沒抱太大希望。\n'
         '[HAM]: [{"subject":"Ben","relation":"散步","object":"咖啡廳附近","time":"下午"}]\n[/HAM]'),
        ('[ACTION]: 對話:最近還好嗎？感覺妳有點累。\n'
         '[THOUGHT]: Ben關心Amy是真的，但也知道這種方式太刻意了。\n'
         '[HAM]: [{"subject":"Ben","relation":"關心","object":"Amy","location":"咖啡廳"}]\n[/HAM]'),
        ('[ACTION]: 休息\n'
         '[THOUGHT]: Ben靠著牆休息，想著有天要怎麼開口跟Amy說。\n'
         '[HAM]: [{"subject":"Ben","relation":"想著","object":"如何追求Amy","time":"午休"}]\n[/HAM]'),
    ],
    'Claire': [
        ('[ACTION]: 工作\n'
         '[THOUGHT]: Claire把報告做完，效率是她最引以為傲的事。\n'
         '[HAM]: [{"subject":"Claire","relation":"工作","object":"辦公室","time":"上午"}]\n[/HAM]'),
        ('[ACTION]: 對話:我有件事想聽你的意見，你有空嗎？\n'
         '[THOUGHT]: Claire說話一向直接，但這次開口前多猶豫了一秒。\n'
         '[HAM]: [{"subject":"Claire","relation":"想說話","object":"Emma","location":"餐廳"}]\n[/HAM]'),
        ('[ACTION]: 工作\n'
         '[THOUGHT]: 有點心不在焉，但Claire還是逼自己把文件看完。\n'
         '[HAM]: [{"subject":"Claire","relation":"工作","object":"文件","time":"下午"}]\n[/HAM]'),
        ('[ACTION]: 整理\n'
         '[THOUGHT]: Claire整理桌面，做事喜歡有條理，這樣看起來清晰多了。\n'
         '[HAM]: [{"subject":"Claire","relation":"整理","object":"辦公桌","location":"辦公室"}]\n[/HAM]'),
        ('[ACTION]: 對話:有件事我想直接說，說清楚比較好。\n'
         '[THOUGHT]: Claire深吸一口氣，決定不再兜圈子，直接說出心裡的話。\n'
         '[HAM]: [{"subject":"Claire","relation":"表達","object":"心裡的感受","location":"餐廳"}]\n[/HAM]'),
        ('[ACTION]: 休息\n'
         '[THOUGHT]: 工作完成，Claire坐著喝杯水，讓頭腦放空一下。\n'
         '[HAM]: [{"subject":"Claire","relation":"休息","object":"辦公室","time":"午休"}]\n[/HAM]'),
    ],
    'David': [
        ('[ACTION]: 工作\n'
         '[THOUGHT]: David專注處理文件，一切按計畫進行，這讓他感到踏實。\n'
         '[HAM]: [{"subject":"David","relation":"工作","object":"辦公室","time":"上午"}]\n[/HAM]'),
        ('[ACTION]: 對話:今天的拿鐵，謝謝你。\n'
         '[THOUGHT]: David說得很簡單，但這句話帶著他平時不會說出口的溫度。\n'
         '[HAM]: [{"subject":"David","relation":"感謝","object":"Amy","location":"咖啡廳"}]\n[/HAM]'),
        ('[ACTION]: 工作\n'
         '[THOUGHT]: 下午繼續，David不喜歡拖拖拉拉，有事就做完。\n'
         '[HAM]: [{"subject":"David","relation":"完成","object":"工作","time":"下午"}]\n[/HAM]'),
        ('[ACTION]: 散步\n'
         '[THOUGHT]: 難得有個空檔，David到公司附近走走，把腦子清空一下。\n'
         '[HAM]: [{"subject":"David","relation":"散步","object":"公司附近","time":"午休"}]\n[/HAM]'),
        ('[ACTION]: 對話:有件事，我想直接問你。\n'
         '[THOUGHT]: David停頓了一下，選擇開口，這對他來說並不容易。\n'
         '[HAM]: [{"subject":"David","relation":"嘗試","object":"表達感受","location":"咖啡廳"}]\n[/HAM]'),
        ('[ACTION]: 休息\n'
         '[THOUGHT]: 今天的工作差不多了，David讓自己稍微放鬆一下。\n'
         '[HAM]: [{"subject":"David","relation":"休息","object":"辦公室","time":"傍晚"}]\n[/HAM]'),
    ],
    'Emma': [
        ('[ACTION]: 工作\n'
         '[THOUGHT]: Emma忙著準備今天的菜單，工作讓她感到充實又踏實。\n'
         '[HAM]: [{"subject":"Emma","relation":"工作","object":"餐廳","time":"上午"}]\n[/HAM]'),
        ('[ACTION]: 對話:你今天看起來心情不錯？\n'
         '[THOUGHT]: Emma察覺到對方情緒有些不一樣，自然地開口問了一句。\n'
         '[HAM]: [{"subject":"Emma","relation":"察覺","object":"Claire的情緒","location":"餐廳"}]\n[/HAM]'),
        ('[ACTION]: 工作\n'
         '[THOUGHT]: 今天客人特別多，Emma很認真但也感到疲憊，繼續撐著。\n'
         '[HAM]: [{"subject":"Emma","relation":"工作","object":"餐廳","time":"下午"}]\n[/HAM]'),
        ('[ACTION]: 整理\n'
         '[THOUGHT]: Emma把廚房整理好，做事喜歡有條理，這樣才能安心工作。\n'
         '[HAM]: [{"subject":"Emma","relation":"整理","object":"餐廳","time":"傍晚"}]\n[/HAM]'),
        ('[ACTION]: 對話:你剛說的那件事……我想多了解一下。\n'
         '[THOUGHT]: Emma說完，感覺到這句話帶著她自己也沒預料到的重量。\n'
         '[HAM]: [{"subject":"Emma","relation":"回應","object":"Claire說的話","location":"餐廳"}]\n[/HAM]'),
        ('[ACTION]: 休息\n'
         '[THOUGHT]: 午餐前的準備告一段落，Emma靜靜坐著休息一下。\n'
         '[HAM]: [{"subject":"Emma","relation":"休息","object":"餐廳","time":"中午"}]\n[/HAM]'),
    ],
}

# _CHAR_OUTPUTS_WITH_TARGET: 有目標角色時優先使用（對話佔比更高）
_CHAR_OUTPUTS_WITH_TARGET = {
    name: [o for o in pool if '[ACTION]: 對話' in o]
    for name, pool in _CHAR_OUTPUTS.items()
}


# ── 各角色對話回應池（收到對方說話時使用） ──────────────────────
_DIALOGUE_RESPONSES = {
    'Amy': [
        ('對話:好啊，你說吧，我在聽。',
         'Amy把手邊的事放下，抬頭看著對方，認真地準備聆聽。'),
        ('對話:嗯，我也有點想跟你說這件事。',
         'Amy眼神稍微亮起來，心裡有些說不清的期待。'),
        ('對話:是嗎……我沒想到你會這樣說。',
         'Amy輕輕咬了一下嘴唇，一時不知道怎麼接話。'),
        ('對話:謝謝你，這樣說讓我放心多了。',
         'Amy心裡暖了一下，笑了笑，感覺輕鬆不少。'),
        ('對話:對，我也是這樣想的。',
         'Amy點頭，表面平靜，心裡其實也在轉。'),
    ],
    'Ben': [
        ('對話:是啊，最近還不錯，謝謝妳問。',
         'Ben盡量表現得自然，但說話時有點緊張。'),
        ('對話:嗯，我明白你的意思了。',
         'Ben認真地回應，心裡想著下次要更有勇氣。'),
        ('對話:其實……我也有點想說什麼，但不知道怎麼開口。',
         'Ben想說些什麼，目光落在遠處，最後還是退縮了一半。'),
        ('對話:好，我知道了，你說的我記住了。',
         'Ben點頭，心裡有點落寞但沒有表現出來。'),
        ('對話:沒關係，我只是想來看看你。',
         'Ben輕描淡寫地說，掩飾內心其實很在乎的情緒。'),
    ],
    'Claire': [
        ('對話:嗯，我就知道你能理解，謝謝你。',
         'Claire說話直接，但這次語氣比平常溫柔了幾分。'),
        ('對話:你說得對，就這樣處理吧。',
         'Claire直接給出回應，這是她一貫俐落的風格。'),
        ('對話:好，我聽明白了，你的想法跟我有點不同。',
         'Claire稍微停頓，思考對方說的話。'),
        ('對話:我知道，你放心，我會處理好的。',
         'Claire回答得簡潔，但心裡其實有些感動。'),
        ('對話:有話直說就好，不用拐彎抹角。',
         'Claire帶著一貫直率的語氣說，但眼神其實很溫柔。'),
    ],
    'David': [
        ('對話:好，謝謝你告訴我。',
         'David冷靜地回應，但語氣裡帶著一絲少見的溫度。'),
        ('對話:嗯，讓我想想。',
         'David習慣在開口前多想一秒，這次也一樣。'),
        ('對話:我懂你的意思，謝謝你。',
         'David回答簡短，但眼神清晰地告訴對方他確實在聽。'),
        ('對話:其實我也想說一件事。',
         'David停頓了一下，選擇主動說話，這對他來說並不容易。'),
        ('對話:好，我記住了。',
         'David說話穩重，話雖不多，但份量十足。'),
    ],
    'Emma': [
        ('對話:謝謝你說這些，我會放在心上的。',
         'Emma溫和地回應，讓對方感到被認真對待。'),
        ('對話:你這樣說，讓我輕鬆多了。',
         'Emma說完，臉上帶著真誠的微笑。'),
        ('對話:其實我最近也有在想這件事。',
         'Emma說話時，眼神裡透著認真和溫柔。'),
        ('對話:嗯，我理解你，我們不著急。',
         'Emma輕聲說，語氣溫柔而堅定。'),
        ('對話:謝謝你敢說出來，這樣就好多了。',
         'Emma感受到對方的誠意，心裡也有些觸動。'),
    ],
}

# ── 各角色睡眠濃縮資料（按角色 × 天數） ─────────────────────────
_MOCK_SLEEP_BY_CHAR = {
    'Amy': {
        'day_1': {
            'emotion':    '開心',
            'ltm_select': '[{"subject":"Amy","relation":"遇見","object":"David","location":"咖啡廳","time":"早上"},{"subject":"Amy","relation":"工作","object":"咖啡廳"}]',
            'summary':    'Amy今天在咖啡廳工作，David來買咖啡，兩人有短暫交流，讓Amy心情愉快。',
            'relation_D': 'Amy今天為David做了咖啡，心裡有些悸動但沒有說出口。',
        },
        'day_2': {
            'emotion':    '期待',
            'ltm_select': '[{"subject":"Amy","relation":"感受","object":"心跳加速","location":"咖啡廳"},{"subject":"Amy","relation":"想到","object":"David的眼神","time":"午休"}]',
            'summary':    'Amy越來越在意David，每次他來咖啡廳都讓她心跳加速，說不清是什麼感覺。',
            'relation_D': '兩人互動增多，Amy有種說不清的期待感。',
        },
        'day_3': {
            'emotion':    '不安',
            'ltm_select': '[{"subject":"Amy","relation":"意識到","object":"對David的感情"},{"subject":"Amy","relation":"工作","object":"咖啡廳","time":"下午"}]',
            'summary':    'Amy開始意識到自己對David的感情，工作之餘常常在想他，心裡有些不安。',
            'relation_D': 'Amy對David的感情逐漸明朗，但仍然選擇把感受壓在心裡。',
        },
        'day_4': {
            'emotion':    '平靜',
            'ltm_select': '[{"subject":"Amy","relation":"決定","object":"繼續暗戀"},{"subject":"Amy","relation":"珍惜","object":"每次相遇","location":"咖啡廳"}]',
            'summary':    'Amy決定把感情藏在心裡，在平靜的日常中珍惜每一次與David的相遇。',
            'relation_D': 'Amy仍暗戀David，選擇用行動而非言語表達在意。',
        },
    },
    'Ben': {
        'day_1': {
            'emotion':    '平靜',
            'ltm_select': '[{"subject":"Ben","relation":"工作","object":"超市"}]',
            'summary':    'Ben今天在超市工作，沒有什麼特別的事發生。',
            'relation_A': '',
        },
        'day_2': {
            'emotion':    '失落',
            'ltm_select': '[{"subject":"Ben","relation":"想見","object":"Amy"},{"subject":"Ben","relation":"工作","object":"超市"}]',
            'summary':    'Ben想去找Amy說話，但一直沒有好的機會，心裡有些失落。',
            'relation_A': 'Ben想跟Amy更靠近，但每次都找不到好時機。',
        },
        'day_3': {
            'emotion':    '不安',
            'ltm_select': '[{"subject":"Ben","relation":"猶豫","object":"如何表達對Amy的感情"},{"subject":"Ben","relation":"工作","object":"超市"}]',
            'summary':    'Ben心裡越來越不安，想跟Amy說清楚但每次都退縮了。',
            'relation_A': 'Ben對Amy的感情越來越明確，但說不出口讓他焦慮。',
        },
        'day_4': {
            'emotion':    '疲憊',
            'ltm_select': '[{"subject":"Ben","relation":"鼓起勇氣","object":"找Amy說話"},{"subject":"Ben","relation":"又退縮","object":"表白"}]',
            'summary':    'Ben今天又沒說出口，帶著說不清的疲憊感回家，下次還是要鼓起勇氣。',
            'relation_A': 'Ben想對Amy表白但一直沒機會說，心裡既期待又害怕。',
        },
    },
    'Claire': {
        'day_1': {
            'emotion':    '平靜',
            'ltm_select': '[{"subject":"Claire","relation":"工作","object":"辦公室"}]',
            'summary':    'Claire今天工作順利完成，一切都在掌握之中。',
            'relation_E': '',
        },
        'day_2': {
            'emotion':    '興奮',
            'ltm_select': '[{"subject":"Claire","relation":"遇見","object":"Emma","location":"餐廳"},{"subject":"Claire","relation":"感受","object":"心跳加速"}]',
            'summary':    'Claire在餐廳看到Emma，莫名心跳加速，今天心情特別好，說不清為什麼。',
            'relation_E': 'Claire開始更在意Emma，但還不確定自己的感受是什麼。',
        },
        'day_3': {
            'emotion':    '興奮',
            'ltm_select': '[{"subject":"Claire","relation":"主動找","object":"Emma"},{"subject":"Claire","relation":"感受","object":"自在"}]',
            'summary':    'Claire越來越想找機會和Emma說話，和Emma在一起的時候感到很自在。',
            'relation_E': 'Claire確認了對Emma有特殊感情，考慮找機會說出口。',
        },
        'day_4': {
            'emotion':    '開心',
            'ltm_select': '[{"subject":"Claire","relation":"告白","object":"Emma"},{"subject":"Claire","relation":"說出","object":"心裡的感受"}]',
            'summary':    'Claire終於向Emma說出了心裡的話，感到前所未有的輕鬆，不管結果如何都覺得釋懷了。',
            'relation_E': 'Claire向Emma告白了，等待Emma的回應，但心裡已經輕鬆許多。',
        },
    },
    'David': {
        'day_1': {
            'emotion':    '平靜',
            'ltm_select': '[{"subject":"David","relation":"去","object":"咖啡廳","location":"咖啡廳"},{"subject":"David","relation":"工作","object":"辦公室"}]',
            'summary':    'David按照習慣去咖啡廳，工作一切正常，今天沒有什麼特別的事。',
            'relation_A': '',
            'relation_E': '',
        },
        'day_2': {
            'emotion':    '困惑',
            'ltm_select': '[{"subject":"David","relation":"注意到","object":"Amy的眼神"},{"subject":"David","relation":"工作","object":"辦公室"}]',
            'summary':    'David今天工作中分了些心，Amy的眼神讓他有一瞬間說不清楚的感覺。',
            'relation_A': 'David隱約注意到Amy看他的眼神有些不一樣，但他沒有深究。',
            'relation_E': '',
        },
        'day_3': {
            'emotion':    '困惑',
            'ltm_select': '[{"subject":"David","relation":"思考","object":"和Amy的互動"},{"subject":"David","relation":"完成","object":"合約"}]',
            'summary':    'David完成了合約，但腦子裡偶爾想起Amy，那種說不清的感覺讓他有點困惑。',
            'relation_A': 'David開始在意Amy，但他習慣把感受壓在心裡不說。',
            'relation_E': '',
        },
        'day_4': {
            'emotion':    '開心',
            'ltm_select': '[{"subject":"David","relation":"說出","object":"心裡的話"},{"subject":"David","relation":"對話","object":"Amy","location":"咖啡廳"}]',
            'summary':    'David今天意外說出了一些心裡的話，有些出乎自己意料，但感覺出奇地好。',
            'relation_A': 'David和Amy的互動更深了，他開始正視那種說不清的感受。',
            'relation_E': '',
        },
    },
    'Emma': {
        'day_1': {
            'emotion':    '平靜',
            'ltm_select': '[{"subject":"Emma","relation":"工作","object":"餐廳"}]',
            'summary':    'Emma今天工作忙碌，一切都很正常。',
            'relation_C': '',
        },
        'day_2': {
            'emotion':    '平靜',
            'ltm_select': '[{"subject":"Emma","relation":"工作","object":"餐廳"},{"subject":"Emma","relation":"遇見","object":"Claire","location":"餐廳"}]',
            'summary':    'Emma今天遇到了Claire，覺得她好像有些特別的地方，但說不清是什麼。',
            'relation_C': 'Emma覺得Claire最近好像有些不一樣，但說不清楚是什麼。',
        },
        'day_3': {
            'emotion':    '興奮',
            'ltm_select': '[{"subject":"Emma","relation":"注意到","object":"Claire的眼神"},{"subject":"Emma","relation":"感受","object":"被特別在意"}]',
            'summary':    'Emma開始感覺到Claire對她有特別的在意，這讓她有種說不清的感覺，心情也變好了。',
            'relation_C': 'Emma感受到Claire對她有特殊的關注，心裡有些期待但還不確定。',
        },
        'day_4': {
            'emotion':    '不安',
            'ltm_select': '[{"subject":"Emma","relation":"收到","object":"Claire的告白"},{"subject":"Emma","relation":"沉默","object":"不知道如何回應"}]',
            'summary':    'Emma收到了Claire的告白，沉默了很久，心裡還沒有想清楚自己的答案。',
            'relation_C': 'Claire向Emma告白了，Emma還在消化這件事，不確定自己的感受。',
        },
    },
}

_mock_call_counter = 0
_mock_current_day  = 1


def _make_dialogue_output(char_name: str, counter: int) -> str:
    """生成對話回應格式的輸出。"""
    responses = _DIALOGUE_RESPONSES.get(char_name, _DIALOGUE_RESPONSES['Amy'])
    text, thought = responses[counter % len(responses)]
    return (f'[ACTION]: {text}\n'
            f'[THOUGHT]: {thought}\n'
            f'[HAM]: [{{"subject":"{char_name}","relation":"對話","object":"對方","time":"現在"}}]\n[/HAM]')


class _SmartFusion:
    def fuse_inputs(self, text, image_inputs, **kw):
        # 保留 prompt 文字，供 generate() 判斷脈絡
        return {'_text': text}

    def generate(self, fused_inputs, gen_cfg=None):
        global _mock_call_counter
        _mock_call_counter += 1

        prompt    = fused_inputs.get('_text', '') if isinstance(fused_inputs, dict) else ''
        char_name = _extract_char_from_prompt(prompt)

        # 對話脈絡：input 含有「XXX對你說：」→ 回對話
        if '對你說：' in prompt or '開始對話' in prompt:
            return _make_dialogue_output(char_name, _mock_call_counter)

        # 有目標角色在場（prompt 含關係區塊）→ 一定選對話輸出
        if '【與對方的關係】' in prompt:
            dialogue_pool = _CHAR_OUTPUTS_WITH_TARGET.get(char_name, [])
            if dialogue_pool:
                return dialogue_pool[_mock_call_counter % len(dialogue_pool)]

        # 一般決策：角色專屬輸出池循環
        char_outputs = _CHAR_OUTPUTS.get(char_name, _CHAR_OUTPUTS['Amy'])
        return char_outputs[_mock_call_counter % len(char_outputs)]


class _MockText:
    @staticmethod
    def build_prompt(text, num_images=0, system_text=None):
        return type('P', (), {'prompt': text})()


class _MockVision:
    @staticmethod
    def encode(images):
        return type('V', (), {'to_dict': lambda self: {}})()


class SmartMockLoader:
    """
    智慧假模型：
    - 各角色有獨立輸出池，性格符合
    - 對話脈絡下自動回「對話:...」，不會說出不相干的動作
    - 睡眠濃縮按角色 × 天數返回不同情緒與記憶，避免所有人情緒相同
    """
    fusion = _SmartFusion()
    text   = _MockText()
    vision = _MockVision()

    def load(self):
        pass

    def is_loaded(self) -> bool:
        return True

    def make_model_fn(self, max_new_tokens=256, temperature=0.0):
        day_key = f'day_{_mock_current_day}'

        def model_fn(prompt: str) -> str:
            char_name  = _extract_char_from_prompt(prompt)
            char_sleep = _MOCK_SLEEP_BY_CHAR.get(char_name, _MOCK_SLEEP_BY_CHAR['Amy'])
            sleep_data = char_sleep.get(day_key, char_sleep.get('day_4', {}))

            if any(kw in prompt for kw in ['值得長期', '選出值得', '哪些值得']):
                return sleep_data.get('ltm_select', '[]')
            if any(kw in prompt for kw in ['總結', '摘要', '1-2句']):
                return sleep_data.get('summary', f'{char_name}今天過得平靜。')
            if '關係摘要' in prompt or '兩人關係' in prompt:
                # 找出 prompt 中提到的關係對象
                for code, key in [('Amy','relation_A'), ('David','relation_D'),
                                   ('Emma','relation_E'), ('Claire','relation_C'),
                                   ('Ben','relation_B')]:
                    if code in prompt and key in sleep_data:
                        val = sleep_data[key]
                        if val:
                            return val
                return sleep_data.get('summary', '兩人關係維持現狀。')
            if '情緒' in prompt:
                return sleep_data.get('emotion', '平靜')
            return sleep_data.get('emotion', '平靜')

        return model_fn


# ================================================================
# 四天情境設計
# ================================================================

# 每天 4 個時間段的場景設定
# setup_fn : 在執行此 slot 之前設定各角色的 location / action
# events   : [(code, scene, input_text, target_code), ...]
FOUR_DAY_PLAN = [
    # ── Day 1 ────────────────────────────────────────────────────
    {
        'day': 1,
        'slots': [
            {
                'time': '07:00',
                'desc': '日常起床，各自準備',
                'setup': {
                    # 初始狀態：所有人在自己家，current_action = 休息
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓三樓，早晨，陽光灑入', '', None),
                    ('B', '公寓二樓，醒來感覺還不錯', '', None),
                    ('C', '公寓一樓，鬧鐘響了', '', None),
                    ('D', '獨棟房子，安靜的早晨', '', None),
                    ('E', '獨棟房子2，準備上班', '', None),
                ],
            },
            {
                'time': '08:00',
                'desc': 'Amy 開始工作；David 到咖啡廳買咖啡，有緊急電話',
                'setup': {
                    # David 與 Amy 在同一地點（咖啡廳）
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '辦公室', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '工作',
                                  'C': '工作', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    # D runs first: K=0.6(緊急+休息) → DELIBERATE → 對話 → forward to A → cached
                    ('D', '咖啡廳，Amy正在吧台工作',
                     '緊急！Claire打來說辦公室出了狀況，但你還在咖啡廳', 'A'),
                    # A runs second: returns cached dialogue response from D's forward
                    ('A', '咖啡廳，早晨開店中，David走進來', '', None),
                    ('B', '超市，整理貨架', '', None),
                    ('C', '辦公室，等David回來', '', None),
                    ('E', '餐廳，開始備料', '', None),
                ],
            },
            {
                'time': '12:00',
                'desc': '午休；A 與 D 在咖啡廳繼續互動',
                'setup': {
                    'locations': {'A': '咖啡廳後場', 'B': '超市附近',
                                  'C': '公司附近', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '休息', 'B': '吃飯',
                                  'C': '吃飯', 'D': '工作', 'E': '工作'},
                },
                'events': [
                    # A: current_action="休息" + input "緊急" → K=0.6 → C>0.45 → Deliberate
                    ('A', '咖啡廳後場，午休，David敲門進來',
                     '緊急！剛才有客人抱怨你的咖啡品質。', 'D'),
                    ('D', '咖啡廳，剛整理完午餐，看到Amy', '', 'A'),
                    ('B', '超市附近，午餐時間', '', None),
                    ('C', '辦公室附近小店吃飯', '', None),
                    ('E', '餐廳，忙碌的午餐時段', '', None),
                ],
            },
            {
                'time': '18:00',
                'desc': '收工，各自返家',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '辦公室', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '整理', 'B': '整理',
                                  'C': '工作', 'D': '工作', 'E': '整理'},
                },
                'events': [
                    ('A', '咖啡廳，打烊準備整理', '', None),
                    ('B', '超市，收工整理', '', None),
                    ('C', '辦公室，還有報告沒寫完', 'David今天來晚了，讓人擔心', 'D'),
                    ('D', '辦公室，今天有點特別', '', None),
                    ('E', '餐廳，最後的清場', '', None),
                ],
            },
        ],
    },

    # ── Day 2 ────────────────────────────────────────────────────
    {
        'day': 2,
        'slots': [
            {
                'time': '07:00',
                'desc': '第二天早晨（有 LTM）；「突然」場景 → S 觸發複雜思考',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    # "突然" in scene → S=0.25（Amy 有 LTM）+ "心情有些" → K=0.25 → C>0.45 → Deliberate
                    ('A', '公寓，早晨突然想到昨天David說的話，心情有些複雜', '', None),
                    ('B', '公寓二樓，突然想去Amy那裡買杯咖啡', '', None),
                    ('C', '公寓一樓，突然收到David的訊息說要早點到公司', '', None),
                    ('D', '獨棟房子，突然回想起昨天在咖啡廳的事', '', None),
                    ('E', '獨棟房子2，突然有種預感今天會有不一樣的事', '', None),
                ],
            },
            {
                'time': '08:00',
                'desc': 'Ben 到咖啡廳；A 與 B 互動',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '咖啡廳',
                                  'C': '辦公室', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '休息',
                                  'C': '工作', 'D': '工作', 'E': '工作'},
                },
                'events': [
                    ('A', '咖啡廳，Ben突然走進來笑著打招呼', 'Ben走進來說：Amy早安！給我一杯最喜歡的！', 'B'),
                    # B: current_action="休息" + input "緊急" → K=0.6 → 複雜思考
                    ('B', '咖啡廳，Amy正在工作',
                     '緊急！超市電話說今天要早到，但你已經來咖啡廳了', None),
                    ('C', '辦公室，等David', '', None),
                    ('D', '辦公室，處理昨天積累的文件', '', None),
                    ('E', '餐廳，準備午餐食材', '', None),
                ],
            },
            {
                'time': '12:00',
                'desc': 'Claire 與 Emma 在餐廳相遇',
                'setup': {
                    'locations': {'A': '咖啡廳後場', 'B': '超市',
                                  'C': '餐廳', 'D': '公司附近', 'E': '餐廳'},
                    'actions':   {'A': '休息', 'B': '工作',
                                  'C': '吃飯', 'D': '吃飯', 'E': '工作'},
                },
                'events': [
                    ('A', '咖啡廳後場，午休，想著昨天的事', '', None),
                    ('B', '超市，整理下午的工作', '', None),
                    # C: at 餐廳 + E: at 餐廳 → co-location
                    # "不知道如何" in STRONG_EMOTION → K=0.45≥0.4 → force DELIBERATE
                    ('C', '餐廳，看到Emma，突然不知道如何開口，心跳加速', '', 'E'),
                    ('E', '餐廳，忙碌工作中，Claire進來了',
                     '緊急！廚房突然說有人要大批訂單，需要你確認', None),
                    ('D', '公司附近小餐館，獨自用餐', '', None),
                ],
            },
            {
                'time': '18:00',
                'desc': '傍晚活動，回家前的心情',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '辦公室', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '整理',
                                  'C': '工作', 'D': '工作', 'E': '整理'},
                },
                'events': [
                    ('A', '咖啡廳，打烊前，今天心情特別', '', None),
                    ('B', '超市，今天沒見到Amy感到有些失落', '', None),
                    # C & D at 辦公室
                    # "不知道如何" → K=0.45≥0.4 → force DELIBERATE
                    ('C', '辦公室，David還在加班，不知道如何開口關心他', 'David你今天有沒有事？看起來心事重重', 'D'),
                    ('D', '辦公室，Claire走過來', '', 'C'),
                    ('E', '餐廳，收拾最後的東西', '', None),
                ],
            },
        ],
    },

    # ── Day 3 ────────────────────────────────────────────────────
    {
        'day': 3,
        'slots': [
            {
                'time': '07:00',
                'desc': '第三天：LTM 更豐富，情緒更複雜',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，早晨突然想到這兩天積累的情緒', '', None),
                    ('B', '公寓二樓，今天決定要主動找Amy說話', '', None),
                    ('C', '公寓，突然想到Emma昨天的樣子心情變好了', '', None),
                    ('D', '獨棟房子，難得悠閒早晨，突然想起Amy今天在不在', '', None),
                    ('E', '獨棟房子2，突然想到Claire昨天來餐廳的事', '', None),
                ],
            },
            {
                'time': '09:00',
                'desc': 'David 再次去咖啡廳；多組互動',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '咖啡廳',
                                  'C': '辦公室', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '休息',
                                  'C': '工作', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    ('A', '咖啡廳，David和Ben幾乎同時進來，有點意外', '', 'D'),
                    ('B', '咖啡廳，Amy正在忙，有點不好意思打擾',
                     '緊急！電話說超市今天有緊急盤點，你必須去', None),
                    ('D', '咖啡廳，Amy臉上有點疲憊的神情',
                     '緊急！財務說今天要討論一個重要的合約', None),
                    ('C', '辦公室，等老闆', '', None),
                    ('E', '餐廳，又想到昨天Claire來的事', '', None),
                ],
            },
            {
                'time': '13:00',
                'desc': '下午，Claire 主動找 Emma；A 內心獨白',
                'setup': {
                    'locations': {'A': '咖啡廳後場', 'B': '超市',
                                  'C': '餐廳', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '休息', 'B': '工作',
                                  'C': '休息', 'D': '工作', 'E': '休息'},
                },
                'events': [
                    ('A', '咖啡廳後場，安靜的下午，突然覺得心裡悶悶的', '', None),
                    ('B', '超市，工作中想著下次要怎麼跟Amy說', '', None),
                    # C & E 在餐廳
                    # "不知道如何" → K=0.45≥0.4 → force DELIBERATE
                    ('C', '餐廳，不知道如何開口，想找Emma說說話', '我想跟你說昨天廚房的事情', 'E'),
                    ('E', '餐廳，Claire突然來訪',
                     '緊急！剛才有食客反映餐廳的問題，Claire你有聽說嗎', 'C'),
                    ('D', '辦公室，看著合約上Amy的咖啡廳位置，想起了什麼', '', None),
                ],
            },
            {
                'time': '18:00',
                'desc': '第三天收工，情緒沉澱',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '辦公室', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '整理', 'B': '整理',
                                  'C': '工作', 'D': '工作', 'E': '整理'},
                },
                'events': [
                    ('A', '咖啡廳，打烊，今天心情說不清', '', None),
                    ('B', '超市，收工，帶著一絲失落', '', None),
                    ('C', '辦公室，今天收穫很多，心情輕鬆一些', '', None),
                    ('D', '辦公室，打算明天去一趟咖啡廳', '', None),
                    ('E', '餐廳，想著明天和Claire的事', '', None),
                ],
            },
        ],
    },

    # ── Day 4 ────────────────────────────────────────────────────
    {
        'day': 4,
        'slots': [
            {
                'time': '07:00',
                'desc': '第四天：全面複雜思考，情緒發酵',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，這幾天積累的情緒突然湧上來，思緒一片混亂', '', None),
                    ('B', '公寓二樓，今天決定要直接說清楚了，突然鼓起勇氣', '', None),
                    ('C', '公寓，突然想起昨天Emma眼神裡的溫柔，心跳加速', '', None),
                    ('D', '獨棟房子，今天一定要去咖啡廳，突然有了某種使命感', '', None),
                    ('E', '獨棟房子2，突然覺得Claire真的不一樣，心裡說不清', '', None),
                ],
            },
            {
                'time': '09:00',
                'desc': '最終對話高峰：多組相遇',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '咖啡廳',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    # D 先跑：K=0.6(緊急) → DELIBERATE → 對話 → forward to A → cached
                    ('D', '咖啡廳，決定今天要對Amy說些什麼',
                     '緊急！Claire說有合約要在中午前完成簽署', 'A'),
                    # A 後跑：命中 D 的 forward cache → 正確對話配對
                    ('A', '咖啡廳，David 和 Ben 都進來了，氣氛有點意外', '', 'D'),
                    ('B', '咖啡廳，鼓起勇氣想跟Amy說話',
                     '緊急！主管說今天必須早點回超市，但你偏偏在咖啡廳', None),
                    # C & E 在餐廳
                    # "第一次"→S=0.5，"心跳加速"→K=0.25；C=0.3+0.1+0.15=0.55≥0.5→deliberate
                    ('C', '餐廳，Claire第一次帶著花束走進來，心跳加速', '我有件事想對你說', 'E'),
                    ('E', '餐廳，看到Claire帶花進來，突然心跳加速',
                     '緊急！老闆說今天餐廳有媒體採訪', 'C'),
                ],
            },
            {
                'time': '13:00',
                'desc': '事件發展，各自思考',
                'setup': {
                    'locations': {'A': '咖啡廳後場', 'B': '超市',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '休息', 'B': '工作',
                                  'C': '工作', 'D': '工作', 'E': '工作'},
                },
                'events': [
                    ('A', '咖啡廳後場，David剛才的話讓Amy思緒很亂', '', None),
                    ('B', '超市，今天又沒說出口，有點難受', '', None),
                    # C: "不知道如何" → K=0.45≥0.4 → force DELIBERATE → 對話 → forward E
                    ('C', '餐廳，說出了心裡話，不知道如何面對Emma的眼神', '', 'E'),
                    # E: "不知道如何" → K=0.45 (backup, also cache from C's forward)
                    ('E', '餐廳，Claire說的話讓Emma沉默了很久，不知道如何回應', '', 'C'),
                    ('D', '咖啡廳，待在這裡，思考剛才跟Amy的對話', '', None),
                ],
            },
            {
                'time': '20:00',
                'desc': '第四天夜晚，各自回家沉澱',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，夜晚突然一個人靜靜坐著，想了很多', '', None),
                    ('B', '公寓二樓，夜晚，心情有些沉重但也有些釋懷', '', None),
                    ('C', '公寓，今天的事情讓Claire心情很複雜但也充實', '', None),
                    ('D', '獨棟房子，今天意外說了很多，有點意外自己', '', None),
                    ('E', '獨棟房子2，夜裡突然想通了某些事', '', None),
                ],
            },
        ],
    },

    # ── Day 5 ────────────────────────────────────────────────────
    {
        'day': 5,
        'slots': [
            {
                'time': '07:00',
                'desc': '第五天早晨：昨天的情緒持續發酵',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    # A threshold=0.45，"突然"→S=0.25 + "心跳"→K=0.25 → C=0.55>0.45→DELIBERATE
                    ('A', '公寓，突然想起David昨天的話，心跳加速', '', None),
                    ('B', '公寓二樓，昨天又沒說出口，今天再試試', '', None),
                    ('C', '公寓，想著Emma的眼神，有點難以入睡', '', None),
                    # D threshold=0.55，"突然"→S=0.25 + "期待"→K=0.25 → C=0.4+0.1+0.05=0.55→DELIBERATE
                    ('D', '獨棟房子，突然有種期待，想再去見Amy', '', None),
                    ('E', '獨棟房子2，突然想通了一件事，心情輕鬆', '', None),
                ],
            },
            {
                'time': '09:00',
                'desc': 'David再度拜訪；Ben嘗試；Claire詢問Emma',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '咖啡廳',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '休息',
                                  'C': '工作', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    # D先：K=0.6(緊急+休息) → DELIBERATE → 對話 → forward A
                    ('D', '咖啡廳，Amy在工作，不知道如何再開口',
                     '緊急！合約今天截止，但David還是先來咖啡廳', 'A'),
                    ('A', '咖啡廳，David今天又來了，突然心動', '', 'D'),
                    # B：K=0.6 → DELIBERATE → 對話 → forward A（A已跑 → cache wasted）
                    ('B', '咖啡廳，Amy和David都在，Ben有些失落',
                     '緊急！超市主管又催了，但Ben想先說清楚', None),
                    # C先：K=0.45("不知道如何") → force DELIBERATE → forward E
                    ('C', '餐廳，不知道如何問Emma有沒有想好', '', 'E'),
                    ('E', '餐廳，Claire明顯在等答案，不知道如何回應', '', 'C'),
                ],
            },
            {
                'time': '13:00',
                'desc': '下午：各自沉思，Claire繼續等Emma',
                'setup': {
                    'locations': {'A': '咖啡廳後場', 'B': '超市',
                                  'C': '餐廳', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '休息', 'B': '工作',
                                  'C': '工作', 'D': '工作', 'E': '工作'},
                },
                'events': [
                    ('A', '咖啡廳後場，和David的對話讓Amy思緒一片混亂', '', None),
                    ('B', '超市，今天又沒有機會，有點灰心', '', None),
                    ('C', '餐廳，不知道如何繼續等待Emma的回應', '', 'E'),
                    ('E', '餐廳，Claire還在等，Emma不知道如何開口', '', 'C'),
                    ('D', '辦公室，想著今天和Amy說話的感覺', '', None),
                ],
            },
            {
                'time': '20:00',
                'desc': '第五天夜晚：整理心情',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，夜晚，突然心情有些複雜但也期待', '', None),
                    ('B', '公寓二樓，今天靜靜地想了很多', '', None),
                    ('C', '公寓，一個人靜靜等待Emma的回答', '', None),
                    ('D', '獨棟房子，今天說的話感覺還不夠', '', None),
                    ('E', '獨棟房子2，突然有了一個新的想法', '', None),
                ],
            },
        ],
    },

    # ── Day 6 ────────────────────────────────────────────────────
    {
        'day': 6,
        'slots': [
            {
                'time': '07:00',
                'desc': '第六天早晨：感情更加清晰',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，突然想好了一件事，心情有些輕鬆', '', None),
                    ('B', '公寓二樓，今天決定要更直接一點', '', None),
                    ('C', '公寓，突然想到一個辦法，心跳加速', '', None),
                    ('D', '獨棟房子，突然決定今天要說得更清楚', '', None),
                    ('E', '獨棟房子2，想著Claire，突然有種溫暖的感覺', '', None),
                ],
            },
            {
                'time': '09:00',
                'desc': 'David說得更直接；B去超市；Claire再問Emma',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '工作',
                                  'C': '工作', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    # D先：K=0.6 → DELIBERATE → 對話 → forward A
                    ('D', '咖啡廳，今天要對Amy說得更清楚，不知道如何開口',
                     '緊急！Claire說今天合約最後期限，但David還是先來這裡', 'A'),
                    ('A', '咖啡廳，David今天感覺很不一樣，突然心動', '', 'D'),
                    # B 在超市，不在咖啡廳
                    ('B', '超市，沒去咖啡廳，專心工作，心裡有些想法', '', None),
                    ('C', '餐廳，不知道如何等到Emma開口', '', 'E'),
                    ('E', '餐廳，Claire又來了，不知道如何面對這份期待', '', 'C'),
                ],
            },
            {
                'time': '13:00',
                'desc': '下午：Amy主動回應David；Claire等Emma的答案',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '工作',
                                  'C': '休息', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    # A先主動找D：A threshold=0.45，"突然"→S=0.25，C>0.45→DELIBERATE
                    ('A', '咖啡廳，突然下定決心，想跟David說些什麼', '', 'D'),
                    # D後：得到A的forward cache
                    ('D', '咖啡廳，Amy今天感覺有些不一樣', '', 'A'),
                    ('B', '超市，想著明天也許可以鼓起勇氣', '', None),
                    ('C', '餐廳，不知道如何再問Emma一次', '', 'E'),
                    ('E', '餐廳，Claire的眼神讓Emma不知道如何回避', '', 'C'),
                ],
            },
            {
                'time': '20:00',
                'desc': '第六天夜晚：各自回味今天',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，今天突然主動說了什麼，心情有些緊張', '', None),
                    ('B', '公寓二樓，今天在超市一整天，決定明天去咖啡廳', '', None),
                    ('C', '公寓，一個人靜靜等待，心情複雜', '', None),
                    ('D', '獨棟房子，今天Amy說的話讓David思緒一片混亂', '', None),
                    ('E', '獨棟房子2，突然想到一個答案，心跳加速', '', None),
                ],
            },
        ],
    },

    # ── Day 7 ────────────────────────────────────────────────────
    {
        'day': 7,
        'slots': [
            {
                'time': '07:00',
                'desc': '第七天早晨：Ben決定今天要說清楚',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，突然想著這幾天的事，心情有些複雜', '', None),
                    # B threshold=0.6，"不知道如何" → K=0.45≥0.4 → force DELIBERATE，但獨處→fallback
                    ('B', '公寓二樓，不知道如何說出口，但今天要試試', '', None),
                    ('C', '公寓，Emma給了個不明確的眼神，心情很複雜', '', None),
                    ('D', '獨棟房子，今天有重要的事要處理，可能沒法去咖啡廳', '', None),
                    ('E', '獨棟房子2，突然想好了怎麼回應Claire', '', None),
                ],
            },
            {
                'time': '09:00',
                'desc': 'David去辦公室；Ben終於和Amy說話；Claire等到了Emma',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '咖啡廳',
                                  'C': '餐廳', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '休息',
                                  'C': '工作', 'D': '工作', 'E': '工作'},
                },
                'events': [
                    # B先：K=0.6(緊急+休息) → DELIBERATE → 對話 → forward A（D不在咖啡廳）
                    ('B', '咖啡廳，今天只有Amy，不知道如何開口',
                     '緊急！超市主管說今天一定要早回，但Ben還是先來了', 'A'),
                    # A後：得到B的forward cache
                    ('A', '咖啡廳，今天David沒來，只有Ben，突然有點意外', '', 'B'),
                    # D在辦公室，不在咖啡廳
                    ('D', '辦公室，今天要處理積累的工作', '', None),
                    ('C', '餐廳，不知道如何開口，今天感覺有點不一樣', '', 'E'),
                    ('E', '餐廳，今天Claire來了，Emma突然決定說清楚，不知道如何開口',
                     '', 'C'),
                ],
            },
            {
                'time': '13:00',
                'desc': '下午：Ben說完了感覺輕鬆；Emma終於給Claire一個答案',
                'setup': {
                    'locations': {'A': '咖啡廳後場', 'B': '超市',
                                  'C': '餐廳', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '休息', 'B': '工作',
                                  'C': '休息', 'D': '工作', 'E': '工作'},
                },
                'events': [
                    ('A', '咖啡廳後場，Ben今天說的話讓Amy有點意外', '', None),
                    ('B', '超市，說出來了，感覺輕鬆了不少', '', None),
                    # E先主動找C說答案：E threshold=0.5，"不知道如何"→K=0.45→force DELIBERATE
                    ('E', '餐廳，不知道如何跟Claire說清楚自己的心情，但今天要開口', '', 'C'),
                    ('C', '餐廳，突然感到有些緊張，不知道Emma會說什麼', '', 'E'),
                    ('D', '辦公室，想著Amy和David的事，有點走神', '', None),
                ],
            },
            {
                'time': '20:00',
                'desc': '第七天夜晚：各自消化今天的對話',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，今天Ben說的話讓Amy有些想法', '', None),
                    ('B', '公寓二樓，今天說出來了，感覺鬆了一口氣', '', None),
                    ('C', '公寓，Emma給的答案讓Claire心情很複雜但也很開心', '', None),
                    ('D', '獨棟房子，今天在辦公室，突然很想去看Amy', '', None),
                    ('E', '獨棟房子2，今天說清楚了，感覺前所未有的輕鬆', '', None),
                ],
            },
        ],
    },

    # ── Day 8 ────────────────────────────────────────────────────
    {
        'day': 8,
        'slots': [
            {
                'time': '07:00',
                'desc': '第八天早晨：情感達到最高點',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    # A threshold=0.45，"突然"→S=0.25 + "心跳"→K=0.25 → C=0.55>0.45→DELIBERATE
                    ('A', '公寓，突然決定今天要對David說些什麼，心跳加速', '', None),
                    ('B', '公寓二樓，說出來以後，感覺自己清醒了很多', '', None),
                    # C threshold=0.5，"不知道如何" → K=0.45 → force DELIBERATE
                    ('C', '公寓，不知道如何消化Emma昨天說的話，心情很複雜', '', None),
                    ('D', '獨棟房子，今天一定要跟Amy說清楚，突然有種使命感', '', None),
                    ('E', '獨棟房子2，昨天說完感覺很輕鬆，突然想好好感謝Claire', '', None),
                ],
            },
            {
                'time': '09:00',
                'desc': 'Amy主動找David；Emma感謝Claire',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '工作',
                                  'C': '休息', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    # A先主動：A threshold=0.45，"突然"→S=0.25 → C=0.475>0.45→DELIBERATE
                    ('A', '咖啡廳，今天突然決定，準備主動跟David說', '', 'D'),
                    # D後：得到A的forward cache
                    ('D', '咖啡廳，今天感覺Amy有什麼話想說', '', 'A'),
                    ('B', '超市，今天在忙，心情卻很輕鬆', '', None),
                    # E先：E threshold=0.5，"不知道如何" → K=0.45 → force DELIBERATE
                    ('E', '餐廳，不知道如何感謝Claire昨天的勇氣', '', 'C'),
                    ('C', '餐廳，Emma突然找來，突然有點緊張', '', 'E'),
                ],
            },
            {
                'time': '13:00',
                'desc': '下午：David回應Amy；Claire和Emma的關係更進一步',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '工作',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    # D先：K=0.6(緊急+休息) → forward A
                    ('D', '咖啡廳，Amy今天說的話讓David不知道如何回應，但一定要說',
                     '緊急！Claire催著合約，但David要先跟Amy說完', 'A'),
                    ('A', '咖啡廳，David今天感覺要說什麼重要的事，突然心跳加速', '', 'D'),
                    ('B', '超市，工作時心情輕鬆多了', '', None),
                    ('C', '餐廳，不知道如何接受這份突如其來的幸福', '', 'E'),
                    ('E', '餐廳，Claire說的話讓Emma不知道如何承受這份感動', '', 'C'),
                ],
            },
            {
                'time': '20:00',
                'desc': '第八天夜晚：情感高峰後的沉澱',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，今天說了很多，心裡既緊張又輕鬆', '', None),
                    ('B', '公寓二樓，今天不在咖啡廳，但心情很平靜', '', None),
                    ('C', '公寓，今天和Emma的對話讓Claire感到幸福', '', None),
                    ('D', '獨棟房子，今天Amy的眼神讓David思緒一片混亂', '', None),
                    ('E', '獨棟房子2，今天說了很多，感覺前所未有的輕鬆', '', None),
                ],
            },
        ],
    },

    # ── Day 9 ────────────────────────────────────────────────────
    {
        'day': 9,
        'slots': [
            {
                'time': '07:00',
                'desc': '第九天早晨：關係進入新的狀態',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，突然覺得和David的關係不一樣了，心裡有點暖', '', None),
                    ('B', '公寓二樓，說出來之後，感覺前所未有的輕鬆', '', None),
                    ('C', '公寓，突然收到Emma的訊息，心跳加速', '', None),
                    ('D', '獨棟房子，突然想著Amy，覺得今天是新的開始', '', None),
                    ('E', '獨棟房子2，昨天之後，不知道如何面對新的自己', '', None),
                ],
            },
            {
                'time': '09:00',
                'desc': 'David和Amy繼續；Ben和Amy友好相遇',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '咖啡廳',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '休息',
                                  'C': '工作', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    # D先：K=0.6 → forward A
                    ('D', '咖啡廳，昨天說了很多，今天想繼續，不知道如何開口',
                     '緊急！有個會議等著，但David想先跟Amy說話', 'A'),
                    ('A', '咖啡廳，David今天表情很不一樣，突然感到溫暖', '', 'D'),
                    # B也在，但Amy已跑過 → B forward goes to A, wasted
                    ('B', '咖啡廳，今天心情輕鬆，想跟Amy打個招呼',
                     '緊急！剛才朋友說要一起吃飯，但Ben想先在咖啡廳待一下', None),
                    ('C', '餐廳，不知道如何跟Emma說接下來的事', '', 'E'),
                    ('E', '餐廳，Claire又來了，不知道如何表達這幾天的心情', '', 'C'),
                ],
            },
            {
                'time': '13:00',
                'desc': '下午：Amy和Ben友好地說說話',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '咖啡廳',
                                  'C': '餐廳', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '休息',
                                  'C': '休息', 'D': '工作', 'E': '工作'},
                },
                'events': [
                    # B先找Amy說話（D在辦公室，Amy可以配對B）
                    ('B', '咖啡廳，今天David不在，不知道如何跟Amy說說話', '', 'A'),
                    ('A', '咖啡廳，Ben今天感覺不一樣，突然有點意外', '', 'B'),
                    ('D', '辦公室，心情好多了，想著Amy', '', None),
                    ('C', '餐廳，不知道如何開口說接下來想一起做的事', '', 'E'),
                    ('E', '餐廳，Claire說的話讓Emma不知道如何拒絕這份期待', '', 'C'),
                ],
            },
            {
                'time': '20:00',
                'desc': '第九天夜晚：一切慢慢清晰',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，今天和Ben聊了聊，心情輕鬆了不少', '', None),
                    ('B', '公寓二樓，和Amy說說話，感覺這樣也很好', '', None),
                    ('C', '公寓，突然覺得和Emma的事情很美好', '', None),
                    ('D', '獨棟房子，今天Amy說的話讓David感到很安心', '', None),
                    ('E', '獨棟房子2，一切慢慢清晰，心情前所未有的輕鬆', '', None),
                ],
            },
        ],
    },

    # ── Day 10 ───────────────────────────────────────────────────
    {
        'day': 10,
        'slots': [
            {
                'time': '07:00',
                'desc': '第十天早晨：新的平衡',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    # A threshold=0.45，"突然"→S=0.25 → C>0.45→DELIBERATE
                    ('A', '公寓，突然覺得一切都不一樣了，心情前所未有的平靜', '', None),
                    ('B', '公寓二樓，突然覺得一切都清晰了，很輕鬆', '', None),
                    ('C', '公寓，和Emma的事已經清晰了，心情很好', '', None),
                    ('D', '獨棟房子，突然很想去看Amy，心情有點期待', '', None),
                    ('E', '獨棟房子2，一切都在新的起點上，突然感到滿足', '', None),
                ],
            },
            {
                'time': '09:00',
                'desc': '新的日常開始：Amy和David的第一個平常的早晨',
                'setup': {
                    'locations': {'A': '咖啡廳', 'B': '超市',
                                  'C': '餐廳', 'D': '咖啡廳', 'E': '餐廳'},
                    'actions':   {'A': '工作', 'B': '工作',
                                  'C': '工作', 'D': '休息', 'E': '工作'},
                },
                'events': [
                    # D先：K=0.6(緊急+休息) → forward A（溫和的日常對話）
                    ('D', '咖啡廳，一切都不同了，不知道如何開始這個新的日常',
                     '緊急！有個合約需要確認，但David先想和Amy說說話', 'A'),
                    ('A', '咖啡廳，David今天的樣子讓Amy突然覺得很幸福', '', 'D'),
                    ('B', '超市，今天在工作，心情很平靜', '', None),
                    ('C', '餐廳，不知道如何跟Emma說今天的計畫', '', 'E'),
                    ('E', '餐廳，Claire說的話讓Emma不知道如何表達喜悅', '', 'C'),
                ],
            },
            {
                'time': '13:00',
                'desc': '下午：平靜的新日常',
                'setup': {
                    'locations': {'A': '咖啡廳後場', 'B': '超市',
                                  'C': '餐廳', 'D': '辦公室', 'E': '餐廳'},
                    'actions':   {'A': '休息', 'B': '工作',
                                  'C': '休息', 'D': '工作', 'E': '工作'},
                },
                'events': [
                    ('A', '咖啡廳後場，今天感覺和之前的每一天都不一樣', '', None),
                    ('B', '超市，工作中心情很輕鬆，偶爾想起這幾天', '', None),
                    ('C', '餐廳，不知道如何跟Emma說下一步的計畫', '', 'E'),
                    ('E', '餐廳，Claire問的問題讓Emma不知道如何回答但心裡很開心',
                     '', 'C'),
                    ('D', '辦公室，工作時也覺得心情特別好', '', None),
                ],
            },
            {
                'time': '20:00',
                'desc': '第十天夜晚：模擬結束，新的開始',
                'setup': {
                    'locations': {'A': '公寓', 'B': '公寓二樓',
                                  'C': '公寓一樓', 'D': '獨棟房子', 'E': '獨棟房子 2'},
                    'actions':   {'A': '休息', 'B': '休息',
                                  'C': '休息', 'D': '休息', 'E': '休息'},
                },
                'events': [
                    ('A', '公寓，十天下來，心裡有好多感受，但都是好的', '', None),
                    ('B', '公寓二樓，這十天改變了很多，但也學到了很多', '', None),
                    ('C', '公寓，和Emma的事讓這十天充滿了色彩', '', None),
                    ('D', '獨棟房子，突然覺得生活很美好，因為有了Amy', '', None),
                    ('E', '獨棟房子2，這十天有Claire陪伴，感覺很幸運', '', None),
                ],
            },
        ],
    },
]


# ================================================================
# 異常偵測
# ================================================================

def detect_anomalies(manager) -> list:
    """檢查各角色狀態是否有邏輯異常或數值異常。"""
    issues = []
    for code in manager.all_codes():
        agent = manager.get_agent(code)
        char  = manager.get_character(code)
        name  = char.name

        # LTM strength 範圍
        for p in agent.ltm.get_all():
            if not (0.0 <= p['strength'] <= 1.0):
                issues.append(f'[{name}] LTM {p["id"]} strength={p["strength"]:.3f} 超出 [0,1]')

        # LTM ID 唯一性
        ids = [p['id'] for p in agent.ltm.get_all()]
        if len(ids) != len(set(ids)):
            dupes = [i for i in ids if ids.count(i) > 1]
            issues.append(f'[{name}] LTM ID 重複：{dupes}')

        # STM 容量
        if agent.stm.count() > agent.stm.capacity():
            issues.append(
                f'[{name}] STM 超出容量：{agent.stm.count()} > {agent.stm.capacity()}'
            )

        # 情緒合法性
        if char.emotion not in VALID_EMOTIONS:
            issues.append(f'[{name}] 非法情緒：{char.emotion}')

        # 位置合法性
        loc = char.current_location
        if loc and loc not in VALID_LOCATIONS:
            issues.append(f'[{name}] 非法位置：{loc}')

        # day 合理性
        if char.day < 1 or char.day > 30:
            issues.append(f'[{name}] day 異常：{char.day}')

        # LTM access_count 合理性
        for p in agent.ltm.get_all():
            if p.get('access_count', 0) < 0:
                issues.append(f'[{name}] LTM {p["id"]} access_count < 0')

    return issues


# ================================================================
# 主模擬流程
# ================================================================

def run_four_day_simulation(loader, manager, clock, max_days: int = None):
    """
    執行多天完整模擬：
    - 每天 4 個時間段
    - 每個時間段設定角色位置/行動後執行 step_character
    - 每天結束後全員睡眠 → STM→LTM 濃縮
    max_days : 最多跑幾天（None = 全部）
    """
    global _mock_current_day

    from config.world_config import CHARACTER_NAMES as CN
    from core.markov import format_probs_display

    day_records = []  # 報告資料

    plan = FOUR_DAY_PLAN if max_days is None else FOUR_DAY_PLAN[:max_days]
    for day_plan in plan:
        day_num = day_plan['day']
        _mock_current_day = day_num

        section(f'Day {day_num} 模擬')
        day_data = {'day': day_num, 'slots': [], 'sleep': {}, 'anomalies': []}

        for slot in day_plan['slots']:
            slot_time = slot['time']
            slot_desc = slot['desc']
            setup     = slot.get('setup', {})
            events    = slot['events']

            subsection(f'{slot_time} — {slot_desc}')
            print(f'    時鐘：{clock.scene_prefix()}')

            # 清除上一個時間段遺留的對話快取，避免污染本段決策
            manager.clear_conversation_cache()

            # 套用場景設定（位置、行動）
            for code, loc in setup.get('locations', {}).items():
                manager.get_character(code).current_location = loc
            for code, act in setup.get('actions', {}).items():
                manager.get_character(code).current_action = act

            slot_data = {
                'time':    slot_time,
                'desc':    slot_desc,
                'clock':   clock.scene_prefix(),
                'agents':  {},
                'conversations': [],
            }

            for code, scene, input_text, target_code in events:
                char   = manager.get_character(code)
                agent  = manager.get_agent(code)
                prev_stm = agent.stm.count()
                prev_ltm = agent.ltm.count()

                try:
                    result = manager.step_character(
                        code        = code,
                        scene       = scene,
                        input_text  = input_text,
                        target_code = target_code,
                    )
                except Exception as e:
                    print(f'    [{code}] ERROR: {e}')
                    traceback.print_exc()
                    _results['fail'] += 1
                    continue

                mode  = result['mode']
                c_val = result['confusion']['C']
                u_val = result['confusion']['U']
                k_val = result['confusion']['K']
                s_val = result['confusion']['S']
                act   = result['action']
                probs = result.get('action_probs', {})
                thought = result.get('thought', '')
                ham_n   = len(result.get('ham', []))

                # 格式化顯示
                mode_icon = '🔵' if mode == 'markov' else '🟣'
                print(f'    {mode_icon} [{code}:{char.name}] {mode.upper():12s} '
                      f'C={c_val:.3f}(U={u_val:.2f},K={k_val:.2f},S={s_val:.2f})  '
                      f'→ {act}')
                if probs and mode == 'markov':
                    print(f'       機率: {format_probs_display(probs, top_n=4)}')
                if thought and mode == 'deliberate':
                    print(f'       想法: {thought[:70]}{"..." if len(thought) > 70 else ""}')
                if ham_n > 0:
                    print(f'       HAM: {ham_n} 筆  STM: {prev_stm}→{agent.stm.count()}  LTM: {prev_ltm}→{agent.ltm.count()}')
                else:
                    print(f'       STM: {prev_stm}→{agent.stm.count()}  LTM: {prev_ltm}→{agent.ltm.count()}')

                slot_data['agents'][code] = {
                    'name':         char.name,
                    'action':       act,
                    'verb':         result['verb'],
                    'mode':         mode,
                    'confusion_C':  round(c_val, 4),
                    'confusion_U':  round(u_val, 4),
                    'confusion_K':  round(k_val, 4),
                    'confusion_S':  round(s_val, 4),
                    'action_probs': probs,
                    'thought':      thought,
                    'ham_count':    ham_n,
                    'stm_count':    agent.stm.count(),
                    'ltm_count':    agent.ltm.count(),
                    'location':     char.current_location,
                    'emotion':      char.emotion,
                    'should_sleep': result.get('should_sleep', False),
                }
                _results['pass'] += 1

            # 收集本時間段完成的對話配對
            slot_data['conversations'] = manager.pop_dialogue_history()

            day_data['slots'].append(slot_data)
            clock.tick()

        # ── 全員睡眠 ────────────────────────────────────────────
        subsection(f'Day {day_num} — 全員睡眠濃縮')

        for code in sorted(manager.all_codes()):
            agent = manager.get_agent(code)
            char  = manager.get_character(code)
            stm_b = agent.stm.count()
            ltm_b = agent.ltm.count()
            emo_b = char.emotion
            day_b = char.day

            try:
                sleep_res = manager._do_sleep(code)
            except Exception as e:
                print(f'    [{code}] Sleep ERROR: {e}')
                traceback.print_exc()
                _results['fail'] += 1
                continue

            print(f'    [{code}:{char.name}] STM:{stm_b}→{agent.stm.count()}  '
                  f'LTM:{ltm_b}→{agent.ltm.count()}  '
                  f'情緒:{emo_b}→{char.emotion}  '
                  f'+{sleep_res["new_propositions"]}命題  '
                  f'剪枝:{sleep_res["ltm_pruned"]}  '
                  f'Day:{day_b}→{char.day}')
            if sleep_res.get('summary'):
                print(f'       摘要: {sleep_res["summary"][:80]}')

            day_data['sleep'][code] = {
                'name':          char.name,
                'new_props':     sleep_res['new_propositions'],
                'ltm_total':     sleep_res['ltm_total'],
                'ltm_pruned':    sleep_res['ltm_pruned'],
                'stm_before':    stm_b,
                'ltm_before':    ltm_b,
                'ltm_after':     agent.ltm.count(),
                'emotion_before': emo_b,
                'emotion_after':  char.emotion,
                'summary':        sleep_res.get('summary', ''),
                'day_after':      char.day,
            }

        # ── 異常偵測 ──────────────────────────────────────────────
        issues = detect_anomalies(manager)
        if issues:
            print(f'\n  ⚠️  Day {day_num} 發現異常：')
            for iss in issues:
                print(f'    • {iss}')
        else:
            print(f'  ✓ Day {day_num} 無異常')

        day_data['anomalies'] = issues
        day_records.append(day_data)

    return day_records


# ================================================================
# 驗證函式
# ================================================================

def validate_model() -> bool:
    from config.model_config import MODEL_ID
    section('模型下載與驗證')

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as e:
        print(f'  [ERROR] 缺少依賴：{e}')
        _results['fail'] += 1
        return False

    try:
        from config.model_config import MODEL_NUM_CROPS
        import time
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, num_crops=MODEL_NUM_CROPS
        )
        print(f'  processor 載入完成（{time.time()-t0:.1f}s）')
    except Exception as e:
        print(f'  [ERROR] Processor 載入失敗：{e}')
        _results['fail'] += 1
        return False

    try:
        from model.model_loader import pick_device
        device = pick_device()
        t1 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True,
            torch_dtype='auto', _attn_implementation='eager'
        )
        model = model.to(device)
        print(f'  model 載入完成（{time.time()-t1:.1f}s），裝置：{device}')
    except Exception as e:
        print(f'  [ERROR] Model 載入失敗：{e}')
        _results['fail'] += 1
        return False

    return True


# ================================================================
# 統計分析
# ================================================================

def compute_stats(day_records: list) -> dict:
    """計算模擬整體統計數據。"""
    total_markov     = 0
    total_deliberate = 0
    total_conv       = 0
    c_values         = []
    ltm_by_day       = {}
    emotion_changes  = {}

    for day_data in day_records:
        day_n = day_data['day']
        ltm_snapshot = {}

        for slot_data in day_data['slots']:
            for code, ad in slot_data['agents'].items():
                mode = ad['mode']
                if mode == 'markov':
                    total_markov += 1
                elif mode == 'deliberate':
                    total_deliberate += 1

                if ad['verb'] == '對話':
                    total_conv += 1

                c_values.append({
                    'day':  day_n,
                    'time': slot_data['time'],
                    'code': code,
                    'name': ad['name'],
                    'C':    ad['confusion_C'],
                    'U':    ad['confusion_U'],
                    'K':    ad['confusion_K'],
                    'S':    ad['confusion_S'],
                    'mode': mode,
                })
                ltm_snapshot[code] = ad['ltm_count']

        ltm_by_day[day_n] = ltm_snapshot

        for code, sd in day_data['sleep'].items():
            if code not in emotion_changes:
                emotion_changes[code] = []
            emotion_changes[code].append({
                'day':    day_n,
                'before': sd['emotion_before'],
                'after':  sd['emotion_after'],
            })

    total_acts   = total_markov + total_deliberate
    deliberate_r = total_deliberate / total_acts * 100 if total_acts > 0 else 0

    return {
        'total_markov':     total_markov,
        'total_deliberate': total_deliberate,
        'deliberate_rate':  round(deliberate_r, 1),
        'total_conv':       total_conv,
        'c_values':         c_values,
        'ltm_by_day':       ltm_by_day,
        'emotion_changes':  emotion_changes,
    }


# ================================================================
# 詳細 HTML 報告生成
# ================================================================

def generate_report(day_records: list, stats: dict,
                    test_results: dict,
                    output_path: str = 'simulate_report.html'):
    """生成詳細視覺化 HTML 報告。"""
    from config.world_config import CHARACTER_NAMES as CN

    pass_n = test_results['pass']
    fail_n = test_results['fail']
    total  = pass_n + fail_n
    rate   = f"{pass_n/total*100:.1f}%" if total > 0 else "N/A"

    all_codes  = sorted(CN.keys())
    mode_color = {'markov': '#3b82f6', 'deliberate': '#f59e0b'}

    # ── CSS ───────────────────────────────────────────────────────
    CSS = '''
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:'Segoe UI',sans-serif; background:#0a0f1e; color:#e2e8f0;
       line-height:1.5; padding:24px; font-size:14px; }
h1 { color:#f8fafc; font-size:1.8rem; margin-bottom:4px; }
h2 { color:#94a3b8; font-size:.95rem; font-weight:normal; margin-bottom:24px; }
.summary-bar { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:28px; }
.sc { background:#1e293b; border-radius:12px; padding:14px 20px;
      flex:1; min-width:120px; text-align:center; border:1px solid #334155; }
.sc .val { font-size:1.9rem; font-weight:700; }
.sc .lbl { font-size:.75rem; color:#94a3b8; margin-top:3px; }
.pass { color:#22c55e; } .fail { color:#ef4444; }
.rate { color:#f59e0b; } .blue { color:#38bdf8; } .amber { color:#f59e0b; }
.green { color:#4ade80; } .pink { color:#f472b6; }
.day-block { margin-bottom:32px; }
.day-title { background:linear-gradient(135deg,#1e293b,#334155);
             border-radius:10px 10px 0 0; padding:12px 20px;
             display:flex; align-items:center; gap:12px; border:1px solid #475569; }
.day-num { font-size:1.2rem; font-weight:700; color:#f8fafc; }
.day-sub { color:#94a3b8; font-size:.88rem; }
.slots-container { border:1px solid #334155; border-top:none;
                   border-radius:0 0 10px 10px; overflow:hidden; }
.slot-section { border-bottom:1px solid #1e293b; }
.slot-header { background:#1a2332; padding:10px 20px;
               display:flex; align-items:center; gap:14px; }
.slot-time { font-size:1rem; font-weight:700; color:#38bdf8;
             background:#0f172a; padding:2px 10px; border-radius:5px; }
.slot-desc { color:#94a3b8; font-size:.85rem; }
.slot-clock { color:#475569; font-size:.78rem; margin-left:auto; }
.agents-grid { display:grid;
               grid-template-columns:repeat(auto-fill,minmax(290px,1fr));
               gap:10px; padding:14px; background:#0f172a; }
.agent-card { background:#1e293b; border-radius:8px; padding:13px;
              border:1px solid #334155; }
.agent-header { display:flex; justify-content:space-between;
                align-items:center; margin-bottom:7px; }
.agent-name { font-weight:700; color:#f1f5f9; }
.mode-badge { font-size:.68rem; padding:2px 8px; border-radius:999px;
              color:#fff; white-space:nowrap; }
.action-display { font-size:.92rem; background:#0f172a; padding:6px 10px;
                  border-radius:5px; margin-bottom:7px;
                  border-left:3px solid var(--mode-color,#3b82f6); }
.confusion-row { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:7px; }
.ctag { font-size:.72rem; padding:2px 7px; border-radius:4px;
        background:#0f172a; color:#94a3b8; }
.ctag.C-high { color:#f97316; background:#2d1a09; }
.metrics { display:flex; gap:7px; flex-wrap:wrap; margin-bottom:7px; }
.metric { font-size:.72rem; color:#94a3b8; background:#0f172a;
          padding:2px 7px; border-radius:4px; }
.metric.loc { color:#38bdf8; }
.metric.emo { color:#f472b6; }
.prob-chart { margin-top:6px; }
.prob-row { display:flex; align-items:center; gap:5px; margin-bottom:2px; }
.prob-label { font-size:.68rem; color:#94a3b8; width:48px; text-align:right;
              overflow:hidden; white-space:nowrap; }
.prob-bg { flex:1; background:#0f172a; border-radius:3px; height:8px; }
.prob-fill { background:#3b82f6; border-radius:3px; height:8px; }
.prob-val { font-size:.67rem; color:#64748b; width:36px; }
.thought-btn { background:#0f172a; color:#64748b; border:1px solid #1e293b;
               border-radius:4px; font-size:.72rem; padding:3px 9px; cursor:pointer;
               margin-top:6px; width:100%; text-align:left; transition:background .15s; }
.thought-btn:hover { background:#1e293b; color:#94a3b8; }
.thought-box { display:none; margin-top:5px; font-size:.78rem; color:#94a3b8;
               background:#0f172a; padding:6px 9px; border-radius:5px;
               border-left:2px solid #f59e0b; font-style:italic; }
.ham-tag { display:inline-block; background:#1a2814; color:#86efac;
           font-size:.68rem; padding:1px 6px; border-radius:3px; margin-top:4px; }
.sleep-grid { display:grid;
              grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
              gap:10px; padding:14px; background:#060d1a; }
.sleep-card { background:#1e293b; border-radius:8px; padding:12px;
              border:1px solid #1e3a28; }
.sleep-name { font-weight:700; color:#86efac; margin-bottom:6px; }
.sleep-row { font-size:.78rem; color:#94a3b8; margin-bottom:3px; }
.sleep-summary { font-size:.75rem; color:#64748b; margin-top:6px;
                 font-style:italic; border-left:2px solid #1e4d2e;
                 padding-left:6px; }
.section-title { color:#f8fafc; font-size:1.1rem; font-weight:700;
                 margin:28px 0 12px; border-left:4px solid #3b82f6;
                 padding-left:12px; }
.anomaly-box { background:#2d1a09; border:1px solid #7c2d12;
               border-radius:8px; padding:12px 16px; margin:10px 0;
               font-size:.83rem; color:#fdba74; }
.anomaly-ok  { background:#0a1f14; border:1px solid #1e4d2e;
               border-radius:8px; padding:10px 14px; margin:10px 0;
               font-size:.83rem; color:#4ade80; }
table { width:100%; border-collapse:collapse; margin:12px 0; }
th { background:#1e293b; color:#94a3b8; font-size:.78rem; padding:9px 12px;
     text-align:left; border-bottom:1px solid #334155; }
td { padding:8px 12px; border-bottom:1px solid #1e293b; font-size:.83rem; }
tr:hover td { background:#1e293b; }
.stat-grid { display:grid;
             grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
             gap:10px; margin:12px 0; }
.stat-card { background:#1e293b; border-radius:8px; padding:12px 16px;
             border:1px solid #334155; }
.stat-title { font-size:.78rem; color:#94a3b8; margin-bottom:4px; }
.stat-val   { font-size:1.3rem; font-weight:700; color:#f1f5f9; }
.bar-wrap { margin:6px 0; }
.bar-label { font-size:.72rem; color:#94a3b8; margin-bottom:2px; }
.bar-bg { background:#0f172a; border-radius:4px; height:12px; }
.bar-fill-m { background:#3b82f6; border-radius:4px; height:12px; }
.bar-fill-d { background:#f59e0b; border-radius:4px; height:12px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:20px; }
.li { display:flex; align-items:center; gap:6px; font-size:.83rem; }
.ld { width:13px; height:13px; border-radius:3px; }
footer { margin-top:40px; color:#334155; font-size:.75rem; text-align:center; }
/* ── Day Tabs ── */
.tab-bar { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:18px; }
.tab-btn { background:#1e293b; border:1px solid #334155; color:#94a3b8;
           padding:7px 18px; border-radius:8px; cursor:pointer;
           font-size:.85rem; font-weight:600; transition:all .15s; }
.tab-btn:hover { background:#334155; color:#f1f5f9; }
.tab-btn.active { background:#3b82f6; border-color:#3b82f6; color:#fff; }
/* ── Conversation Pairs ── */
.conv-section { background:#060d1a; border-top:1px solid #1e293b; padding:12px 16px; }
.conv-header { font-size:.85rem; color:#64748b; margin-bottom:8px;
               display:flex; align-items:center; gap:8px; cursor:pointer;
               user-select:none; padding:8px 10px; border-radius:6px;
               border:1px solid #1e293b; transition:background .15s; }
.conv-header:hover { background:#0f172a; color:#94a3b8; }
.conv-toggle { transition:transform .2s; display:inline-block; font-size:.8rem; }
.conv-toggle.open { transform:rotate(90deg); }
.conv-list { display:none; margin-top:8px; }
.conv-list.open { display:block; }
/* 每一組對話配對 */
.conv-pair { border-radius:8px; margin-bottom:10px;
             border:1px solid #1e293b; overflow:hidden; }
.conv-pair:last-child { margin-bottom:0; }
.conv-pair-title { background:#0d1520; padding:6px 12px;
                   font-size:.75rem; color:#475569; border-bottom:1px solid #1e293b; }
/* 單輪（發話/回話） */
.conv-turn { padding:10px 14px; }
.conv-turn.sender-turn   { background:#0d1a2d; border-left:3px solid #3b82f6; }
.conv-turn.receiver-turn { background:#1a130a; border-left:3px solid #f59e0b; }
.conv-turn-header { display:flex; align-items:center; gap:7px; margin-bottom:6px; }
.conv-who { font-size:.75rem; font-weight:700; min-width:52px;
            padding:2px 8px; border-radius:4px; white-space:nowrap; }
.conv-who.sender   { background:#1d3a5f; color:#60a5fa; }
.conv-who.receiver { background:#2d1a09; color:#fbbf24; }
.conv-conf { font-size:.68rem; color:#475569; margin-left:auto; }
.conv-msg { font-size:.84rem; color:#cbd5e1; line-height:1.55;
            padding:4px 8px; background:rgba(255,255,255,0.03);
            border-radius:4px; margin-bottom:4px; }
.conv-thought { font-size:.76rem; color:#64748b; font-style:italic;
                padding:4px 8px; border-radius:4px;
                background:rgba(255,255,255,0.02); }
.conv-divider { text-align:center; color:#334155; font-size:.8rem;
                padding:4px 0; background:#0a0f1e; }
'''

    # ── 工具：機率條 ─────────────────────────────────────────────
    def prob_bars(probs):
        if not probs:
            return ''
        items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        rows  = []
        for a, p in items:
            pct = int(p * 100)
            rows.append(
                f'<div class="prob-row">'
                f'<span class="prob-label">{a}</span>'
                f'<div class="prob-bg"><div class="prob-fill" style="width:{pct}%"></div></div>'
                f'<span class="prob-val">{p:.3f}</span>'
                f'</div>'
            )
        return f'<div class="prob-chart">{"".join(rows)}</div>'

    # ── 工具：代理人卡片 ─────────────────────────────────────────
    def agent_card(code, ad, uid=''):
        mode  = ad['mode']
        color = mode_color.get(mode, '#6b7280')
        label = '簡單思考 Markov' if mode == 'markov' else '複雜思考 LTM'

        C = ad['confusion_C']
        c_cls = 'C-high' if C >= 0.5 else ''

        pb = prob_bars(ad['action_probs']) if mode == 'markov' else ''
        th = ''
        if ad.get('thought'):
            tid = f'th-{uid or code}'
            th = (f'<button class="thought-btn" id="btn-{tid}" '
                  f'onclick="toggleThought(\'{tid}\')">💭 顯示想法</button>'
                  f'<div class="thought-box" id="{tid}">{ad["thought"]}</div>')

        ham_tag = ''
        if ad['ham_count'] > 0:
            ham_tag = f'<span class="ham-tag">HAM ×{ad["ham_count"]}</span>'

        return f'''
<div class="agent-card" style="--mode-color:{color}; border-top:3px solid {color}">
  <div class="agent-header">
    <span class="agent-name">{ad["name"]} ({code})</span>
    <span class="mode-badge" style="background:{color}">{label}</span>
  </div>
  <div class="action-display">→ <strong>{ad["action"]}</strong>{ham_tag}</div>
  <div class="confusion-row">
    <span class="ctag {c_cls}">C={C:.3f}</span>
    <span class="ctag">U={ad["confusion_U"]:.2f}</span>
    <span class="ctag">K={ad["confusion_K"]:.2f}</span>
    <span class="ctag">S={ad["confusion_S"]:.2f}</span>
  </div>
  <div class="metrics">
    <span class="metric">STM={ad["stm_count"]}</span>
    <span class="metric">LTM={ad["ltm_count"]}</span>
    <span class="metric loc">📍{ad["location"]}</span>
    <span class="metric emo">💫{ad["emotion"]}</span>
  </div>
  {pb}{th}
</div>'''

    # ── 工具：睡眠卡片 ────────────────────────────────────────────
    def sleep_card(code, sd):
        delta  = sd['ltm_after'] - sd['ltm_before']
        e_same = sd['emotion_before'] == sd['emotion_after']
        emo_html = (
            f'{sd["emotion_before"]}'
            if e_same else
            f'{sd["emotion_before"]} → <strong style="color:#f472b6">{sd["emotion_after"]}</strong>'
        )
        summary_html = ''
        if sd.get('summary'):
            summary_html = f'<div class="sleep-summary">"{sd["summary"][:100]}{"..." if len(sd.get("summary","")) > 100 else ""}"</div>'
        return f'''
<div class="sleep-card">
  <div class="sleep-name">{sd["name"]} ({code})</div>
  <div class="sleep-row">STM: {sd["stm_before"]}→0（清空）</div>
  <div class="sleep-row">LTM: {sd["ltm_before"]}→{sd["ltm_after"]}（+{delta}）  剪枝:{sd["ltm_pruned"]}</div>
  <div class="sleep-row">情緒: {emo_html}</div>
  <div class="sleep-row">第 {sd["day_after"]} 天</div>
  {summary_html}
</div>'''

    # ── 工具：對話區段 HTML ───────────────────────────────────────
    def conv_section_html(conversations, slot_idx, day_n):
        if not conversations:
            return ''
        uid = f'd{day_n}s{slot_idx}'
        pairs_h = ''
        for pair_idx, pair in enumerate(conversations):
            sname = pair.get('sender_name', pair.get('sender_code', '?'))
            rname = pair.get('receiver_name', pair.get('receiver_code', '?'))
            msg         = pair.get('message', '')
            resp        = pair.get('response', '')
            s_thought   = pair.get('sender_thought', '')
            r_thought   = pair.get('receiver_thought', '')
            s_mode      = pair.get('sender_mode', '')
            r_mode      = pair.get('receiver_mode', '')

            def _mode_badge(m):
                if not m:
                    return ''
                col = mode_color.get(m, '#6b7280')
                lbl = 'Markov' if m == 'markov' else 'LTM'
                return (f'<span class="mode-badge" '
                        f'style="background:{col};font-size:.65rem;padding:1px 6px">'
                        f'{lbl}</span>')

            def _thought_html(t):
                if not t:
                    return ''
                return f'<div class="conv-thought">💭 {t}</div>'

            s_badge = _mode_badge(s_mode)
            r_badge = _mode_badge(r_mode)

            pairs_h += f'''
<div class="conv-pair">
  <div class="conv-pair-title">對話 #{pair_idx + 1}</div>
  <div class="conv-turn sender-turn">
    <div class="conv-turn-header">
      <span class="conv-who sender">{sname}</span>
      {s_badge}
      <span class="conv-conf">{s_mode}</span>
    </div>
    <div class="conv-msg">🗣 {msg or "（開啟對話）"}</div>
    {_thought_html(s_thought)}
  </div>
  <div class="conv-divider">↓</div>
  <div class="conv-turn receiver-turn">
    <div class="conv-turn-header">
      <span class="conv-who receiver">{rname}</span>
      {r_badge}
      <span class="conv-conf">{r_mode}</span>
    </div>
    <div class="conv-msg">🗣 {resp or "（無回應）"}</div>
    {_thought_html(r_thought)}
  </div>
</div>'''

        n = len(conversations)
        return f'''<div class="conv-section">
  <div class="conv-header" onclick="toggleConv('{uid}')">
    <span class="conv-toggle" id="arr-{uid}">▶</span>
    💬 本時段完整對話記錄（{n} 組）— 點擊展開 / 收合
  </div>
  <div class="conv-list" id="conv-{uid}">{pairs_h}</div>
</div>'''

    # ── 工具：日期區塊 ────────────────────────────────────────────
    def day_block(dd):
        day_n   = dd['day']
        slots_h = []
        for i, slot in enumerate(dd['slots']):
            agents_h = ''.join(agent_card(c, slot['agents'][c], f"d{day_n}s{i}c{c}")
                               for c in all_codes if c in slot['agents'])
            conv_h = conv_section_html(slot.get('conversations', []), i, day_n)
            slots_h.append(f'''
<div class="slot-section">
  <div class="slot-header">
    <span class="slot-time">🕐 {slot["time"]}</span>
    <span class="slot-desc">{slot["desc"]}</span>
    <span class="slot-clock">{slot["clock"]}</span>
  </div>
  <div class="agents-grid">{agents_h}</div>
  {conv_h}
</div>''')

        sleep_h = ''.join(sleep_card(c, dd['sleep'][c])
                          for c in sorted(dd['sleep'].keys()))

        anom_h = ''
        if dd['anomalies']:
            anom_items = ''.join(f'<div>⚠️ {a}</div>' for a in dd['anomalies'])
            anom_h = f'<div class="anomaly-box">{anom_items}</div>'
        else:
            anom_h = '<div class="anomaly-ok">✓ 無異常偵測</div>'

        return f'''
<div class="day-block">
  <div class="day-title">
    <span class="day-num">Day {day_n}</span>
    <span class="day-sub">共 {len(dd["slots"])} 個時間段  ·  {len(dd["sleep"])} 位角色入睡</span>
  </div>
  <div class="slots-container">
    {"".join(slots_h)}
    <div class="slot-section">
      <div class="slot-header"><span class="slot-time">🌙 睡眠濃縮</span>
        <span class="slot-desc">STM → LTM 記憶鞏固</span></div>
      <div class="sleep-grid">{sleep_h}</div>
    </div>
  </div>
  {anom_h}
</div>'''

    # ── 統計區塊 ──────────────────────────────────────────────────
    def stats_block():
        s  = stats
        tm = s['total_markov']
        td = s['total_deliberate']
        tt = tm + td
        dr = s['deliberate_rate']
        # Mode 分布條
        m_pct = int(tm / tt * 100) if tt > 0 else 50
        d_pct = 100 - m_pct
        mode_bar = f'''
<div class="bar-wrap">
  <div class="bar-label">簡單思考 {tm} 次 ({m_pct}%) vs 複雜思考 {td} 次 ({d_pct}%)</div>
  <div style="display:flex;height:14px;border-radius:4px;overflow:hidden">
    <div style="width:{m_pct}%;background:#3b82f6"></div>
    <div style="width:{d_pct}%;background:#f59e0b"></div>
  </div>
</div>'''

        # LTM 成長表格
        ltm_rows = ''
        for day_n, snap in sorted(s['ltm_by_day'].items()):
            cells = ''.join(f'<td>{snap.get(c,0)}</td>' for c in all_codes)
            total_ltm = sum(snap.values())
            ltm_rows += f'<tr><td>Day {day_n}</td>{cells}<td><strong>{total_ltm}</strong></td></tr>'

        ltm_table = f'''
<table>
  <tr><th>天數</th>{"".join(f"<th>{CN.get(c,c)}</th>" for c in all_codes)}<th>合計</th></tr>
  {ltm_rows}
</table>'''

        # C 值趨勢表格（前 20 筆）
        cv_rows = ''
        for cv in s['c_values'][:24]:
            badge_col = mode_color.get(cv['mode'], '#6b7280')
            cv_rows += (
                f'<tr>'
                f'<td>Day{cv["day"]} {cv["time"]}</td>'
                f'<td>{cv["name"]}</td>'
                f'<td style="color:{"#f97316" if cv["C"]>=0.5 else "#94a3b8"}">{cv["C"]:.3f}</td>'
                f'<td>{cv["U"]:.2f}</td>'
                f'<td>{cv["K"]:.2f}</td>'
                f'<td>{cv["S"]:.2f}</td>'
                f'<td><span class="mode-badge" style="background:{badge_col}">{cv["mode"]}</span></td>'
                f'</tr>'
            )

        c_table = f'''
<table>
  <tr><th>時段</th><th>角色</th><th>C</th><th>U</th><th>K</th><th>S</th><th>路徑</th></tr>
  {cv_rows}
</table>'''

        # 情緒變化表格
        emo_rows = ''
        for code in all_codes:
            changes = s['emotion_changes'].get(code, [])
            emo_cells = ''.join(
                f'<td>Day{c["day"]}: {c["before"]}→{c["after"]}</td>'
                for c in changes
            )
            if changes:
                emo_rows += f'<tr><td>{CN.get(code,code)}</td>{emo_cells}</tr>'

        emo_table = f'''
<table>
  <tr><th>角色</th><th colspan="4">天數別情緒變化</th></tr>
  {emo_rows}
</table>'''

        return f'''
<div class="section-title">📊 統計分析</div>
<div class="stat-grid">
  <div class="stat-card"><div class="stat-title">總行動次數</div><div class="stat-val blue">{tt}</div></div>
  <div class="stat-card"><div class="stat-title">複雜思考率</div><div class="stat-val amber">{dr}%</div></div>
  <div class="stat-card"><div class="stat-title">對話行動次數</div><div class="stat-val pink">{s["total_conv"]}</div></div>
  <div class="stat-card"><div class="stat-title">模擬天數</div><div class="stat-val green">{len(day_records)}</div></div>
</div>
{mode_bar}
<div class="section-title">📈 LTM 成長（各天末）</div>
{ltm_table}
<div class="section-title">🧠 困惑指數趨勢（前 24 筆）</div>
{c_table}
<div class="section-title">💫 情緒變化</div>
{emo_table}'''

    # ── 彙整 HTML ────────────────────────────────────────────────
    n_days = len(day_records)
    all_anomalies = [a for dd in day_records for a in dd.get('anomalies', [])]
    global_anom_html = ''
    if all_anomalies:
        items = ''.join(f'<div>⚠️ {a}</div>' for a in all_anomalies)
        global_anom_html = (
            f'<div class="section-title">⚠️ 全域異常報告</div>'
            f'<div class="anomaly-box">{items}</div>'
        )
    else:
        global_anom_html = (
            '<div class="section-title">✅ 異常偵測</div>'
            f'<div class="anomaly-ok">{n_days} 天模擬期間未偵測到任何邏輯或數值異常</div>'
        )

    # Day tab bar
    tab_btns = ''.join(
        f'<button class="tab-btn{"  active" if i == 0 else ""}" '
        f'onclick="switchDay({dd["day"]})">Day {dd["day"]}</button>'
        for i, dd in enumerate(day_records)
    )
    tab_bar_html = f'<div class="tab-bar">{tab_btns}</div>'

    # Day content divs (only first visible)
    days_html = ''
    for i, dd in enumerate(day_records):
        vis = '' if i == 0 else ' style="display:none"'
        days_html += f'<div id="day-content-{dd["day"]}"{vis}>{day_block(dd)}</div>'

    JS = '''
function switchDay(n) {
  document.querySelectorAll('[id^="day-content-"]').forEach(function(el) {
    el.style.display = 'none';
  });
  document.querySelectorAll('.tab-btn').forEach(function(btn) {
    btn.classList.remove('active');
  });
  var target = document.getElementById('day-content-' + n);
  if (target) target.style.display = '';
  var btns = document.querySelectorAll('.tab-btn');
  if (btns[n-1]) btns[n-1].classList.add('active');
}
function toggleConv(uid) {
  var list = document.getElementById('conv-' + uid);
  var arr  = document.getElementById('arr-' + uid);
  if (!list) return;
  var open = list.classList.toggle('open');
  if (arr) arr.classList.toggle('open', open);
}
function toggleThought(uid) {
  var box = document.getElementById(uid);
  var btn = document.getElementById('btn-' + uid);
  if (!box) return;
  var show = box.style.display !== 'block';
  box.style.display = show ? 'block' : 'none';
  if (btn) btn.textContent = show ? '💭 收合想法' : '💭 顯示想法';
}
'''

    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGI {n_days} 天模擬報告</title>
<style>{CSS}</style>
</head>
<body>
<h1>AGI 自主生活模擬 — {n_days} 天完整報告</h1>
<h2>兩段式時間軸 × 馬可夫鏈（簡單思考）× LTM 自主推理（複雜思考）× 記憶濃縮</h2>

<div class="summary-bar">
  <div class="sc"><div class="val pass">{pass_n}</div><div class="lbl">驗證通過</div></div>
  <div class="sc"><div class="val fail">{fail_n}</div><div class="lbl">驗證失敗</div></div>
  <div class="sc"><div class="val rate">{rate}</div><div class="lbl">通過率</div></div>
  <div class="sc"><div class="val blue">{n_days}</div><div class="lbl">模擬天數</div></div>
  <div class="sc"><div class="val amber">{stats["total_deliberate"]}</div><div class="lbl">複雜思考次數</div></div>
  <div class="sc"><div class="val pink">{stats["total_conv"]}</div><div class="lbl">對話行動次數</div></div>
</div>

<div class="legend">
  <div class="li"><div class="ld" style="background:#3b82f6"></div><span>簡單思考（馬可夫鏈）C &lt; threshold</span></div>
  <div class="li"><div class="ld" style="background:#f59e0b"></div><span>複雜思考（LTM 自主推理）C ≥ threshold</span></div>
  <div class="li"><div class="ld" style="background:#f97316"></div><span>高困惑指數（C ≥ 0.5）</span></div>
</div>

{tab_bar_html}
{days_html}

{stats_block()}
{global_anom_html}

<footer>AGI 自主生活模擬系統 — simulate.py {n_days} 天模擬報告</footer>
<script>{JS}</script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ================================================================
# 自主模擬（真實模型模式）
# ================================================================

def run_autonomous_simulation(manager, max_days: int = 3):
    """
    自主模式：角色完全自主決定行動與睡眠時間（06:00 起，最晚凌晨04:00）。
    使用 AgentManager.run_autonomous_day() 驅動每一天。
    """
    day_records = []

    for day_n in range(1, max_days + 1):
        section(f'Day {day_n} 自主模擬')

        day_data = manager.run_autonomous_day(day_n)

        # 印出每個 slot 摘要
        for slot in day_data['slots']:
            subsection(f'{slot["time"]} — {slot["desc"]}')
            print(f'    時鐘：{slot["clock"]}')
            for code, ad in slot['agents'].items():
                mode_icon = '🔵' if ad['mode'] == 'markov' else '🟡'
                print(f'    {mode_icon} [{code}:{ad["name"]}] {ad["mode"].upper():12s} '
                      f'C={ad["confusion_C"]:.3f}  → {ad["action"]}')
                if ad.get('thought') and ad['mode'] == 'deliberate':
                    print(f'       想法: {ad["thought"][:70]}{"..." if len(ad["thought"]) > 70 else ""}')

        # 印出睡眠濃縮
        subsection(f'Day {day_n} — 睡眠濃縮')
        for code, sd in sorted(day_data['sleep'].items()):
            print(f'    [{code}:{sd["name"]}] STM:{sd["stm_before"]}→0  '
                  f'LTM:{sd["ltm_before"]}→{sd["ltm_after"]}(+{sd["ltm_after"]-sd["ltm_before"]})  '
                  f'情緒:{sd["emotion_before"]}→{sd["emotion_after"]}  '
                  f'剪枝:{sd["ltm_pruned"]}  Day→{sd["day_after"]}')
            if sd.get('summary'):
                print(f'       摘要: {sd["summary"][:80]}')

        # 異常偵測
        issues = detect_anomalies(manager)
        if issues:
            print(f'\n  ⚠️  Day {day_n} 發現異常：')
            for iss in issues:
                print(f'    • {iss}')
        else:
            print(f'  ✓ Day {day_n} 無異常')

        day_data['anomalies'] = issues
        day_records.append(day_data)
        _results['pass'] += sum(len(s['agents']) for s in day_data['slots'])

    return day_records


# ================================================================
# CLI 進入點
# ================================================================

def main(use_real_model: bool = True, num_slots: int = 4, max_days: int = None):
    from utils.test_reset import reset_all
    from model.model_loader import ModelLoader
    from world.world_clock import WorldClock
    from agent.agent_manager import AgentManager

    # 重置
    section('0. 重置測試環境')
    reset_all()
    print('  角色 JSON 已重置')

    # 初始化
    section('1. 系統初始化')
    if use_real_model:
        loader = ModelLoader()
        loader.load()
        print('  Phi 模型載入完成')
    else:
        loader = SmartMockLoader()
        print('  SmartMockLoader 就緒')

    clock   = WorldClock()
    manager = AgentManager(loader=loader, clock=clock)
    print(f'  角色：{manager.all_codes()}  時鐘：{clock.scene_prefix()}')

    # 多天模擬
    if use_real_model:
        n_days = max_days if max_days is not None else 3
        day_records = run_autonomous_simulation(manager, max_days=n_days)
    else:
        day_records = run_four_day_simulation(loader, manager, clock, max_days=max_days)

    # 統計分析
    stats = compute_stats(day_records)

    # 最終異常總報告
    section('最終異常偵測（全域）')
    all_issues = detect_anomalies(manager)
    if all_issues:
        for iss in all_issues:
            print(f'  ⚠️  {iss}')
        _results['fail'] += len(all_issues)
    else:
        print('  ✓ 無異常')
        _results['pass'] += 1

    # 統計輸出
    section('模擬結果彙整')
    total = _results['pass'] + _results['fail']
    rate  = _results['pass'] / total * 100 if total > 0 else 0
    print(f'  PASS: {_results["pass"]}  FAIL: {_results["fail"]}  通過率: {rate:.1f}%')
    print(f'  簡單思考: {stats["total_markov"]} 次  複雜思考: {stats["total_deliberate"]} 次'
          f'  ({stats["deliberate_rate"]}%)')
    print(f'  對話行動: {stats["total_conv"]} 次')

    # 生成 HTML 報告
    generate_report(day_records, stats, _results, 'simulate_report.html')
    print('\n  HTML 報告已生成：simulate_report.html')

    return _results['fail'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AGI 多天完整模擬')
    parser.add_argument('--no-model',    action='store_true')
    parser.add_argument('--check-only',  action='store_true')
    parser.add_argument('--slots',       type=int, default=4)
    parser.add_argument('--rounds',      type=int, default=None)
    parser.add_argument('--days',        type=int, default=None,
                        help='最多跑幾天（預設全部 10 天）')
    args = parser.parse_args()

    if args.rounds:
        args.slots = args.rounds

    if args.check_only:
        ok = validate_model()
        sys.exit(0 if ok else 1)
    elif args.no_model:
        success = main(use_real_model=False, num_slots=args.slots,
                       max_days=args.days)
    else:
        ok = validate_model()
        if not ok:
            sys.exit(1)
        success = main(use_real_model=True, num_slots=args.slots,
                       max_days=args.days)

    sys.exit(0 if success else 1)
