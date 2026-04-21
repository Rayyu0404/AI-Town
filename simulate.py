# ================================================================
# simulate.py
# 完整系統模擬 + 模型下載驗證 + 輸出驗證
#
# 執行方式：
#   python simulate.py              → 下載 / 驗證模型，執行完整模擬
#   python simulate.py --no-model   → 使用假模型（快速驗證流程，不需 GPU）
#   python simulate.py --rounds N   → 指定模擬輪數（預設 2）
#   python simulate.py --check-only → 只驗證模型，不執行模擬
#
# 功能：
#   0. 重置角色 JSON 狀態，確保測試環境乾淨
#   1. 下載並驗證 Phi-3.5-Vision 模型（checksum / 推論健全性）
#   2. 初始化全部 5 個角色 + WorldClock + AgentManager
#   3. 模擬多輪推論，涵蓋：單角色、多角色、對話傳遞、圖片輸入
#   4. 逐一驗證每個輸出的格式與邏輯合法性
#   5. 手動觸發睡眠濃縮，驗證 STM→LTM 流程
#   6. 顯示最終記憶狀態與衰減結果
#   7. 輸出 PASS / FAIL 統計
# ================================================================

import argparse
import sys
import io
import json
import os
import time
import numpy as np
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, '.')

from config.world_config import CHARACTER_NAMES
from config.action_list import VALID_ACTIONS
from config.model_config import MODEL_ID


# ================================================================
# 輸出工具
# ================================================================

_results = {'pass': 0, 'fail': 0}


def _check(label: str, cond: bool, detail: str = '') -> bool:
    symbol = 'PASS' if cond else 'FAIL'
    suffix = f'  ({detail})' if detail else ''
    print(f'    [{symbol}] {label}{suffix}')
    _results['pass' if cond else 'fail'] += 1
    return cond


def section(title: str):
    print(f'\n{"="*62}')
    print(f'  {title}')
    print(f'{"="*62}')


def subsection(title: str):
    print(f'\n  ── {title} ──')


# ================================================================
# 假模型（--no-model 模式）
# ================================================================

class _MockFusion:
    def fuse_inputs(self, text, image_inputs, **kw):
        return {}

    def generate(self, fused_inputs, gen_cfg=None):
        return (
            '[ACTION]: 工作\n'
            '[THOUGHT]: 今天繼續在咖啡廳工作，感覺還不錯。\n'
            '[HAM]: [{"subject":"Amy","relation":"工作","object":"咖啡廳",'
            '"location":"咖啡廳","time":"早上"}]\n'
            '[/HAM]'
        )


class _MockText:
    @staticmethod
    def build_prompt(text, num_images=0, system_text=None):
        return type('P', (), {'prompt': text})()


class _MockVision:
    @staticmethod
    def encode(images):
        return type('V', (), {'to_dict': lambda self: {}})()


class MockLoader:
    """不依賴 GPU 的假模型，用固定字串回應，專門驗證流程正確性。"""

    fusion = _MockFusion()
    text   = _MockText()
    vision = _MockVision()

    def load(self):
        pass

    def is_loaded(self) -> bool:
        return True

    def make_model_fn(self, max_new_tokens=256, temperature=0.0):
        def model_fn(prompt: str) -> str:
            if any(kw in prompt for kw in ['值得長期', '選出值得', '哪些值得']):
                return '[{"subject":"Amy","relation":"工作","object":"咖啡廳"}]'
            if any(kw in prompt for kw in ['總結', '摘要', '1-2句']):
                return 'Amy每天在咖啡廳工作，與同事維持良好關係。'
            if '關係摘要' in prompt or '兩人關係' in prompt:
                return '兩人關係友好，偶有互動。'
            if '情緒' in prompt:
                return '平靜'
            return '平靜'
        return model_fn


# ================================================================
# 驗證函式
# ================================================================

def validate_step_result(result: dict, char_name: str) -> bool:
    ok = True
    ok &= _check('action 不為空',       bool(result.get('action')))
    ok &= _check('verb 在合法清單',       result.get('verb') in VALID_ACTIONS,
                 f'verb={result.get("verb")}')
    ok &= _check('thought 存在',          'thought' in result)
    ok &= _check('ham 為 list',           isinstance(result.get('ham'), list))
    ok &= _check('mode 合法',             result.get('mode') in ('intuitive', 'deliberate'),
                 f'mode={result.get("mode")}')
    conf = result.get('confusion', {})
    ok &= _check('confusion 包含 C/U/K/S', all(k in conf for k in ('C','U','K','S')))
    ok &= _check('C 值 0~1',              0 <= conf.get('C', -1) <= 1,
                 f'C={conf.get("C")}')
    ok &= _check('should_sleep 為 bool',   isinstance(result.get('should_sleep'), bool))
    return ok


def validate_sleep_result(result: dict) -> bool:
    ok = True
    ok &= _check('new_propositions 為 int', isinstance(result.get('new_propositions'), int))
    ok &= _check('ltm_total 為 int',         isinstance(result.get('ltm_total'), int))
    ok &= _check('ltm_pruned 為 int',        isinstance(result.get('ltm_pruned'), int))
    ok &= _check('summary 為 str',           isinstance(result.get('summary'), str))
    return ok


def validate_ltm_props(ltm) -> bool:
    ok = True
    props = ltm.get_all()
    ids = [p['id'] for p in props]
    ok &= _check('LTM ID 唯一性',          len(ids) == len(set(ids)),
                 f'ids={ids}')
    for p in props:
        ok &= _check(f'{p["id"]} strength 在 0~1',
                     0.0 <= p.get('strength', -1) <= 1.0,
                     f'strength={p.get("strength")}')
    return ok


# ================================================================
# 模擬事件定義
# ================================================================

DEMO_EVENTS = [
    {
        'code': 'A', 'round_offset': 0,
        'scene': '咖啡廳，早晨，陽光從窗戶灑入',
        'input_text': 'Ben走進咖啡廳，對Amy微笑說：早安！',
        'target': 'B',
        'image': None,
        'desc': 'A: 日常開店，Ben來訪',
    },
    {
        'code': 'B', 'round_offset': 0,
        'scene': '超市，上午，貨架整理中',
        'input_text': '',
        'target': None,
        'image': None,
        'desc': 'B: 獨自在超市工作',
    },
    {
        'code': 'C', 'round_offset': 0,
        'scene': '辦公室，上午，電腦螢幕亮著報表',
        'input_text': 'David走進來說：Claire，今天的報告要提前交。',
        'target': 'D',
        'image': None,
        'desc': 'C: David交代任務',
    },
    {
        'code': 'D', 'round_offset': 0,
        'scene': '辦公室，上午，窗外城市風景',
        'input_text': '',
        'target': None,
        'image': None,
        'desc': 'D: 獨自在辦公室',
    },
    {
        'code': 'E', 'round_offset': 0,
        'scene': '餐廳，午餐時間，忙碌的吧台',
        'input_text': '一位常客點頭打招呼。',
        'target': None,
        'image': None,
        'desc': 'E: 午餐時段',
    },
    # 帶圖片的場景（用 numpy 合成純色假圖）
    {
        'code': 'A', 'round_offset': 1,
        'scene': '咖啡廳，午後',
        'input_text': 'David點了一杯拿鐵，看起來很疲憊。',
        'target': 'D',
        'image': 'synthetic',   # 由執行時替換為 PIL Image
        'desc': 'A: 帶圖片推論（合成圖）',
    },
    {
        'code': 'B', 'round_offset': 1,
        'scene': '超市附近，下午',
        'input_text': '',
        'target': None,
        'image': None,
        'desc': 'B: 下午收工前',
    },
]


# ================================================================
# Step 0：重置測試環境
# ================================================================

def reset_test_data():
    """將角色 JSON 重置為乾淨初始狀態，避免前次執行污染測試結果。"""
    from utils.test_reset import reset_all
    reset_all()
    print('  角色 JSON 已重置為乾淨初始狀態')


# ================================================================
# Step 1：模型下載與驗證
# ================================================================

def validate_model() -> bool:
    """
    下載（若未快取）並驗證 Phi-3.5-Vision 模型。
    驗證項目：
      - HuggingFace 模型 ID 可存取
      - processor 可正常載入
      - model 可正常載入（不做推論，只驗結構）
      - 基礎推論健全性（傳入最短 prompt，模型能回傳非空字串）
    """
    section('1. 模型下載與驗證')

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as e:
        print(f'  [ERROR] 缺少依賴：{e}')
        print('  請執行：pip install torch transformers')
        _results['fail'] += 1
        return False

    # ── 1-1 HF cache 路徑確認 ─────────────────────────────────────
    cache_dir = os.path.join(
        os.path.expanduser('~'), '.cache', 'huggingface', 'hub'
    )
    print(f'  HuggingFace cache: {cache_dir}')
    _check('HF cache 目錄存在', os.path.isdir(cache_dir),
           '首次執行時會自動建立')

    # ── 1-2 Processor 載入（含下載）─────────────────────────────
    print(f'\n  載入 processor：{MODEL_ID}')
    print('  （首次執行會從 HuggingFace 下載約 1GB，請耐心等待）')
    try:
        from config.model_config import MODEL_NUM_CROPS
        print('  [1/4] 初始化 processor...', flush=True)
        _t0 = time.time()
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            num_crops=MODEL_NUM_CROPS,
        )
        print(f'  [1/4] processor 載入完成（{time.time()-_t0:.1f}s）', flush=True)
        _check('Processor 載入成功', True)
        _check('Processor 有 tokenizer',
               hasattr(processor, 'tokenizer'))
        _check('Processor 有 image_processor',
               hasattr(processor, 'image_processor'))
        tok_vocab = len(processor.tokenizer)
        _check('tokenizer vocab > 10000', tok_vocab > 10000,
               f'vocab_size={tok_vocab}')
    except Exception as e:
        print(f'  [ERROR] Processor 載入失敗：{e}')
        _results['fail'] += 1
        return False

    # ── 1-3 Model 載入（含下載）──────────────────────────────────
    print(f'\n  載入 model：{MODEL_ID}')
    print('  （首次執行需下載約 7GB，請耐心等待）')
    try:
        from model.model_loader import pick_device
        device = pick_device()
        print(f'  使用裝置：{device}')

        print('  [2/4] 載入模型權重（從快取讀取，約需 1~3 分鐘）...', flush=True)
        _t1 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype='auto',
            _attn_implementation='eager',
        )
        print(f'  [2/4] 模型載入完成（{time.time()-_t1:.1f}s）', flush=True)
        print('  [3/4] 移至裝置...', flush=True)
        _t2 = time.time()
        model = model.to(device)
        print(f'  [3/4] 移至裝置完成（{time.time()-_t2:.1f}s）', flush=True)
        _check('Model 載入成功', True)

        param_count = sum(p.numel() for p in model.parameters())
        _check('Model 參數量 > 1B', param_count > 1_000_000_000,
               f'{param_count/1e9:.2f}B 參數')

        # 驗證關鍵模組存在
        _check('Model 有 config',     hasattr(model, 'config'))
        _check('Model config 有 vocab_size',
               hasattr(model.config, 'vocab_size'))

    except Exception as e:
        print(f'  [ERROR] Model 載入失敗：{e}')
        _results['fail'] += 1
        return False

    # ── 1-4 基礎推論健全性測試 ───────────────────────────────────
    print('\n  [4/4] 執行基礎推論健全性測試...', flush=True)
    try:
        import torch
        from model.text_encoder import TextEncoder
        from model.fusion_decoder import FusionDecoder, GenerationConfig

        te   = TextEncoder(processor)
        fd   = FusionDecoder(model, processor, device)

        # 最短合法 prompt
        probe_prompt = (
            '[ACTION]: （從行動清單選一個）\n'
            '[THOUGHT]: （一句內心想法）\n'
            '[HAM]: []\n[/HAM]'
        )
        tp    = te.build_prompt('請完成以下格式：\n' + probe_prompt, num_images=0)
        fused = fd.fuse_inputs(text=tp.prompt, image_inputs={})
        fused['use_cache'] = False

        print('  [4/4] 推論中（首次約 10~30 秒）...', flush=True)
        _t3 = time.time()
        with torch.inference_mode():
            output = fd.generate(
                fused,
                GenerationConfig(max_new_tokens=60, temperature=0.0, do_sample=False)
            )
        print(f'  [4/4] 推論完成（{time.time()-_t3:.1f}s）', flush=True)

        _check('推論輸出非空',    bool(output and output.strip()))
        _check('推論輸出為字串', isinstance(output, str))
        _check('推論輸出長度 > 0', len(output.strip()) > 0,
               f'長度={len(output.strip())}')
        print(f'  推論輸出（前 80 字）：{output.strip()[:80]}')

        del model  # 釋放 GPU 記憶體，後面由 ModelLoader 重新載入
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f'  [ERROR] 推論健全性測試失敗：{e}')
        _results['fail'] += 1
        return False

    print('\n  模型驗證完成')
    return True


# ================================================================
# 主要模擬流程
# ================================================================

def run_simulation(use_real_model: bool = True, rounds: int = 2):
    from PIL import Image
    from model.model_loader import ModelLoader
    from world.world_clock import WorldClock
    from agent.agent_manager import AgentManager

    section('0. 重置測試環境')
    reset_test_data()

    section('1. 系統初始化')

    if use_real_model:
        print('  [INFO] 載入 Phi-3.5-Vision 模型（首次需要 5~10 分鐘）...')
        loader = ModelLoader()
        loader.load()
        print('  [INFO] 模型載入完成')
    else:
        print('  [INFO] 使用 MockLoader（假模型，快速驗證流程）')
        loader = MockLoader()

    clock          = WorldClock(start_time='07:00', minutes_per_tick=30)
    clock_day_init = clock.day
    manager        = AgentManager(loader=loader, clock=clock)

    _check('AgentManager 初始化', len(manager.all_codes()) == 5,
           f'codes={manager.all_codes()}')
    _check('WorldClock 初始時間', clock.time_str == '07:00')
    print(f'  角色：{manager.all_codes()}')
    print(f'  時間：{clock.scene_prefix()}')

    # 合成假圖（RGB 純藍色 320x240）
    synthetic_img = Image.fromarray(
        (np.ones((240, 320, 3), dtype=np.uint8) * [100, 149, 237]).astype(np.uint8)
    )

    # ── 多輪推論 ──────────────────────────────────────────────────
    for round_idx in tqdm(range(1, rounds + 1), desc='模擬輪數', unit='輪'):
        section(f'2.{round_idx} 第 {round_idx} 輪推論')
        clock.tick()
        print(f'  時間推進到：{clock.scene_prefix()}')

        round_events = [e for e in DEMO_EVENTS if e['round_offset'] == (round_idx - 1)]
        if not round_events:
            round_events = DEMO_EVENTS[:5]  # 超出定義範圍用預設事件

        for evt in tqdm(round_events, desc=f'  第{round_idx}輪角色', leave=False, unit='人'):
            code = evt['code']
            char = manager.get_character(code)
            subsection(f'{code}:{char.name}  {evt["desc"]}')

            image = synthetic_img if evt['image'] == 'synthetic' else None

            try:
                result = manager.step_character(
                    code        = code,
                    scene       = evt['scene'],
                    input_text  = evt['input_text'],
                    target_code = evt['target'],
                    image       = image,
                )

                print(f'    行動：{result["action"]}')
                if result['thought']:
                    print(f'    想法：{result["thought"][:60]}...'
                          if len(result['thought']) > 60 else f'    想法：{result["thought"]}')
                print(f'    模式：{result["mode"]}  |  C={result["confusion"]["C"]:.3f}  '
                      f'U={result["confusion"]["U"]:.2f} '
                      f'K={result["confusion"]["K"]:.2f} '
                      f'S={result["confusion"]["S"]:.2f}')
                print(f'    HAM：{len(result["ham"])} 筆')
                print(f'    should_sleep：{result["should_sleep"]}')
                validate_step_result(result, char.name)

            except Exception as e:
                print(f'    [ERROR] {e}')
                _results['fail'] += 1

        # STM 狀態快照
        subsection('STM 狀態快照')
        for code in sorted(manager.all_codes()):
            agent   = manager.get_agent(code)
            stm_cnt = agent.stm.count()
            ltm_cnt = agent.ltm.count()
            print(f'    {code}: STM={stm_cnt}筆  LTM={ltm_cnt}筆')

    # ── 睡眠濃縮 ──────────────────────────────────────────────────
    section('3. 睡眠濃縮（所有角色）')

    for code in tqdm(sorted(manager.all_codes()), desc='睡眠濃縮', unit='角色'):
        agent = manager.get_agent(code)
        char  = manager.get_character(code)
        subsection(f'{code}:{char.name}')

        stm_before  = agent.stm.count()
        ltm_before  = agent.ltm.count()
        day_before  = char.day
        print(f'    濃縮前 STM={stm_before}  LTM={ltm_before}  day={day_before}')

        try:
            # 透過 manager._do_sleep 完整走 clock 同步流程
            result = manager._do_sleep(code)
            print(f'    新增命題：{result["new_propositions"]}')
            print(f'    LTM 總計：{result["ltm_total"]}')
            print(f'    修剪命題：{result["ltm_pruned"]}')
            if result['summary']:
                print(f'    LTM 摘要：{result["summary"][:60]}...'
                      if len(result['summary']) > 60 else f'    LTM 摘要：{result["summary"]}')
            print(f'    情緒更新：{char.emotion}  |  推進到第 {char.day} 天')

            validate_sleep_result(result)
            _check('STM 清空',   agent.stm.count() == 0)
            _check('day 推進（+1）', char.day == day_before + 1,
                   f'{day_before}→{char.day}')
        except Exception as e:
            print(f'    [ERROR] {e}')
            _results['fail'] += 1

    # ── WorldClock 同步 ───────────────────────────────────────────
    section('4. WorldClock / Character day 同步驗證')
    if manager.all_sleeping():
        # clock.day 應比角色的 day 少 1（因為 clock 還未跨日 tick）
        # 或等於角色 day（advance_day 已被觸發）
        print(f'  所有角色已入睡，clock.day={clock.day}（初始={clock_day_init}）')
        _check('clock.day 推進 +1', clock.day == clock_day_init + 1,
               f'{clock_day_init}→{clock.day}')
    else:
        print(f'  （部分角色仍有 STM，跳過全局同步驗證）')

    # ── LTM 狀態 ─────────────────────────────────────────────────
    section('5. LTM 最終狀態')

    for code in sorted(manager.all_codes()):
        agent = manager.get_agent(code)
        char  = manager.get_character(code)
        props = agent.ltm.get_all()
        subsection(f'{code}:{char.name}  LTM={len(props)}筆  情緒={char.emotion}')

        for p in props[:5]:
            loc  = f'  @{p["location"]}' if p.get('location') else ''
            time = f'  [{p["time"]}]'     if p.get('time') else ''
            print(f'    {p["id"]}  {p["subject"]} -{p["relation"]}-> {p["object"]}'
                  f'{loc}{time}  '
                  f'strength={p["strength"]:.3f}  access={p["access_count"]}')
        if len(props) > 5:
            print(f'    ... 還有 {len(props)-5} 筆')

        validate_ltm_props(agent.ltm)

    # ── LTM ID 唯一性（prune 後新增不應重複）─────────────────────
    section('6. LTM ID 唯一性壓力測試')
    subsection('對 A 的 LTM 連續新增 / prune / 再新增')

    agent_a = manager.get_agent('A')
    ltm_a   = agent_a.ltm

    n_before = ltm_a.count()
    ltm_a.encode('Amy', '測試1', 'X', day=2)
    ltm_a.encode('Amy', '測試2', 'Y', day=2)
    ltm_a.encode('Amy', '測試3', 'Z', day=2)

    # 強制把中間那筆的 strength 降到可 prune 的值
    all_props = ltm_a.get_all()
    for p in all_props:
        if p['relation'] == '測試2':
            p['strength'] = 0.1  # 低於 FORGET_THRESHOLD
    removed = ltm_a.prune()
    print(f'    prune 移除 {removed} 筆（測試2）')

    ltm_a.encode('Amy', '測試4', 'W', day=2)
    final_ids = [p['id'] for p in ltm_a.get_all()]
    _check('prune 後 ID 不重複', len(final_ids) == len(set(final_ids)),
           f'ids={final_ids[-4:]}')
    print(f'    最後 4 個 ID：{final_ids[-4:]}')

    # ── 防遞迴驗證 ────────────────────────────────────────────────
    section('7. 對話防遞迴驗證')
    subsection('A→B 對話，B 回應不再傳給 A')

    call_log = []
    orig_b = manager._agents['B'].step

    def tracked_b(**kw):
        call_log.append('B_called')
        return orig_b(**kw)

    manager._agents['B'].step = tracked_b
    manager._forward_dialogue('A', 'B', '早安！', '咖啡廳')
    manager._agents['B'].step = orig_b  # 還原

    b_calls = call_log.count('B_called')
    _check('B 被呼叫 <= 1 次（防無限遞迴）', b_calls <= 1, f'實際呼叫 {b_calls} 次')

    # ── 最終統計 ──────────────────────────────────────────────────
    section('模擬結果彙整')
    total = _results['pass'] + _results['fail']
    rate  = _results['pass'] / total * 100 if total > 0 else 0
    print(f'  PASS : {_results["pass"]}')
    print(f'  FAIL : {_results["fail"]}')
    print(f'  總計 : {total} 項驗證')
    print(f'  通過率: {rate:.1f}%')
    print()
    if _results['fail'] == 0:
        print('  所有驗證通過')
    else:
        print(f'  有 {_results["fail"]} 項驗證失敗，請檢查上方 [FAIL] 項目')

    return _results['fail'] == 0


# ================================================================
# CLI 進入點
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AGI 模擬系統完整流程測試（含模型下載驗證）'
    )
    parser.add_argument(
        '--no-model', action='store_true',
        help='使用假模型（MockLoader），不需 GPU，快速驗證流程'
    )
    parser.add_argument(
        '--rounds', type=int, default=2,
        help='模擬輪數（預設 2）'
    )
    parser.add_argument(
        '--check-only', action='store_true',
        help='只下載並驗證模型，不執行完整模擬'
    )
    args = parser.parse_args()

    if args.no_model:
        # 假模型模式：跳過模型驗證，直接跑流程
        success = run_simulation(use_real_model=False, rounds=args.rounds)
    elif args.check_only:
        # 只驗證模型
        ok = validate_model()
        section('驗證結果')
        print(f'  PASS: {_results["pass"]}  FAIL: {_results["fail"]}')
        sys.exit(0 if ok else 1)
    else:
        # 完整模式：先驗證模型，再執行模擬
        model_ok = validate_model()
        if not model_ok:
            section('模型驗證失敗，中止模擬')
            sys.exit(1)
        success = run_simulation(use_real_model=True, rounds=args.rounds)

    sys.exit(0 if success else 1)
