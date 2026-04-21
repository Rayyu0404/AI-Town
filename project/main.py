# ================================================================
# main.py
# 程式進入點
#
# 兩種執行模式：
#   python main.py          → 啟動 WebSocket 伺服器（與 UE 連接）
#   python main.py --demo   → 離線模擬模式（不需要 UE，用假資料測試）
# ================================================================

import argparse
import sys

from model.model_loader import ModelLoader
from world.world_clock import WorldClock
from agent.agent_manager import AgentManager
from perception.yolo_handler import YoloHandler
from server.ws_server import WSServer
from utils.logger import get_logger

logger = get_logger("main")


def build_system() -> tuple:
    """
    載入模型、初始化時鐘與 AgentManager。
    回傳 (loader, clock, manager, yolo)。
    """
    logger.info("=== AI 角色自主生活模擬系統 啟動 ===")

    # 1. 載入模型（所有 Agent 共用）
    loader = ModelLoader()
    loader.load()

    # 2. 建立世界時鐘
    clock = WorldClock(start_time="07:00", minutes_per_tick=30)

    # 3. 建立角色管理器
    manager = AgentManager(loader=loader, clock=clock)

    # 4. 建立 YOLO 處理器
    yolo = YoloHandler()

    return loader, clock, manager, yolo


# ── 模式 1：WebSocket 伺服器（與 UE 連接）────────────────────────

def run_server(host: str = "localhost", port: int = 8765):
    """啟動 WebSocket 伺服器，等待 UE 連線。"""
    loader, clock, manager, yolo = build_system()
    server = WSServer(manager=manager, yolo=yolo, host=host, port=port)
    server.run()


# ── 模式 2：離線模擬（不需要 UE）────────────────────────────────

def run_demo(rounds: int = 3):
    """
    離線模擬：對每個角色執行數輪推論，觀察記憶與決策行為。
    不需要 UE 或 YOLO，使用純文字模擬輸入。
    """
    loader, clock, manager, _ = build_system()

    demo_events = [
        ("A", "咖啡廳，早晨，陽光充足", "Ben走進咖啡廳，對你微笑", "B"),
        ("B", "咖啡廳，早晨", "Amy正在擦拭吧台，沒有看過來", "A"),
        ("C", "辦公室，上午，安靜", "", None),
        ("D", "辦公室，上午，有點悶熱", "Claire給你一份報告", "C"),
        ("E", "餐廳，午餐時間，忙碌", "", None),
    ]

    logger.info(f"=== 開始離線模擬，共 {rounds} 輪 ===")

    for round_idx in range(1, rounds + 1):
        logger.info(f"--- 第 {round_idx} 輪 ---")
        clock.tick()

        for code, scene, input_text, target in demo_events:
            try:
                result = manager.step_character(
                    code        = code,
                    scene       = scene,
                    input_text  = input_text,
                    target_code = target,
                )
                char = manager.get_character(code)
                print(
                    f"[{char.name}] {result['mode'].upper()} "
                    f"C={result['confusion']['C']:.2f} "
                    f"→ {result['action']}"
                )
                print(f"  THOUGHT: {result['thought']}")
            except Exception as e:
                logger.error(f"[{code}] 模擬失敗：{e}")

    logger.info("=== 模擬結束 ===")


# ── CLI 進入點 ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI 角色自主生活模擬系統"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="離線模擬模式（不需要 UE）"
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="模擬輪數（demo 模式用）"
    )
    parser.add_argument(
        "--host", default="localhost",
        help="WebSocket 伺服器 host"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="WebSocket 伺服器 port"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo(rounds=args.rounds)
    else:
        run_server(host=args.host, port=args.port)
