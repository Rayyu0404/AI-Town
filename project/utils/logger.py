# ================================================================
# utils/logger.py
# 系統 log，每天產生一個 log 檔案
# 使用方式：from utils.logger import get_logger
#           logger = get_logger()
#           logger.info("訊息")
# ================================================================

import logging
import os
from datetime import datetime
from config.world_config import LOG_DIR


def get_logger(name: str = "world") -> logging.Logger:
    """
    取得 logger，同時輸出到 console 和當天的 log 檔。
    多次呼叫同一個 name 會回傳同一個 logger（不重複建立）。
    """
    logger = logging.getLogger(name)

    # 已經初始化過就直接回傳
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 防止 root logger 重複輸出

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # ── console handler ──────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── file handler（每天一個檔）───────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(LOG_DIR, f"{today}.log")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_turn(logger: logging.Logger, code: str, turn_id: str,
             action: str, c_value: float, mode: str):
    """
    每輪結束後記錄一筆摘要 log。
    code    : 角色代號（A B C D E）
    turn_id : 輪次 ID（例如 D001_T003）
    action  : 模型決定的行動
    c_value : 本輪困惑指數 C 值
    mode    : "intuitive" 或 "deliberate"
    """
    logger.info(
        f"[{code}] {turn_id} | mode={mode} | C={c_value:.3f} | action={action}"
    )


def log_consolidation(logger: logging.Logger, code: str, day: int,
                      stm_count: int, ltm_count: int):
    """
    STM->LTM 濃縮完成後記錄。
    """
    logger.info(
        f"[{code}] Day {day} 睡眠濃縮完成 | "
        f"STM={stm_count} 筆 -> LTM 新增後共 {ltm_count} 筆命題"
    )