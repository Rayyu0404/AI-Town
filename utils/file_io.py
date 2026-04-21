# ================================================================
# utils/file_io.py
# 角色 JSON 的讀寫操作
# 執行中角色狀態存在記憶體（dict），
# 只在特定時機（睡眠濃縮後）才寫回磁碟
# ================================================================

import json
import os
from config.world_config import AI_DATA_DIR, CHARACTER_NAMES


def load_character(code: str) -> dict:
    """
    從磁碟讀取角色 JSON，回傳 dict。
    code : 角色代號（A B C D E）
    """
    path = _get_path(code)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到角色檔案：{path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_character(code: str, data: dict):
    """
    將角色 dict 寫回磁碟。
    只在睡眠濃縮或程式關閉時呼叫，不要每輪都寫。
    """
    path = _get_path(code)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_all_characters() -> dict:
    """
    載入 AI_Data 資料夾內所有角色，回傳 {code: data} dict。
    """
    characters = {}
    for filename in os.listdir(AI_DATA_DIR):
        if not filename.endswith("_init.json"):
            continue
        code = filename.replace("_init.json", "")
        if code not in CHARACTER_NAMES:
            continue  # 忽略不在角色表中的檔案
        characters[code] = load_character(code)
    return characters


def save_all_characters(characters: dict):
    """
    將所有角色寫回磁碟。
    characters : {code: data} dict
    """
    for code, data in characters.items():
        save_character(code, data)


def _get_path(code: str) -> str:
    return os.path.join(AI_DATA_DIR, f"{code}_init.json")