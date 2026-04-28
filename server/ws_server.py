# ================================================================
# server/ws_server.py
# WebSocket 伺服器：Unreal Engine ↔ Python 通訊橋接
#
# UE → Python（接收格式）：
# {
#   "character": "A",          # 角色代號
#   "scene":     "咖啡廳早上", # 場景描述
#   "image_b64": "...",        # base64 編碼的 PNG（可省略）
#   "input_text": "...",       # 對話或事件文字（可省略）
#   "target":    "B",          # 對話對象代號（可省略）
# }
#
# Python → UE（回傳格式）：
# {
#   "character": "A",
#   "action":    "前往:咖啡廳",
#   "thought":   "Amy想去咖啡廳看看。",
#   "mode":      "intuitive",
#   "error":     null
# }
# ================================================================

import asyncio
import base64
import io
import json

from PIL import Image
from typing import Optional

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

from agent.agent_manager import AgentManager
from perception.yolo_handler import YoloHandler
from utils.logger import get_logger

logger = get_logger("ws_server")

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 8765


class WSServer:
    """
    WebSocket 伺服器，接受 Unreal Engine 的連線。
    每條訊息路由給對應角色的 Agent，回傳行動指令。
    """

    def __init__(self, manager: AgentManager,
                 yolo: Optional[YoloHandler] = None,
                 host: str = _DEFAULT_HOST,
                 port: int = _DEFAULT_PORT):
        self.manager = manager
        self.yolo    = yolo or YoloHandler()
        self.host    = host
        self.port    = port

    # ── 啟動伺服器 ────────────────────────────────────────────────

    def run(self):
        """啟動 WebSocket 伺服器（阻塞）。"""
        if not _WS_AVAILABLE:
            raise RuntimeError(
                "websockets 未安裝，請執行：pip install websockets"
            )
        logger.info(f"WebSocket 伺服器啟動：ws://{self.host}:{self.port}")
        asyncio.run(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()  # 持續執行

    # ── 單次連線處理 ──────────────────────────────────────────────

    async def _handler(self, websocket):
        client = websocket.remote_address
        logger.info(f"UE 連線：{client}")
        try:
            async for raw_msg in websocket:
                response = await self._process(raw_msg)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except Exception as e:
            logger.error(f"連線 {client} 發生錯誤：{e}")
        finally:
            logger.info(f"UE 斷線：{client}")

    async def _process(self, raw_msg: str) -> dict:
        """解析 UE 訊息，呼叫 AgentManager，組裝回應。"""
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError as e:
            return {"error": f"JSON 解析失敗：{e}"}

        code       = msg.get("character", "").upper()
        scene      = msg.get("scene", "")
        image_b64  = msg.get("image_b64", "")
        input_text = msg.get("input_text", "")
        target     = msg.get("target", None)

        if code not in self.manager.all_codes():
            return {"character": code, "error": f"未知角色代號：{code}"}

        # 解碼圖片
        image, image_desc = None, ""
        if image_b64:
            image, image_desc = self._decode_image(image_b64, scene)

        try:
            result = self.manager.step_character(
                code        = code,
                scene       = scene,
                image       = image,
                input_text  = input_text,
                image_desc  = image_desc,
                target_code = target,
            )
            return {
                "character": code,
                "action":    result["action"],
                "thought":   result["thought"],
                "mode":      result["mode"],
                "error":     None,
            }
        except Exception as e:
            logger.error(f"[{code}] 推論失敗：{e}")
            return {"character": code, "action": "休息", "error": str(e)}

    def _decode_image(self, image_b64: str,
                      scene: str) -> tuple:
        """
        解碼 base64 圖片，執行 YOLO 偵測並回傳 (PIL Image, 語意描述)。
        """
        try:
            img_bytes = base64.b64decode(image_b64)
            image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            _, desc   = self.yolo.process(image, location=scene)
            return image, desc
        except Exception as e:
            logger.warning(f"圖片解碼失敗：{e}")
            return None, ""
