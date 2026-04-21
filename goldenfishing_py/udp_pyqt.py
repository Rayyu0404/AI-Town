import socket
import struct
import sys
import time
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QGridLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from io import BytesIO
from PIL import Image
import asyncio
import websockets
import json
import base64

# === UDP 設定 ===
UDP_PORT = 9000
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_sock.bind(("0.0.0.0", UDP_PORT))
udp_sock.setblocking(False)

def decode_jpeg_packet(data: bytes):
    if len(data) < 12: return None
    char_id, w, h = struct.unpack("<iii", data[:12])
    jpeg_bytes = data[12:]
    try:
        image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return char_id, img_bgr
    except: return None

class SimpleStreamWindow(QWidget):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.setWindowTitle("PyQt WebSocket Client (Standby)")
        self.setGeometry(100, 100, 1280, 720)

        self.labels = {}
        grid = QGridLayout()
        for i in range(1, 6):
            img_label = QLabel(f"Waiting for ID {i}...")
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setStyleSheet("background-color: #111; color: #555; border: 2px solid #333;")
            img_label.setMinimumSize(320, 180)
            self.labels[i] = img_label
            grid.addWidget(img_label, (i-1)//3, (i-1)%3)

        self.setLayout(grid)
        self.frame_buffer = {i: None for i in range(1, 6)}
        self.prev_time = {i: 0 for i in range(1, 6)}
        self.fps_display = {i: 0 for i in range(1, 6)}

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33)

        self.receiver = QTimer()
        self.receiver.timeout.connect(self.receive_udp)
        self.receiver.start(1)

    def receive_udp(self):
        while True:
            try:
                data, _ = udp_sock.recvfrom(65535)
                result = decode_jpeg_packet(data)
                if result:
                    char_id, frame = result
                    if char_id in self.frame_buffer:
                        curr_time = time.time()
                        diff = curr_time - self.prev_time[char_id]
                        if diff > 0: self.fps_display[char_id] = 1.0 / diff
                        self.prev_time[char_id] = curr_time
                        
                        fps_text = f"ID: {char_id} | FPS: {self.fps_display[char_id]:.1f}"
                        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        self.frame_buffer[char_id] = frame
            except BlockingIOError: break
            except Exception as e:
                print(f"UDP Error: {e}")
                break

    def update_ui(self):
        for char_id, label in self.labels.items():
            frame = self.frame_buffer.get(char_id)
            if frame is not None:
                h, w, ch = frame.shape
                qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(qimg).scaled(
                    label.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                label.setPixmap(pixmap)

async def emit_two_images_ws(websocket, frame_buffer, id1=1, id2=2):
    frame1 = frame_buffer.get(id1)
    frame2 = frame_buffer.get(id2)
    if frame1 is not None and frame2 is not None:
        _, buf1 = cv2.imencode('.jpg', frame1, [cv2.IMWRITE_JPEG_QUALITY, 80])
        _, buf2 = cv2.imencode('.jpg', frame2, [cv2.IMWRITE_JPEG_QUALITY, 80])
        payload = {
            "event": "upload_images",
            "data": {
                "name1": chr(64+id1), "img1": base64.b64encode(buf1).decode('utf-8'),
                "name2": chr(64+id2), "img2": base64.b64encode(buf2).decode('utf-8')
            }
        }
        await websocket.send(json.dumps(payload))
        print("📤 影像已上傳至 Server")
    else:
        print("⚠️ 影像不足，取消上傳")

async def listen_ws(websocket, window):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("event") == "send_img_to_server":
                    print("🔔 收到擷取指令")
                    await emit_two_images_ws(websocket, window.frame_buffer)
                elif data.get("event") == "dialogue_result":
                    print(f"💬 收到對話結果: {data.get('data')}")
            except json.JSONDecodeError:
                print(f"📩 收到非 JSON 訊息: {message}")
    except websockets.ConnectionClosed:
        print("❌ WebSocket 連線已中斷")

async def main():
    # 1. 連線 Server
    try:
        ws = await websockets.connect("ws://localhost:8765")
        print("[WebSocket] 已連線至 Server")
        
        # --- 關鍵：傳送 Standby ---
        await ws.send("udp_pyqt_standby")
        print("📤 已傳送 'udp_pyqt_standby'")
    except Exception as e:
        print(f"❌ 無法連線至 Server: {e}")
        return

    # 2. 啟動 UI
    app = QApplication(sys.argv)
    window = SimpleStreamWindow(ws)
    window.show()

    # 3. 監聽指令
    asyncio.create_task(listen_ws(ws, window))

    # 4. 混合迴圈 (重要：確保程式不退出)
    while True:
        app.processEvents()
        await asyncio.sleep(0.01)

if __name__ == "__main__":
    try:
        # 使用 asyncio 執行
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 程式結束")