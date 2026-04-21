import io
import asyncio
import websockets
import json
import base64
import os
from PIL import Image
from llm_func import run_dual_dialogue

# 儲存連線與狀態
clients = {
    "UE": None,
    "PyQt": None
}

status = {
    "system_started": False
}

async def handle_upload_images(websocket, data):
    """
    處理影像上傳與 LLM 生成的異步任務 (終極穩定版)
    """
    try:
        # 1. 解碼與處理影像
        img1_bytes = base64.b64decode(data['img1'])
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2_bytes = base64.b64decode(data['img2'])
        img2 = Image.open(io.BytesIO(img2_bytes))

        print(f"✅ 收到影像：{data['name1']}, {data['name2']}")

        # 2. 執行 LLM 對話 (耗時任務)
        loop = asyncio.get_running_loop()
        dialogue = await loop.run_in_executor(
            None, 
            run_dual_dialogue, 
            data['name1'], img1, data['name2'], img2, 5
        )
        
        # 3. 回傳對話結果 (使用 try-except 保護)
        try:
            send_data = {'event': 'Dialogue', 'data': dialogue}
            if clients["UE"]:
                await clients["UE"].send(json.dumps(send_data, ensure_ascii=False))
                print("📤 對話結果已傳送給 UE")
        except Exception:
            print("⚠️ 對話結果回傳失敗：連線可能已斷開")

        # --- 循環邏輯 ---
        print("⏳ 等待 10 秒後進行下一次擷取...")
        await asyncio.sleep(10) 
        
        # 4. 再次要求 PyQt 傳送影像 (確保對象存在且嘗試發送)
        target_ws = clients.get("PyQt")
        if target_ws:
            try:
                trigger_cmd = json.dumps({"event": "send_img_to_server"})
                await target_ws.send(trigger_cmd)
                print("🔄 已自動發送循環指令給 PyQt")
            except Exception:
                print("⚠️ 循環指令發送失敗：PyQt 已斷線")
                clients["PyQt"] = None # 清理失效連線
                status["system_started"] = False

    except Exception as e:
        print(f"❌ 任務處理錯誤: {e}")

async def handler(websocket):
    global clients, status
    print(f"🔗 新連線進入: {websocket.remote_address}")

    try:
        async for message in websocket:
            # --- 判斷純字串就緒訊號 ---
            if message == "udp_pyqt_standby":
                clients["PyQt"] = websocket
                print("🤖 [系統狀態] PyQt 已就緒")
            
            elif message == "UE_standby":
                clients["UE"] = websocket
                print("🤖 [系統狀態] UE 已就緒")

            # --- 檢查雙方是否都到齊且尚未啟動 ---
            if clients["PyQt"] and clients["UE"] and not status["system_started"]:
                print("🚀 [全系統啟動] UE 與 PyQt 皆已連線，發送初始指令...")
                status["system_started"] = True
                
                trigger_cmd = json.dumps({"event": "send_img_to_server"})
                await clients["PyQt"].send(trigger_cmd)

            # --- 處理 JSON 數據 (影像上傳) ---
            if message.startswith('{'):
                try:
                    data = json.loads(message)
                    if data.get('event') == 'upload_images':
                        # 重要：使用 create_task 讓 handle_upload_images 在背景跑
                        # 這樣 handler 才能繼續處理後續的 WebSocket 封包 (心跳)
                        asyncio.create_task(handle_upload_images(websocket, data['data']))
                except Exception as e:
                    print(f"❌ JSON 處理錯誤: {e}")

    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 連線斷開: {websocket.remote_address}")
    finally:
        # 清理已斷開的連線
        if clients["PyQt"] == websocket: 
            clients["PyQt"] = None
            status["system_started"] = False
        if clients["UE"] == websocket: 
            clients["UE"] = None
            status["system_started"] = False

async def main():
    # 確保資料夾存在
    if not os.path.exists("./ai_data"):
        os.makedirs("./ai_data")
        print("📁 已建立 ai_data 資料夾")

    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("🚀 Server 運行中，等待雙端就緒 (Port: 8765)...")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Server 已手動停止")