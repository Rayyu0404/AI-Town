import json
import torch
import gc
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

_MODEL = None
_PROCESSOR = None

def initialize_model():
    global _MODEL, _PROCESSOR

    if _MODEL is not None and _PROCESSOR is not None:
        print("模型已經載入，跳過初始化。")
        return
    
    print("正在初始化 Phi-3.5-vision 模型，請稍候...")
    
    model_id = "microsoft/Phi-3.5-vision-instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, 
    )

    try:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2"
        ).eval()

        _PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
        print("模型與處理器載入完成。")

    except Exception as e:
        print(f"模型載入失敗: {e}")
        raise e

def unload_model():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        del _MODEL
        del _PROCESSOR
        torch.cuda.empty_cache()
        gc.collect()
        _MODEL = None
        _PROCESSOR = None
        print("模型已釋放，VRAM 已清理。")

class STM:
    def __init__(self, owner_name):
        self.owner = owner_name
        self.personality = self.load_personality(owner_name) # 這裡稍微修復讀取邏輯
        self.logs = self.load_ltm(owner_name) 

    def load_personality(self, name):
        # 簡單的防呆，避免檔案找不到崩潰
        path = f"./ai_data/{name}.json"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return data.get('personality')
        else:
            return ""
        
    def load_ltm(self, name):
        path = f"./ai_data/{name}.json"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                # 確保回傳的是 list，因為你的 add() 方法會用到 .append()
                memory = data.get('memory', [])
                return memory if isinstance(memory, list) else [str(memory)]
        return [] # 找不到檔案時回傳空串列
        
    # 您在 run_dual_dialogue 呼叫 mem.add()，所以這裡統一用 add
    def add(self, text):
        self.logs.append(text)
        if len(self.logs) > 20: 
            self.logs.pop(0)

    def get_context_string(self):
        return "\n".join(self.logs)
    
    def get_personality(self):
        return self.personality
    
    def get_logs(self):
        return self.logs

# 【修正】這裡的 image 參數預期接收 PIL Image 物件，而非路徑
def generate_text_phi3_vision(prompt_text, image, max_new_tokens=128, stop_sequences=None, verbose=False):
    global _MODEL, _PROCESSOR

    if _MODEL is None or _PROCESSOR is None:
        initialize_model()

    # 1. 處理 Prompt 格式
    # Phi-3 必須要有 <|image_1|> 標籤告訴模型圖片在哪裡
    # 如果傳入 image 物件，就加上標籤
    image_tag = "<|image_1|>\n" if image is not None else ""
    
    full_prompt = f"<|user|>\n{image_tag}{prompt_text}<|end|>\n<|assistant|>\n"

    # 2. 轉 tensor (直接使用傳入的 image 物件)
    inputs = _PROCESSOR(
        text=full_prompt, 
        images=image,  # 這裡直接吃 PIL Object
        return_tensors="pt"
    ).to("cuda")

    input_len = inputs.input_ids.shape[1]

    # 3. 生成
    with torch.no_grad():
        generate_ids = _MODEL.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            eos_token_id=_PROCESSOR.tokenizer.eos_token_id
        )

    # 4. 解碼
    generated_tokens = generate_ids[:, input_len:]
    response = _PROCESSOR.batch_decode(
        generated_tokens, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0].strip()

    # 5. 停止詞處理 (模擬 llama.cpp 的 stop 參數)
    if stop_sequences:
        for stop in stop_sequences:
            if stop in response:
                response = response.split(stop)[0]

    return response.strip()

def summarize_to_ltm(name, img, mem_array):
    path = f"./ai_data/{name}.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompt = '將以下文字與圖片內容用一句話總結重點: '
    for mem in mem_array:
        prompt += mem
    summarize = generate_text_phi3_vision(
            prompt_text=prompt,
            image=img, 
            max_new_tokens=100,
            stop_sequences=["\n"],
            verbose=False
            )
    
    data['memory'].append(summarize)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(name, '記憶已新增: ', summarize)

def run_dual_dialogue(
    char_a_name, char_a_img, 
    char_b_name, char_b_img, 
    max_turns=6
):
    mem_a = STM(char_a_name)
    mem_b = STM(char_b_name)

    structured_result = []

    current_name = char_a_name
    current_img = char_a_img
    current_mem = mem_a
    
    opponent_name = char_b_name

    print(f"--- 對話開始 ({char_a_name} vs {char_b_name}) ---")

    for i in range(max_turns):
        context_str = current_mem.get_context_string()
        
        prompt_text = (
            f"### 角色身份：{current_name}\n"
            f"### 你的性格與背景：\n{current_mem.get_personality()}\n\n"
            f"### 最近的對話歷程（參考）：\n{context_str}\n\n"
            f"### 當前任務：\n"
            f"你正透過第一人稱視角看著眼前的畫面並與 {opponent_name} 對話。\n"
            f"請根據畫面內容與當前氣氛，自然地說出一句話。\n\n"
            f"### 限制：\n"
            f"1. 嚴禁重複對話紀錄中已出現過的語句。\n"
            f"2. 嚴禁描述自己的行為（如：我會用開玩笑的方式...）。\n"
            f"3. 嚴禁輸出「好的」、「我知道了」等系統式回應。\n"
            f"4. 直接輸出你的對話內容，不帶姓名與引號。\n"
            f"回應內容："
        )

        reply = generate_text_phi3_vision(
            prompt_text=prompt_text,
            image=current_img, 
            max_new_tokens=100,
            stop_sequences=["\n"],
            verbose=False
        )

        if not reply:
            reply = "..."

        turn_data = {
            "speaker": current_name,
            "listener": opponent_name,
            "text": reply
        }
        structured_result.append(turn_data)

        print(f">> {current_name} -> {opponent_name}: {reply}")

        log_entry = f"{current_name}：{reply}"
        mem_a.add(log_entry)
        mem_b.add(log_entry)

        if current_name == char_a_name:
            current_name, current_img, current_mem = char_b_name, char_b_img, mem_b
            opponent_name = char_a_name
        else:
            current_name, current_img, current_mem = char_a_name, char_a_img, mem_a
            opponent_name = char_b_name
    
    summarize_to_ltm(char_a_name, char_a_img, mem_a.get_logs())
    summarize_to_ltm(char_b_name, char_b_img, mem_b.get_logs())

    return structured_result