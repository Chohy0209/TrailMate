from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from peft import PeftModel
import torch
import time


# === 설정 ===
BASE_MODEL = "K-intelligence/Midm-2.0-Mini-Instruct"
LORA_ADAPTER_PATH = "./lora_model/midm_qlora_model"  # 파인튜닝된 LoRA 어댑터 경로

# # BitsAndBytesConfig 설정 (load_in_4bit 대신)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True
# )

# === 모델 및 토크나이저 로드 ===
print("🔄 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    # quantization_config=bnb_config,
    device_map="auto"
)
print("✅ Base model loaded.")

print("🔄 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
print("✅ LoRA adapter loaded.")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# === 대화 시작 ===
print("\n대화 시작! (종료하려면 'exit' 입력)\n")

while True:
    user_input = input("- 사용자: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("👋 대화를 종료합니다.")
        break

    start_time = time.time()
    # Midm 포맷에 맞춘 프롬프트 구성
    prompt = f"""### 질문:
{user_input}

### 답변:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 출력 텍스트 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 사용자 질문 이후의 부분만 추출
    answer = generated_text.split("### 답변:")[-1].strip()
    
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"- 모델: {answer}\n")
    print(f"경과 시간 : {elapsed:.2f}초\n")
