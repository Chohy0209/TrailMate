import os
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ✅ 사용자 설정 --------------------------------
model_id = "skt/A.X-4.0-Light"
adapter_root_dir = "D:/trailmate_checkpoints/lora_ax_light_qvproj"
eval_file = "eval_prompts.txt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔧 생성 하이퍼파라미터
max_new_tokens = 768
temperature = 0.5
top_p = 0.8
repetition_penalty = 1.3
do_sample = True

# ✅ 프롬프트 템플릿
prompt_template = (
    "[역할] 당신은 캠핑을 가는 사람들을 도와주는 전문가 챗봇입니다. 캠핑 경험과 전문 지식을 바탕으로 실용적인 조언을 제공해야 합니다.\n"
    "[질문] {prompt}\n"
    "[금지] ⚠️ 질문과 직접 관련 없는 정보는 절대 포함하지 마세요. 캠핑과 무관한 내용(예: 엘리베이터, 도심, 일반 일상생활)은 생성하지 말아야 하며, 이런 내용은 응답 품질 저하로 간주됩니다.\n"
    "[답변 요구사항]\n"
    "1. 검색된 정보를 바탕으로 구체적인 답변을 제공하세요.\n"
    "2. 관련된 캠핑장, 장소, 장비가 있다면 구체적으로 언급하세요.\n"
    "3. 실용적인 팁이나 주의사항을 포함하세요.\n"
    "4. 정보가 부족하거나 명확하지 않다면 '추가 정보가 필요합니다'라고 명시하세요.\n"
    "[답변]"
)

# ✅ tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

# ✅ 프롬프트 불러오기
with open(eval_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# ✅ 체크포인트 목록 정렬
checkpoints = sorted([
    os.path.join(adapter_root_dir, d)
    for d in os.listdir(adapter_root_dir)
    if d.startswith("checkpoint-")
], key=lambda x: int(x.split("-")[-1]))

if not checkpoints:
    print("❌ 평가할 체크포인트가 없습니다.")
    exit()

# ✅ 평가 루프
for ckpt in checkpoints:
    checkpoint_name = os.path.basename(ckpt)
    log_dir = "evaluate"
    os.makedirs(log_dir, exist_ok=True)
    output_log_file = os.path.join(log_dir, f"evaluate-{checkpoint_name[11:]}.txt")

    print("=" * 100)
    print(f"🔍 Evaluating {checkpoint_name}")
    print("=" * 100)

    with open(output_log_file, "w", encoding="utf-8") as f:
        f.write(f"🔍 LoRA 체크포인트 평가 - {checkpoint_name}\n\n")

    # ✅ 모델 로딩 (8bit)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, ckpt)
    model.eval()

    for i, prompt in enumerate(prompts):
        full_prompt = prompt_template.format(prompt=prompt)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id
            )
        end = time.time()
        elapsed = end - start

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded.split("[답변]")[-1].strip()

        if len(answer.split()) < 10 or not answer.endswith((".", "다", "요")):
            print("⚠️ 응답이 중간에 끊겼을 수 있습니다.")
            answer += "\n[⚠️ 답변이 중간에 끊겼을 가능성이 있습니다.]"

        print(f"\n[Q{i+1}] {prompt}")
        print(f"[A{i+1}] {answer}")
        print(f"🕒 소요 시간: {elapsed:.2f}초")

        with open(output_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[Q{i+1}] {prompt}\n")
            f.write(f"[A{i+1}] {answer}\n")
            f.write(f"🕒 소요 시간: {elapsed:.2f}초\n")

    used = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\n🧠 VRAM 사용량: {used:.2f} GB (예약: {reserved:.2f} GB)")

    del model
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"✅ VRAM 초기화 완료 after {checkpoint_name}\n")
    print("#" * 100 + "\n")
