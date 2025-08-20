import json
import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from bert_score import score
import csv
# ===== 설정 =====
client = OpenAI(api_key="??")  # OpenAI API 키

# 로컬 LoRA 모델 경로
LOCAL_MODELS = {
    "Local qLoRA v1": "venv_3123/model_ax_merge4",
    "Local qLoRA v2": "venv_3123/kt_midm_mini"  # 새로운 로컬 모델
}

BERT_MODEL = "klue/roberta-large"

# OpenAI GPT 모델들
GPT_MODELS = [
    "ft:gpt-4.1-2025-04-14:ailab::C2eTINXG:ckpt-step-357",
    "ft:gpt-4.1-2025-04-14:ailab::C2eTJjAn:ckpt-step-714",
    "ft:gpt-4.1-mini-2025-04-14:ailab:camping-rag-qa:C2bHhwJM:ckpt-step-714"
]

# 총 모델 리스트
MODEL_NAMES = list(LOCAL_MODELS.keys()) + GPT_MODELS

JSON_FILE = "q_and_a.json"  # 30문항 JSON 경로

# ===== 로컬 모델 로딩 =====
print("로컬 모델 로딩 중...")
local_tokenizers = {}
local_models = {}
for name, path in LOCAL_MODELS.items():
    print(f" - {name} 로딩 중 ({path})...")

    if "model_ax_merge4" in name:  # 4bit 퀀타이즈 모델
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            load_in_4bit=True
        )
    else:  # 일반 LoRA 모델
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.float16
        )

    mod.eval()
    local_tokenizers[name] = tok
    local_models[name] = mod

print("로컬 모델 준비 완료.")

# ===== 답변 생성 함수 =====
def generate_local_answer(model_name: str, question: str) -> str:
    tokenizer = local_tokenizers[model_name]
    model = local_models[model_name]
    inputs = tokenizer(
    question,
    return_tensors="pt",
    padding=True,      # 배치 내 가장 긴 시퀀스 길이에 맞춰 패딩
    truncation=True    # 최대 길이 초과 시 자르기
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,  # 반드시 전달
        max_length=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_openai_answer(model_name: str, question: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": question}],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API 오류 ({model_name}): {e}")
        return ""

# ===== 메인 평가 로직 =====
def main():
    # JSON 로드
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    # 결과 저장 구조
    candidates_dict = {name: [] for name in MODEL_NAMES}
    references_dict = {name: [] for name in MODEL_NAMES}
    time_records = {name: [] for name in MODEL_NAMES}

    # 문항별 답변 생성
    for idx, qa in enumerate(qa_list, start=1):
        question = qa["질문"]
        reference = qa["대답"]

        print(f"\n[{idx}/{len(qa_list)}] 질문: {question}")

        for model_name in MODEL_NAMES:
            start_time = time.time()
            if model_name in LOCAL_MODELS:
                answer = generate_local_answer(model_name, question)
            else:
                answer = generate_openai_answer(model_name, question)
            elapsed = time.time() - start_time

            candidates_dict[model_name].append(answer)
            references_dict[model_name].append(reference)
            time_records[model_name].append(elapsed)

            print(f"[{model_name}] ({elapsed:.2f}s) {answer}")

    # ===== 성능 분석 =====
    print("\n=== 모델별 BERTScore + 속도 분석 ===")
    results = []
    for name in MODEL_NAMES:
        P, R, F1 = score(
            candidates_dict[name],
            references_dict[name],
            model_type=BERT_MODEL,
            lang="ko",
            verbose=False,
            num_layers=24
        )
        avg_time = sum(time_records[name]) / len(time_records[name])
        std_f1 = statistics.stdev(F1.tolist())
        min_time = min(time_records[name])
        max_time = max(time_records[name])
        efficiency = F1.mean().item() / avg_time  # 성능-속도 효율

        results.append({
            "model": name,
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
            "f1_std": std_f1,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "efficiency": efficiency
        })

    # ===== 결과 출력 =====
    for r in results:
        csv_file = "model_eval_results.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "precision", "recall", "f1", "f1_std",
                "avg_time", "min_time", "max_time", "efficiency"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
                print(f"{r['model']} | "
                    f"F1: {r['f1']:.4f} (±{r['f1_std']:.4f}), "
                    f"P: {r['precision']:.4f}, R: {r['recall']:.4f} | "
                    f"AvgTime: {r['avg_time']:.2f}s "
                    f"(Min: {r['min_time']:.2f}s, Max: {r['max_time']:.2f}s) | "
                    f"효율(F1/Time): {r['efficiency']:.4f}")
        print(f"\n✅ 결과가 '{csv_file}'로 저장되었습니다.")
        print(f"{r['model']} | "
              f"F1: {r['f1']:.4f} (±{r['f1_std']:.4f}), "
              f"P: {r['precision']:.4f}, R: {r['recall']:.4f} | "
              f"AvgTime: {r['avg_time']:.2f}s "
              f"(Min: {r['min_time']:.2f}s, Max: {r['max_time']:.2f}s) | "
              f"효율(F1/Time): {r['efficiency']:.4f}")

if __name__ == "__main__":
    main()
