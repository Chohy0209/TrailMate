# # qlora_finetune.py
# import os
# import json
# import torch
# from datasets import load_dataset, Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
# from trl import SFTTrainer, SFTConfig

# # 모델과 토크나이저 설정
# model_name = "K-intelligence/Midm-2.0-Base-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_4bit=True,
#     device_map="auto",
#     trust_remote_code=True
# )

# # LoRA 설정
# model = prepare_model_for_kbit_training(model)

# lora_config = LoraConfig(
#     r = 16, # r 자기 데이터 반영 비율
#     lora_alpha = 32,
#     target_modules = ["q_proj", "v_proj"],  
#     lora_dropout = 0.05,
#     bias = "none",
#     task_type = "CAUSAL_LM"
# )

# model = get_peft_model(model, lora_config)

# # 데이터셋 로딩 함수
# def load_custom_dataset(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         raw_data = json.load(f)
#     formatted = []
#     for item in raw_data:
#         prompt = f"[제목] {item['게시물제목']}\n[내용] {item['게시물내용']}"
#         formatted.append({"text": prompt})
#     return Dataset.from_list(formatted)

# # 데이터셋 불러오기
# dataset = load_custom_dataset("./data/crawling_ultimate.json")

# # 학습 설정
# training_args = TrainingArguments(
#     output_dir = "./lora_model/midm_base_qlora",
#     per_device_train_batch_size = 2,
#     gradient_accumulation_steps = 4,
#     logging_steps = 10,
#     save_steps = 100,
#     learning_rate = 2e-4,
#     num_train_epochs = 3,
#     fp16 = True,
#     save_total_limit = 2,
#     report_to = "none",
#     # max_seq_length = 2048,
# )

# # training_args = SFTConfig(
# #     output_dir = "./midm_qlora",
# #     per_device_train_batch_size = 4,
# #     gradient_accumulation_steps = 4,
# #     logging_steps = 10,
# #     save_steps = 100,
# #     learning_rate = 2e-4,
# #     num_train_epochs = 3,
# #     fp16 = True,
# #     save_total_limit = 2,
# #     report_to = "none",
# #     max_seq_length = 2048,
# # )

# # SFT Trainer로 파인튜닝
# trainer = SFTTrainer(
#     model = model,
#     train_dataset = dataset,
#     processing_class = tokenizer,
#     args = training_args,
#     formatting_func=lambda x : x["text"],
# )

# trainer.train()
# trainer.save_model("./lora_model/midm_base_qlora_model")




# qlora_finetune.py
import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

# BitsAndBytesConfig 설정 (load_in_4bit 대신)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 모델과 토크나이저 설정
model_name = "K-intelligence/Midm-2.0-Mini-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 패딩 토큰 설정 (필요한 경우)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,  # load_in_4bit 대신
    device_map="auto",
    trust_remote_code=True
)

# LoRA 설정
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],      
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 데이터셋 로딩 함수
def load_custom_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    formatted = []
    for item in raw_data:
        prompt = f"[제목] {item['게시물제목']}\n[내용] {item['게시물내용']}"
        formatted.append({"text": prompt})
    return Dataset.from_list(formatted)

# 데이터셋 불러오기
dataset = load_custom_dataset("./data/crawling_ultimate.json")

# 데이터셋 전처리 (길이 제한 적용)
def preprocess_function(examples):
    texts = examples["text"]
    # 토크나이징 및 길이 제한
    model_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None
    )
    # 다시 텍스트로 변환
    examples["text"] = [tokenizer.decode(ids, skip_special_tokens=True) for ids in model_inputs["input_ids"]]
    return examples

# 데이터셋에 전처리 적용
dataset = dataset.map(preprocess_function, batched=True)

# 학습 설정 (fp16 문제 해결)
training_args = TrainingArguments(
    output_dir="./lora_model/midm_qlora",
    per_device_train_batch_size=2,  # 배치 사이즈 줄임
    gradient_accumulation_steps=8,  # 대신 accumulation 늘림
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,  # fp16 대신 bf16 사용 (더 안정적)
    # fp16=True,  # 문제가 있으면 주석처리
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,  # 메모리 문제 해결
    remove_unused_columns=False,  # 추가 안정성
    
)

# SFT Trainer로 파인튜닝
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=lambda x : x["text"],
    args=training_args,
   
)

# 학습 시작
print("Starting training...")
trainer.train()

# 모델 저장
print("Saving model...")
trainer.save_model("./lora_model/midm_qlora_model")
