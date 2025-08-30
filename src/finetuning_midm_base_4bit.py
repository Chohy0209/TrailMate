import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login

# 🔐 Hugging Face 로그인 (토큰은 실제 발급받은 걸로 교체)
login(token="")

# ✅ 모델 설정
model_id = "K-intelligence/Midm-2.0-Base-Instruct"

# ✅ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ✅ QLoRA 4bit 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# ✅ 모델 로딩
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# ✅ LoRA 준비
model = prepare_model_for_kbit_training(model)

# ✅ LoRA 설정 (Attention 전체 조합)
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=[
        "self_attn.q_proj", "self_attn.v_proj",
        "self_attn.k_proj", "self_attn.o_proj","mlp.down_proj","mlp.gate_proj", "mlp.up_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ 데이터셋 로딩
dataset = load_dataset("json", data_files="./json/crawling_ultimate_create_copy.jsonl", split="train")

# ✅ 프롬프트 구성
def format_prompt(example):
    return {
        "text": f"<|user|>\n{example['게시물내용']}\n<|assistant|>\n{example['게시물제목']}"
    }

dataset = dataset.map(format_prompt)

# ✅ 토크나이즈
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize)

# ✅ Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir = "./lora_ax4_8bit_qvproj",       # 저장 위치
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=8,                       # ✅ 8 epoch 학습
    logging_steps=10,
    bf16=True,
    save_total_limit=8,                       # 최대 8개 checkpoint 보존
    save_strategy="epoch",                    # ✅ 매 epoch마다 저장
    report_to="none"
)

# ✅ Trainer 구성
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ✅ 학습 시작
trainer.train()
