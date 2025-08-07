from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch


class LLMModel:
    def __init__(self, model_name="K-intelligence/Midm-2.0-Mini-Instruct", multi_turn=False, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device)
        self.multi_turn = multi_turn
        self.history = []  # [(user, bot), ...]

    def build_prompt(self, user_input):
        if not self.multi_turn or not self.history:
            return f"### User: {user_input}\n### Assistant:"
        else:
            history_text = ""
            for user, bot in self.history:
                history_text += f"### User: {user}\n### Assistant: {bot}\n"
            return history_text + f"### User: {user_input}\n### Assistant:"

    def chat(self, user_input, max_new_tokens=512):
        prompt = self.build_prompt(user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs.pop("token_type_ids", None)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output.split("### Assistant:")[-1].strip()

        if self.multi_turn:
            self.history.append((user_input, answer))

        return answer

    def reset_history(self):
        self.history = []


if __name__ == "__main__":
    model = LLMModel(multi_turn=True)  # 또는 False
    print("LLM 챗봇 시작! 종료하려면 'exit' 입력")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        response = model.chat(user_input)
        print("Bot:", response)






# # llm_model.py
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# class LLMChatModel:
#     def __init__(self, model_name="K-intelligence/Midm-2.0-Mini-Instruct", device = None):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto" if torch.cuda.is_available() else None
#         )
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # 채팅 히스토리
#         self.chat_history = []

#         # 시스템 메시지 설정 (선택적)
#         self.system_prompt = "당신은 친절하고 유능한 도우미입니다."

#     def generate_prompt(self, user_input):
#         prompt = ""
#         if self.system_prompt:
#             prompt += f"[System]\n{self.system_prompt}\n"

#         for user, bot in self.chat_history:
#             prompt += f"[User]: {user}\n[Assistant]: {bot}\n"

#         prompt += f"[User]: {user_input}\n[Assistant]:"
#         return prompt

#     def chat(self, user_input, max_new_tokens=512):
#         prompt = self.generate_prompt(user_input)

#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         # token_type_ids 제거
#         if "token_type_ids" in inputs:
#             inputs.pop("token_type_ids")

#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             pad_token_id=self.tokenizer.eos_token_id
#         )

#         output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # 마지막 Assistant 응답만 추출
#         response = output_text.split("[Assistant]:")[-1].strip()

#         # 히스토리 갱신
#         self.chat_history.append((user_input, response))
#         return response


# # CLI 테스트용 실행
# if __name__ == "__main__":
#     chat_model = LLMChatModel()

#     print("🔹 K-intelligence Midm-2.0-Mini-Instruct Chatbot 🔹")
#     while True:
#         try:
#             user_input = input("You: ")
#             if user_input.lower() in ["exit", "quit"]:
#                 break
#             response = chat_model.chat(user_input)
#             print("AI:", response)
#         except KeyboardInterrupt:
#             print("\n종료합니다.")
#             break
