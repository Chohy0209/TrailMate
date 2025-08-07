# chat_api.py
from llm_model import LLMModel

# 전역 모델 객체 생성
llm_instance = LLMModel()

def chat_with_model(message, history = None):
    return llm_instance.chat(message)
