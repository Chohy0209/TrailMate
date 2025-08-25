# config.py
import os
from dotenv import load_dotenv

# from naver_api import NAVER_CONCURRENCY

load_dotenv()

# --- API Keys & Secrets ---
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
TMAP_API_KEY = os.getenv("TMAP_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NAVER_CONCURRENCY = 3

# --- Model & RAG Parameters ---
EMBEDDING_MODEL_NAME = "dragonkue/BGE-m3-ko"
GPT_MODEL = "ft:gpt-4.1-mini-2025-04-14:ailab:camping-rag-qa:C2bHhwJM:ckpt-step-714"
VECTOR_DIM = 1024
SIM_FUNC = "cosine"
TEMPERATURE = 0.7
MAX_TOKENS = 512


# --- Search Parameters ---
TOPK_CAMP, TOPK_ATTR, TOPK_SUM = 100, 100, 100
ROLLUP_LIMIT, FINAL_TOPN = 2, 2
WEIGHT_CAMP, WEIGHT_ATTR, WEIGHT_SUM = 2, 0.3, 1.2
