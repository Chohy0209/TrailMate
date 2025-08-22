# services.py
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
import config

# OpenAI API 클라이언트 초기화
client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0)

# Neo4j 데이터베이스 드라이버 초기화
driver = AsyncGraphDatabase.driver(
    config.NEO4J_URI, 
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)