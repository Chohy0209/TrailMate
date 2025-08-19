import os
import time
import asyncio
import random
import operator
from typing import TypedDict, Annotated, List, Any, Dict

#naver api
from naver_api import build_snippet_per_doc, format_snippets_as_text

# OpenAI SDK
from openai import OpenAI

# LangChain / Vector store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings.base import Embeddings

# BGE-M3 통합 사용
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

# ===== 환경 설정 =====
# ⚠️ 키는 환경변수로 읽습니다: export OPENAI_API_KEY="sk-..."
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,
)

EMBEDDING_MODEL_NAME = "dragonkue/BGE-m3-ko"
CHROMA_DB_PATH = "./data/camp_vectorDB_BGE_sentence_per_doc"
MAX_LOOPS = 2

GPT_MODEL = "ft:gpt-4.1-mini-2025-04-14:ailab:camping-rag-qa:C2bHhwJM:ckpt-step-714"

# ✅ 새로운 통합 BGE-M3 클래스
class UnifiedBGEM3Embedder:
    def __init__(self, model_name="dragonkue/BGE-m3-ko"):
        self.model = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """BGE-M3 모델 로드"""
        try:
            self.model = BGEM3FlagModel(self.model_name, use_fp16=True)
            print(f"✅ BGE-M3 통합 모델 로드 완료: {self.model_name}")
        except Exception as e:
            print(f"❌ BGE-M3 한국어 모델 로드 실패: {e}")
            try:
                # Fallback: 원본 BGE-M3 모델
                self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
                print("✅ BGE-M3 원본 모델로 Fallback 로드 완료")
            except Exception as e2:
                print(f"❌ BGE-M3 Fallback도 실패: {e2}")
                self.model = None
    
    def encode_for_vector_db(self, texts, batch_size=12):
        """ChromaDB용 dense embedding 생성"""
        if self.model is None:
            raise Exception("BGE-M3 모델이 로드되지 않음")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return embeddings['dense_vecs']
    
    def encode_hybrid(self, query_text, candidate_texts, batch_size=12):
        """하이브리드 검색용 dense + sparse 동시 계산"""
        if self.model is None:
            raise Exception("BGE-M3 모델이 로드되지 않음")
        
        all_texts = [query_text] + candidate_texts
        embeddings = self.model.encode(
            all_texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        return {
            'query_dense': embeddings['dense_vecs'][0],
            'doc_dense': embeddings['dense_vecs'][1:],
            'query_sparse': embeddings['lexical_weights'][0],
            'doc_sparse': embeddings['lexical_weights'][1:]
        }

# ✅ ChromaDB용 커스텀 임베딩 함수
class BGEM3LangChainEmbeddings(Embeddings):
    def __init__(self, unified_embedder):
        self.unified_embedder = unified_embedder
    
    def embed_documents(self, texts):
        """문서들 임베딩"""
        try:
            embeddings = self.unified_embedder.encode_for_vector_db(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ 문서 임베딩 실패: {e}")
            # Fallback: 빈 벡터 반환
            return [[0.0] * 1024 for _ in texts]
    
    def embed_query(self, text):
        """단일 쿼리 임베딩"""
        try:
            embeddings = self.unified_embedder.encode_for_vector_db([text])
            return embeddings[0].tolist()
        except Exception as e:
            print(f"❌ 쿼리 임베딩 실패: {e}")
            # Fallback: 빈 벡터 반환
            return [0.0] * 1024

# ✅ 통합 모델 인스턴스 생성
unified_embedder = UnifiedBGEM3Embedder(EMBEDDING_MODEL_NAME)

# ===== 상태 정의 =====
class GraphState(TypedDict):
    question: str
    original_question: str
    classification: str
    camping_type_preference: str
    context: Annotated[List[Any], operator.add]
    locations: Annotated[List[dict], operator.add]
    final_answer: str
    loop_count: int
    search_attempted: bool
    error_message: str

# ===== 공용 LLM 호출 유틸 =====

def oai_text(prompt: str, model: str = GPT_MODEL) -> Dict[str, Any]:
    """Responses API로 텍스트만 받아오는 헬퍼.
    반환: {"text": str, "request_id": str}
    """
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    # 공식 SDK는 output_text 제공 → 파싱 인덱스 실수 방지
    return {"text": (resp.output_text or "").strip(), "request_id": getattr(resp, "_request_id", None)}

# ===== 임베딩 및 DB 로딩 =====
print("--- ChromaDB 로딩 중 ---")
start_time = time.perf_counter()

# ✅ 변경: 통합 임베딩 함수 사용
embedding_function = BGEM3LangChainEmbeddings(unified_embedder)
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

end_time = time.perf_counter()
print(f"--- 로딩 완료 (소요 시간: {end_time - start_time:.2f}초) ---")

# --- 1차 분류 노드 ---

def classify_question_type(state: GraphState) -> dict:
    t0 = time.perf_counter()
    question = state["question"]
    prompt = (
        "당신은 캠핑에 대한 질문을 하는지 놀러갈 장소를 추천해달라고 하는 질문을 하는지 분류하는 AI 어시스턴트입니다.\n"
        "다음 질문을 읽고 일반 캠핑에 관한 질문이면 '일반 캠핑' 으로 답하고, 놀러가는 장소에 관한 질문이나 문맥상 장소를 추천해야하는 질문은 '장소 추천'으로 답하세요.\n" 
        "'일반 캠핑', '장소 추천' 둘 중 하나로 답변하세요.\n\n"
        f"질문: {question}\n"
        "분류:"
    )
    try:
        out = oai_text(prompt)
        text = out["text"]

        categories = ["일반 캠핑", "장소 추천"]

        # 문장 전체에서 카테고리 단어가 있는지 확인
        classification = next((cat for cat in categories if cat in text), "일반 캠핑")

        print(f"분류: {classification} ({time.perf_counter() - t0:.2f}s)")
        return {"classification": classification, "original_question": question}

    except Exception as e:
        print(f"분류 오류: {e} ({time.perf_counter() - t0:.2f}s)")
        return {"classification": "일반 캠핑", "original_question": question, "error_message": str(e)}
        
# --- 일반 질문 답변 노드 ---

def generate_general_answer(state: GraphState) -> dict:
    t0 = time.perf_counter()
    question = state["question"]
    prompt = (
        "당신은 캠핑 전문가입니다. 다음 캠핑 관련 질문에 답변해주세요.\n\n"
        f"질문: {question}\n"
        "답변:"
    )
    try:
        out = oai_text(prompt)
        answer = out["text"]
        print(f"일반질문 응답 ({time.perf_counter() - t0:.2f}s | req={out['request_id']})")
        return {"final_answer": answer}
    except Exception as e:
        print(f"일반질문 오류: {e} ({time.perf_counter() - t0:.2f}s)")
        return {"final_answer": "답변 생성 중 오류가 발생했습니다.", "error_message": str(e)}

# --- 라우팅 함수 ---

def route_by_classification(state):
    return "general" if state.get("classification") == "일반 캠핑" else "ask_preference"

def route_by_camping_type(state):
    mapping = {
        "유료캠핑장": "paid",
        "글램핑/카라반": "glamping",
        "오지/노지캠핑": "ojee",
    }
    return mapping.get(state.get("camping_type_preference", "유료캠핑장"), "paid")

# --- 캠핑 유형 선택 요청 노드 ---

def ask_camping_preference(state: GraphState) -> dict:
    message = (
        "🏕️ 장소를 추천해드릴게요! 어떤 스타일의 캠핑을 원하시나요?\n\n"
        "1️⃣ 유료캠핑장 (오토캠핑장, 편의시설 완비)  \n"
        "2️⃣ 글램핑/카라반 (럭셔리, 편안한 캠핑)  \n"
        "3️⃣ 오지/노지캠핑 (자연 속 무료 캠핑)  \n\n"
        "인원이나 스타일 등 모두 알려주세요!"
    )
    return {"final_answer": message}

# --- 캠핑 유형 분류 노드 ---
def classify_camping_type(state: GraphState) -> dict:
    t0 = time.perf_counter()
    user_input = state["question"]  # 사용자의 답변
    original_question = state.get("original_question", "")
    
    prompt = (
        "당신은 사용자의 캠핑 유형 선호도를 분류하는 AI 어시스턴트입니다.\n"
        "사용자의 응답과 원래 질문을 읽고 '유료캠핑장', '글램핑/카라반', '오지/노지캠핑' 중 하나로 정확하게 카테고리만을 답변하세요.\n\n"
        f"원래 질문: {original_question}\n"
        f"사용자 응답: {user_input}\n"
        "분류:"
    )
    
    try:
        out = oai_text(prompt)
        text = out["text"]

        categories = ["유료캠핑장", "글램핑/카라반", "오지/노지캠핑"]
        camping_type = next((cat for cat in categories if cat in text), "유료캠핑장")

        print(f"캠핑유형: {camping_type} ({time.perf_counter() - t0:.2f}s)")
        return {"camping_type_preference": camping_type}

    except Exception as e:
        print(f"캠핑유형 분류 오류: {e} ({time.perf_counter() - t0:.2f}s)")
        return {"camping_type_preference": "유료캠핑장", "error_message": str(e)}

# --- ✅ 통합된 RAG + 웹검색 노드 ---
async def search_camping(state: dict, camping_type: str, vectordb) -> dict:
    """개선된 BGE-M3 하이브리드 검색 - 실제 점수 사용"""

    search_query = f"{state.get('original_question', '')} {state.get('question', '')}".strip()
    print(f"\n--- 🔍 {camping_type} 유형으로 통합 BGE-M3 하이브리드 검색 시작 ---")
    print(f"🔹 [DEBUG] 검색 쿼리: {search_query}")

    try:
        # 1단계: Dense Search로 상위 10개 후보 추출
        dense_candidates = await asyncio.to_thread(
            vectordb.similarity_search_with_score,  # ✅ 점수와 함께 반환
            query=search_query,
            k=10,
            filter={"캠핑유형": camping_type},
        )
        
        print(f"🔹 [DEBUG] Dense 후보 {len(dense_candidates)}개 추출")
        if not dense_candidates:
            return {"locations": []}

        # Dense 결과 분리 (document, score)
        dense_docs = [item[0] for item in dense_candidates]  # Document 객체
        dense_raw_scores = [item[1] for item in dense_candidates]  # 실제 유사도 점수

        # 2단계: BGE-M3 Sparse 점수 계산
        try:
            candidate_texts = [doc.page_content for doc in dense_docs]
            
            print("🔹 [DEBUG] BGE-M3 Sparse 임베딩 계산 중...")
            embeddings = unified_embedder.encode_hybrid(search_query, candidate_texts)
            
            query_sparse = embeddings['query_sparse']
            doc_sparse_list = embeddings['doc_sparse']
            
            # ✅ 실제 Sparse 유사도 점수 계산
            sparse_raw_scores = []
            for doc_sparse in doc_sparse_list:
                score = 0.0
                # 공통 토큰들의 가중치 곱의 합
                for token_id, query_weight in query_sparse.items():
                    if token_id in doc_sparse:
                        score += query_weight * doc_sparse[token_id]
                sparse_raw_scores.append(score)
            
            print(f"🔹 [DEBUG] Sparse 원시 점수 범위: {min(sparse_raw_scores):.4f} ~ {max(sparse_raw_scores):.4f}")
            
        except Exception as e:
            print(f"❌ [ERROR] BGE-M3 Sparse 계산 실패: {e}")
            sparse_raw_scores = [0.0] * len(dense_docs)

        # 3단계: 점수 정규화 및 하이브리드 계산
        def normalize_scores(scores):
            """점수를 0~1 범위로 정규화"""
            if not scores or max(scores) == min(scores):
                return [0.5] * len(scores)  # 모든 점수가 같으면 중간값
            
            min_score, max_score = min(scores), max(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]

        # Dense 점수 정규화 (거리 → 유사도로 변환 후)
        # ChromaDB의 cosine distance를 similarity로 변환: similarity = 1 - distance
        dense_similarities = [1 - score for score in dense_raw_scores]
        dense_normalized = normalize_scores(dense_similarities)
        
        # Sparse 점수 정규화
        sparse_normalized = normalize_scores(sparse_raw_scores)
        
        # ✅ 하이브리드 점수 계산
        hybrid_results = []
        for i, (doc, dense_norm, sparse_norm, dense_raw, sparse_raw) in enumerate(
            zip(dense_docs, dense_normalized, sparse_normalized, dense_raw_scores, sparse_raw_scores)
        ):
            # 하이브리드 점수: Dense 60% + Sparse 40%
            final_score = dense_norm * 0.6 + sparse_norm * 0.4
            
            hybrid_results.append((doc, final_score, dense_norm, sparse_norm))
            
            print(f"[DEBUG][Rank {i+1}] Name:{doc.metadata.get('캠핑장이름', '이름없음')} "
                  f"Dense:{dense_norm:.3f}({dense_raw:.3f}) "
                  f"Sparse:{sparse_norm:.3f}({sparse_raw:.4f}) "
                  f"Final:{final_score:.3f}")

        # 4단계: 하이브리드 점수로 재정렬 → 상위 2개 선택
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        top_2_results = hybrid_results[:2]
        
        print(f"🔹 [DEBUG] 최종 상위 2개 선택:")
        for i, (doc, final_score, dense_score, sparse_score) in enumerate(top_2_results):
            print(f"[DEBUG][Final Top {i+1}] Name:{doc.metadata.get('캠핑장이름', '이름없음')} "
                  f"Score:{final_score:.3f}")

        # 5단계: 웹 스니펫 추가 (기존과 동일)
        web_snippets_map = {}
        if camping_type != "오지/노지캠핑" and top_2_results:
            docs_with_metadata = [(doc, score) for doc, score, _, _ in top_2_results]
            
            snippets = await build_snippet_per_doc(
                docs_with_metadata=docs_with_metadata,
                per_type_display=20,
                fetch_timeout=8,
                max_chars=2000,
                enforce_name_in_title=True,
                include_when_no_local_modified=True,
            )
            web_snippets_map = {s['장소이름']: s for s in snippets}

        # 6단계: 최종 결과 반환
        locations_data = []
        unique_names = set()

        for doc, final_score, _, _ in top_2_results:
            meta = getattr(doc, "metadata", {}) or {}
            location_name = meta.get("캠핑장이름", "이름 정보 없음")
            
            if location_name in unique_names:
                continue
            unique_names.add(location_name)

            location_info = {
                "local_document": {
                    "metadata": meta, 
                    "content": doc.page_content, 
                    "score": final_score
                },
                "web_snippet": web_snippets_map.get(location_name)
            }
            locations_data.append(location_info)

        return {"locations": locations_data}

    except Exception as e:
        print(f"❌ [ERROR] search_camping 전체 실패: {e}")
        return {"locations": []}

        
async def search_paid_camping(state):
    return await search_camping(state, "유료캠핑장", vectordb)

async def search_glamping_caravan(state):
    return await search_camping(state, "글램핑/카라반", vectordb)

async def search_ojee_camping(state):
    return await search_camping(state, "오지/노지캠핑", vectordb)

# --- 장소 추천 최종 답변 생성 ---
def generate_location_answer(state: GraphState) -> dict:
    t0 = time.perf_counter()
    
    original_question = state.get("original_question", "")
    second_question = state.get("question", "")
    camping_type = state.get("camping_type_preference", "")
    locations = state.get("locations", [])
    print(f"[DEBUG] 최종 반환될 장소 개수: {len(locations)}개")
    
    context_strs = []
    final_locations = []
    for loc in locations:
        local_meta = loc.get("local_document", {}).get("metadata", {})
        local_content = loc.get("local_document", {}).get("content", "")
        snippet = loc.get("web_snippet", {})
        snippet_text = snippet.get("snippet", "") if snippet else ""
        
        # 올바른 메타데이터 접근 및 final_locations 생성
        location_data = {
            "name": local_meta.get('캠핑장이름'),
            "address": local_meta.get('캠핑장주소'),
            "latitude": local_meta.get('캠핑장 위도'),
            "longitude": local_meta.get('캠핑장 경도'),
            "local_meta": local_meta,
        }
        final_locations.append(location_data)
        
        context_strs.append(
            f"메타데이터: {final_locations}"
            f"본문: {local_content}"
            f"네이버 정보: {snippet_text}"
        )

    context_str = "\n---\n".join(context_strs[:2])

    prompt = (
        f"당신은 캠핑에이전트 챗봇입니다. 아래 문맥을 참고하여 '{camping_type}' 유형에 맞는 장소를 추천해주세요.\n"
        "추천 시에는 반드시 문맥 내 메타데이터와 최신 네이버 정보를 참고하여 답변하세요.\n"
        "사이트나 전화번호를 말해줄때는 \"모든 정보는 최신이 아닐 수 있으니 공식 사이트/전화로 재확인 바랍니다.\"라고 덧붙여주세요.\n\n"
        
        f"첫 번째 질문: {original_question}\n"
        f"두 번째 질문 및 사용자의 캠핑 유형 답변: {second_question}\n\n"
        
        f"문맥:\n{context_str}\n\n"
        
        "위 내용을 고려하여 답변을 작성해 주세요.\n"
        "답변:"
    )
    try:
        out = oai_text(prompt)
        answer = out["text"]
        print(f"✅ 장소 추천 답변 생성 완료 ({time.perf_counter() - t0:.2f}s | req={out['request_id']}) ")
        # 프론트엔드로 전달할 최종 locations 포맷으로 반환
        return {"final_answer": answer, "locations": final_locations}
    except Exception as e:
        print(f"❌ 답변 생성 오류: {e} ({time.perf_counter() - t0:.2f}s)")
        return {"final_answer": "답변 생성 중 오류가 발생했습니다.", "error_message": str(e), "locations": final_locations}


# --- 메인 워크플로우 ---
workflow = StateGraph(GraphState)
workflow.add_node("classify_type", classify_question_type)
workflow.add_node("generate_general", generate_general_answer)
workflow.add_node("ask_preference", ask_camping_preference)
workflow.set_entry_point("classify_type")
workflow.add_conditional_edges(
    "classify_type",
    route_by_classification,
    {
        "general": "generate_general",
        "ask_preference": "ask_preference",
    },
)
workflow.add_edge("generate_general", END)
workflow.add_edge("ask_preference", END)

# --- 캠핑 유형 후속 워크플로우 ---
continuation = StateGraph(GraphState)
continuation.add_node("classify_camping", classify_camping_type)
continuation.add_node("search_paid", search_paid_camping)
continuation.add_node("search_glamping", search_glamping_caravan)
continuation.add_node("search_ojee", search_ojee_camping)
continuation.add_node("generate_location", generate_location_answer)
continuation.set_entry_point("classify_camping")
continuation.add_conditional_edges(
    "classify_camping",
    route_by_camping_type,
    {
        "paid": "search_paid",
        "glamping": "search_glamping",
        "ojee": "search_ojee",
    },
)
continuation.add_edge("search_paid", "generate_location")
continuation.add_edge("search_glamping", "generate_location")
continuation.add_edge("search_ojee", "generate_location")
continuation.add_edge("generate_location", END)

main_app = workflow.compile()
continuation_app = continuation.compile()

# --- 메인 실행 함수 ---

def main():
    print("🏕️ 캠핑 챗봇에 오신 것을 환영합니다! '종료'를 입력하면 종료됩니다.")
    waiting_for_camping_choice = False
    original_q_cache: str | None = None

    while True:
        user_input = input("\n❓ 사용자 질문: ").strip()
        if user_input.lower() in ["종료", "quit", "exit"]:
            print("👋 챗봇을 종료합니다.")
            break

        try:
            if waiting_for_camping_choice:
                # 캠핑 유형 입력 대기 중
                state: GraphState = {
                    "question": user_input,
                    "original_question": original_q_cache or user_input,
                    "context": [],
                    "search_attempted": False,
                    "loop_count": 0,
                }
                result = asyncio.run(continuation_app.ainvoke(state))
                waiting_for_camping_choice = False
                original_q_cache = None
            else:
                # 새로운 질문 처리
                state: GraphState = {
                    "question": user_input,
                    "context": [],
                    "search_attempted": False,
                    "loop_count": 0,
                }
                result = asyncio.run(main_app.ainvoke(state))

                # 다음 입력으로 유형을 받도록 전환
                if "어떤 스타일의 캠핑을 원하시나요?" in result.get("final_answer", ""):
                    waiting_for_camping_choice = True
                    original_q_cache = state["question"]

            print(f"\n📝 답변: {result.get('final_answer')}\n")

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            waiting_for_camping_choice = False
            original_q_cache = None

if __name__ == "__main__":
    main()