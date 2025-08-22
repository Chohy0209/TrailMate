import os
import time
import asyncio
import random
import operator
from typing import TypedDict, Annotated, List, Any, Dict

#naver api
from naver_api import build_snippet_per_doc, format_snippets_as_text

# OpenAI SDK
from openai import AsyncOpenAI

# LangChain / Vector store
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from dotenv import load_dotenv
import numpy as np
from neo4j import AsyncGraphDatabase

# BGE-M3 통합 사용
from FlagEmbedding import BGEM3FlagModel
import json

load_dotenv()

  
#Neo4j 드라이버 (환경변수에서 가져오기 권장)
uri = os.getenv("NEO4J_URI")
driver = AsyncGraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))



# --- 가중치 & 파라미터 설정 ---
TOPK_CAMP = 100       # Camp에서 초기 후보 수
TOPK_ATTR = 100        # Attribute에서 초기 후보 수
TOPK_SUM = 100       # Summary에서 초기 후보 수
ROLLUP_LIMIT = 2     # Camp 집계 후 상위 몇 개로 줄일지
FINAL_TOPN = 2        # 최종 추천 개수
WEIGHT_CAMP = 3.0
WEIGHT_ATTR = 0.3
WEIGHT_SUM = 0.3

VECTOR_DIM = 1024     # BGE-M3 Dense 차원
SIM_FUNC = "cosine"   # 'cosine' 추천


# ===== 환경 설정 =====
# ⚠️ 키는 환경변수로 읽습니다: export OPENAI_API_KEY="sk-..."
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,
)

EMBEDDING_MODEL_NAME = "dragonkue/BGE-m3-ko"
MAX_LOOPS = 2

GPT_MODEL = "ft:gpt-4.1-mini-2025-04-14:ailab:camping-rag-qa:C2bHhwJM:ckpt-step-714"

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

    # --- 이 부분을 추가해야 합니다 ---
    def encode_for_vector_db(self, texts: List[str]):
        """Neo4j용 dense embedding 생성"""
        if self.model is None:
            raise Exception("BGE-M3 모델이 로드되지 않았습니다.")
        
        # BGE-M3 모델의 encode 메소드를 사용하여 dense 벡터를 생성
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        
        return embeddings['dense_vecs']
  

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

#===== 공용 LLM 호출 유틸 =====

async def oai_text(prompt: str, model: str = GPT_MODEL) -> Dict[str, Any]:
    """Responses API로 텍스트만 받아오는 헬퍼.
    반환: {"text": str, "request_id": str}
    """
    resp = await client.responses.create(
        model=model,
        input=prompt,
    )
    # 공식 SDK는 output_text 제공 → 파싱 인덱스 실수 방지
    return {"text": (resp.output_text or "").strip(), "request_id": getattr(resp, "_request_id", None)}

# --- 1차 분류 노드 ---

async def classify_question_type(state: GraphState) -> dict:
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
        out = await oai_text(prompt)
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

async def generate_general_answer(state: GraphState) -> dict:
    t0 = time.perf_counter()
    question = state["question"]
    prompt = (
        "당신은 캠핑 전문가입니다. 다음 캠핑 관련 질문에 답변해주세요.\n\n"
        f"질문: {question}\n"
        "줄 바꿈을 이용하여 가독성 좋게 답변 해 주세요."
        "답변:"
    )
    try:
        out = await oai_text(prompt)
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

async def ask_camping_preference(state: GraphState) -> dict:
    message = (
        "🏕️ 장소를 추천해드릴게요! 어떤 스타일의 캠핑을 원하시나요?\n\n"
        "▶ 유료캠핑장 (오토캠핑장, 편의시설 완비)  \n"
        "▶ 글램핑/카라반 (럭셔리, 편안한 캠핑)  \n"
        "▶ 오지/노지캠핑 (자연 속 무료 캠핑)  \n\n"
        "인원이나 스타일 등 모두 알려주세요!"
    )
    return {"final_answer": message}

# --- 캠핑 유형 분류 노드 ---
async def classify_camping_type(state: GraphState) -> dict:
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
        out = await oai_text(prompt)
        text = out["text"]

        categories = ["유료캠핑장", "글램핑/카라반", "오지/노지캠핑"]
        camping_type = next((cat for cat in categories if cat in text), "유료캠핑장")

        print(f"캠핑유형: {camping_type} ({time.perf_counter() - t0:.2f}s)")
        return {"camping_type_preference": camping_type}

    except Exception as e:
        print(f"캠핑유형 분류 오류: {e} ({time.perf_counter() - t0:.2f}s)")
        return {"camping_type_preference": "유료캠핑장", "error_message": str(e)}



async def ensure_vector_indexes(session):
    """Camp/Attribute/Summary 벡터 인덱스를 보장 (이미 있으면 무시)."""
    # Neo4j 5.x: IF NOT EXISTS 지원. 미지원 버전이면 try/except로 무시.
    stmts = [
        f"""
        CREATE VECTOR INDEX camp_embedding_index IF NOT EXISTS
        FOR (c:Camp) ON (c.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {VECTOR_DIM},
            `vector.similarity_function`: '{SIM_FUNC}'
          }}
        }};
        """,
         f"""
        CREATE VECTOR INDEX attr_embedding_index IF NOT EXISTS
        FOR (a:Attribute) ON (a.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {VECTOR_DIM},
            `vector.similarity_function`: '{SIM_FUNC}'
          }}
        }};
        """,
         f"""
        CREATE VECTOR INDEX summary_embedding_index IF NOT EXISTS
        FOR (s:Summary) ON (s.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {VECTOR_DIM},
            `vector.similarity_function`: '{SIM_FUNC}'
          }}
        }};
        """
    ]
    for stmt in stmts:
        try:
            await session.run(stmt)
        except Exception as _:
            # 이미 존재하거나 버전 이슈면 조용히 통과
            pass
def _rollup_query():
    """
    Camp/Attribute/Summary 각각의 벡터 검색 결과를 Camp로 올려 집계하는 Cypher.
    - camp.type = $camping_type 필터
    - score는 소스별 가중치를 곱해 합산
    - 상위 $rollup_limit까지 반환
    """
    return """
    // 1) Source별 벡터 검색
    CALL {
      // Camp 직접 검색
      CALL db.index.vector.queryNodes('camp_embedding_index', $topk_camp, $q) YIELD node, score
      WITH node AS camp, score * $w_camp AS s, 'camp' AS src
      WHERE camp.type = $camping_type
      RETURN camp, s, src
      UNION
      // Attribute에서 검색 후 부모 Camp로 승격
      CALL db.index.vector.queryNodes('attr_embedding_index', $topk_attr, $q) YIELD node, score
      MATCH (camp:Camp)-[:HAS_ATTRIBUTE]->(node)
      WITH DISTINCT camp, score * $w_attr AS s, 'attr' AS src
      WHERE camp.type = $camping_type
      RETURN camp, s, src
      UNION
      // Summary에서 검색 후 부모 Camp로 승격
      CALL db.index.vector.queryNodes('summary_embedding_index', $topk_sum, $q) YIELD node, score
      MATCH (camp:Camp)-[:HAS_SUMMARY]->(node)
      WITH DISTINCT camp, score * $w_sum AS s, 'summary' AS src
      WHERE camp.type = $camping_type
      RETURN camp, s, src
    } 

    // 2) Camp 단위로 점수 집계 (WHERE 필터는 각 UNION 내부로 이동했으므로 제거)
    WITH camp, collect({src:src, score:s}) AS parts, reduce(total=0.0, x IN collect(s) | total + x) AS totalScore
    ORDER BY totalScore DESC
    LIMIT $rollup_limit

    // 3) 관련 Attribute / Summary 긁어오기
    OPTIONAL MATCH (camp)-[ra:HAS_ATTRIBUTE]->(a:Attribute)
    WITH camp, parts, totalScore, collect({type: ra.type, text: a.text}) AS attributes
    OPTIONAL MATCH (camp)-[:HAS_SUMMARY]->(s:Summary)
    WITH camp, parts, totalScore, attributes, collect(s.text) AS summaries

    RETURN camp, parts, totalScore, attributes, summaries
    ORDER BY totalScore DESC
    """

def _camp_to_location_dict(record):
    """Cypher 반환 레코드를 locations[] 요소로 변환."""
    camp_node = record["camp"]
    # attributes = record.get("attributes", []) or []  # 삭제
    # summaries = record.get("summaries", []) or []  # 삭제
    score = float(record.get("totalScore", 0.0))
    parts = record.get("parts", []) or []

    # === 로깅 추가 ===
    camp_name = camp_node.get("name", "Unknown")
    print(f"캠핑장: {camp_name}")
    print(f"  총점: {score:.4f}")

    # parts 전체 구조 확인
    print(f"  parts 개수: {len(parts)}")
    for i, part in enumerate(parts):
        src = part.get("src", "unknown")
        part_score = part.get("score", 0.0)
        print(f"  - [{i}] {src}: {part_score:.4f}")

    # camp 소스가 없는지 명시적 체크
    camp_sources = [p for p in parts if p.get("src") == "camp"]
    if not camp_sources:
        print(f"  ⚠️  이 캠핑장은 camp 직접 검색에서는 발견되지 않음")
    print()  # 구분용 빈 줄

    # camp_node는 Node 객체. 프로퍼티 꺼내기
    camp_props = dict(camp_node)

    # meta가 JSON 문자열로 저장되어 있다면 그대로 content로 사용
    raw_meta = camp_props.get("meta")
    content_str = ""
    if isinstance(raw_meta, str) and raw_meta:
        content_str = raw_meta
    else:
        # 안전망: 핵심 프로퍼티 텍스트로 구성
        content_str = json.dumps({
            "캠핑장이름": camp_props.get("name"),
            "운영상태": camp_props.get("status"),
            "캠핑장주소": camp_props.get("address"),
            "캠핑유형": camp_props.get("type"),
            "캠핑장시설": camp_props.get("facilities"),
            "즐길거리": camp_props.get("activities")
        }, ensure_ascii=False)

    # 문맥 강화용: Attribute / Summary 삭제
    # attr_texts = [f"[{a.get('type','')}] {a.get('text','')}" for a in attributes if a.get("text")]
    # sum_texts = [s for s in summaries if s]

    # combined_context에서 attr과 summary 관련 내용 모두 제거
    combined_context = "\n".join([
        "## CAMP META",
        content_str,
        # "## ATTRIBUTES",
        # "\n".join(attr_texts[:20]),
        # "## SUMMARIES",
        # "\n".join(sum_texts[:20]),
    ])

    camp_meta = json.loads(camp_props.get("meta"))
    
    lat = camp_meta["캠핑장 위도"] # 위도
    lon = camp_meta["캠핑장 경도"]

    # LangChain Document로 감싸 네이버 함수와 호환
    # 메타데이터 키는 기존 Chroma 버전과 동일하게 맞춰줌
    meta_for_doc = {
        "캠핑장이름": camp_props.get("name", ""),
        "운영상태": camp_props.get("status"),
        "캠핑장주소": camp_props.get("address"),
        "캠핑유형": camp_props.get("type"),
        "캠핑장시설": camp_props.get("facilities"),
        "즐길거리": camp_props.get("activities"),
        "캠핑장 위도": lat,
        "캠핑장 경도": lon
    }
    doc = Document(page_content=combined_context, metadata=meta_for_doc)

    # locations[] 표준 형태
    location_info = {
        "local_document": {
            "metadata": meta_for_doc,
            "content": combined_context,
            "score": score
        },
        "doc_for_web": doc,     # 네이버 스니펫 함수에 넘기기 위한 원본
        "web_snippet": None     # 나중에 조건부로 채움
    }
    return location_info


async def search_camping(state: dict, camping_type: str) -> dict:
    """
    Neo4j 기반 GraphRAG 검색:
    - Camp/Attribute/Summary에서 각각 벡터 검색 → Camp로 승격/집계
    - Camp의 연결(Attributes, Summaries)까지 모두 실어 문맥 구성
    - camping_type이 '오지/노지캠핑'이면 RAG만 사용, 그 외 유형이면 웹 스니펫 병합
    """
    search_query = (
    f"사용자 원래 질문: {state.get('original_question', '')}\n"
    f"사용자 캠핑 유형 답변: {state.get('question', '')}").strip()
    print(search_query)
    print(f"\n--- 🔍 Neo4j GraphRAG (필터: {camping_type}) ---")

    try:
        # 0) 쿼리 임베딩
        dense_vecs = await asyncio.to_thread(unified_embedder.encode_for_vector_db, [search_query])
        query_vec = dense_vecs[0].tolist()

        async with driver.session() as session:
            await ensure_vector_indexes(session)  # 인덱스 보장

            # 1) Roll-up 검색 실행
            records = await session.run(
                _rollup_query(),
                q=query_vec,
                camping_type=camping_type,
                topk_camp=TOPK_CAMP,
                topk_attr=TOPK_ATTR,
                topk_sum=TOPK_SUM,
                w_camp=WEIGHT_CAMP,
                w_attr=WEIGHT_ATTR,
                w_sum=WEIGHT_SUM,
                rollup_limit=ROLLUP_LIMIT,
            )
            # 결과를 비동기적으로 가져옴
            rows = [record async for record in records]

        if not rows:
            print("검색 결과 없음")
            return {"locations": []}

        print(f"📄 Neo4j에서 Camp 후보 {len(rows)}개 집계됨")

        # 2) Python 측에서 최종 TOP N 고르고 locations 구조 만들기
        locations = []
        for rec in rows[:FINAL_TOPN]:
            locations.append(_camp_to_location_dict(rec))

        # 3) 캠핑유형에 따라 웹 스니펫 호출 여부 결정
        if camping_type != "오지/노지캠핑" and locations:
            # 유료/글램핑일 때만 네이버 API 호출
            docs_with_metadata = []
            for loc in locations:
                doc = loc["doc_for_web"]
                score = float(loc["local_document"]["score"])
                docs_with_metadata.append((doc, score))

            try:
                snippets = await build_snippet_per_doc(
                    docs_with_metadata=docs_with_metadata,
                    per_type_display=20,
                    fetch_timeout=8,
                    max_chars=2000,
                    enforce_name_in_title=True,
                    include_when_no_local_modified=True,
                )
                snippet_map = {s["장소이름"]: s for s in snippets}
                for loc in locations:
                    place_name = loc["local_document"]["metadata"].get("캠핑장이름", "")
                    if place_name in snippet_map:
                        loc["web_snippet"] = snippet_map[place_name]
            except Exception as e:
                print(f"[웹 스니펫 오류] {e}")

        # 4) 내부용 키 제거
        for loc in locations:
            loc.pop("doc_for_web", None)

        return {"locations": locations}

    except Exception as e:
        print(f"❌ Neo4j GraphRAG 검색 실패: {e}")
        return {"locations": []}


# --- 래퍼: 유형별 검색 ---
async def search_paid_camping(state):
    return await search_camping(state, "유료캠핑장")

async def search_glamping_caravan(state):
    return await search_camping(state, "글램핑/카라반")

async def search_ojee_camping(state):
    return await search_camping(state, "오지/노지캠핑")


# --- 장소 추천 최종 답변 생성 ---
async def generate_location_answer(state: GraphState) -> dict:
    t0 = time.perf_counter()    
    original_question = state.get("original_question", "")
    second_question = state.get("question", "")
    camping_type = state.get("camping_type_preference", "")
    context = state.get("context", [])
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

    context_str = "\n---".join(context_strs[:2])  # 최대 2개까지만

    prompt = (
        f"당신은 캠핑에이전트 챗봇입니다. 아래 문맥을 참고하여 '{camping_type}' 유형에 맞는 장소를 추천해주세요. 최대한 친절하게 설명하세요.\n"
        "추천 시에는 반드시 문맥 내 메타데이터와 최신 네이버 정보를 참고하여 답변하세요.\n"
        "사이트나 전화번호를 말해줄때는 \"모든 정보는 최신이 아닐 수 있으니 공식 사이트/전화로 재확인 바랍니다.\"라고 덧붙여주세요.\n\n"
        
        f"첫 번째 질문: {original_question}\n"
        f"두 번째 질문 및 사용자의 캠핑 유형 답변: {second_question}\n\n"
        
        f"문맥:\n{context_str}\n\n"
        
        "위 내용을 고려하여 줄바꿈을 이용하여 가독성 좋게 답변을 작성해 주세요.\n"
        "답변:"
    )
    try:
        out = await oai_text(prompt)
        answer = out["text"]
        print(f"✅ 장소 추천 답변 생성 완료 ({time.perf_counter() - t0:.2f}s | req={out['request_id']})")
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

async def main():
    print("🏕️ 캠핑 챗봇에 오신 것을 환영합니다! '종료'를 입력하면 종료됩니다.")
    waiting_for_camping_choice = False
    original_q_cache: str | None = None

    while True:
        user_input = await asyncio.to_thread(input, "\n❓ 사용자 질문: ")
        user_input = user_input.strip()
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
                result = await continuation_app.ainvoke(state)
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
                result = await main_app.ainvoke(state)

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
    asyncio.run(main())
