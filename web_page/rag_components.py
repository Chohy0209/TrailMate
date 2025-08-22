# rag_components.py
import asyncio
import json
import operator
import time
from typing import Annotated, Any, Dict, List, TypedDict

from FlagEmbedding import BGEM3FlagModel
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

import config
from naver_api import build_snippet_per_doc
from services import client, driver

# --- 1. BGE-M3 Embedder Class ---

class UnifiedBGEM3Embedder:
    """BGE-M3 임베딩 모델을 비동기 환경에서 안전하게 관리하는 클래스"""
    def __init__(self):
        self.model = None
        self.model_name = None
        self.lock = asyncio.Lock()  # 동시 접근 제어를 위한 Lock

    def _load_model_sync(self, model_name: str):
        """동기적으로 모델을 로드하는 내부 함수"""
        self.model_name = model_name
        try:
            self.model = BGEM3FlagModel(self.model_name, use_fp16=True)
            print(f"✅ BGE-M3 통합 모델 로드 완료: {self.model_name}")
        except Exception as e:
            print(f"❌ BGE-M3 '{self.model_name}' 모델 로드 실패: {e}")
            try:
                self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
                print("✅ BGE-M3 원본 모델로 Fallback 로드 완료")
            except Exception as e2:
                print(f"❌ BGE-M3 Fallback도 실패: {e2}")
                self.model = None

    @classmethod
    async def create(cls, model_name: str = config.EMBEDDING_MODEL_NAME):
        """비동기적으로 클래스 인스턴스를 생성하고 모델을 로드"""
        embedder = cls()
        await asyncio.to_thread(embedder._load_model_sync, model_name)
        return embedder

    async def encode_for_vector_db(self, texts: List[str], task_id: str) -> List[List[float]]:
        """Neo4j용 dense embedding을 생성 (비동기 Lock 포함)"""
        if self.model is None:
            raise Exception("BGE-M3 모델이 로드되지 않았습니다.")

        def _encode_sync():
            embeddings = self.model.encode(
                texts, batch_size=32, return_dense=True,
                return_sparse=False, return_colbert_vecs=False
            )
            return [arr.tolist() for arr in embeddings['dense_vecs']]

        print(f"  [TASK: {task_id}] 🟡 임베딩 Lock 대기 시작...")
        async with self.lock:
            print(f"  [TASK: {task_id}] 🟢 임베딩 Lock 확보! 인코딩 작업 시작...")
            start_time = time.time()
            result = await asyncio.to_thread(_encode_sync)
            duration = time.time() - start_time
            print(f"  [TASK: {task_id}] ✅ 임베딩 완료! ({duration:.2f}초)")
            return result


# --- 2. LangGraph State Definition ---

class GraphState(TypedDict):
    """LangGraph의 상태를 정의하는 클래스"""
    question: str
    original_question: str
    classification: str
    camping_type_preference: str
    context: Annotated[List[Any], operator.add]
    locations: Annotated[List[dict], operator.add]
    final_answer: str
    error_message: str
    unified_embedder: UnifiedBGEM3Embedder # Embedder를 상태에 포함


# --- 3. LangGraph Node Functions ---

async def oai_text(prompt: str) -> str:
    """OpenAI API를 호출하여 텍스트 응답을 받는 유틸리티 함수"""
    resp = await client.chat.completions.create(
        model=config.GPT_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return (resp.choices[0].message.content or "").strip()

async def classify_question_type(state: GraphState) -> dict:
    """질문의 유형을 '일반 캠핑'과 '장소 추천'으로 분류하는 노드"""
    prompt = (
        "당신은 캠핑에 대한 질문을 하는지 놀러갈 장소를 추천해달라고 하는 질문을 하는지 분류하는 AI 어시스턴트입니다.\n"
        "다음 질문을 읽고 일반 캠핑에 관한 질문이면 '일반 캠핑' 으로 답하고, 놀러가는 장소에 관한 질문이나 문맥상 장소를 추천해야하는 질문은 '장소 추천'으로 답하세요.\n"
        "'일반 캠핑', '장소 추천' 둘 중 하나로 답변하세요.\n\n"
        f"질문: {state['question']}\n분류:"
    )
    try:
        text = await oai_text(prompt)
        classification = "장소 추천" if "장소 추천" in text else "일반 캠핑"
        return {"classification": classification, "original_question": state["question"]}
    except Exception as e:
        print(f"분류 오류: {e}")
        return {"classification": "일반 캠핑", "original_question": state["question"], "error_message": str(e)}

async def generate_general_answer(state: GraphState) -> dict:
    """일반 캠핑 질문에 대해 답변을 생성하는 노드"""
    prompt = (
        "당신은 캠핑 전문가입니다. 다음 캠핑 관련 질문에 답변해주세요.\n\n"
        f"질문: {state['question']}\n줄 바꿈을 이용하여 가독성 좋게 답변 해 주세요.\n답변:"
    )
    try:
        answer = await oai_text(prompt)
        return {"final_answer": answer}
    except Exception as e:
        print(f"일반질문 오류: {e}")
        return {"final_answer": "답변 생성 중 오류가 발생했습니다.", "error_message": str(e)}

async def ask_camping_preference(state: GraphState) -> dict:
    """사용자에게 선호하는 캠핑 유형을 묻는 노드"""
    message = (
        "🏕️ 장소를 추천해드릴게요! 어떤 스타일의 캠핑을 원하시나요?\n\n"
        "▶ 유료캠핑장 (오토캠핑장, 편의시설 완비)\n"
        "▶ 글램핑/카라반 (럭셔리, 편안한 캠핑)\n"
        "▶ 오지/노지캠핑 (자연 속 무료 캠핑)\n\n"
        "인원이나 스타일 등 모두 알려주세요!"
    )
    return {"final_answer": message}

async def classify_camping_type(state: GraphState) -> dict:
    """사용자의 답변을 바탕으로 캠핑 유형을 분류하는 노드"""
    prompt = (
        "당신은 사용자의 캠핑 유형 선호도를 분류하는 AI 어시스턴트입니다.\n"
        "사용자의 응답과 원래 질문을 읽고 '유료캠핑장', '글램핑/카라반', '오지/노지캠핑' 중 하나로 정확하게 카테고리만을 답변하세요.\n\n"
        f"원래 질문: {state.get('original_question', '')}\n"
        f"사용자 응답: {state['question']}\n분류:"
    )
    try:
        text = await oai_text(prompt)
        categories = ["유료캠핑장", "글램핑/카라반", "오지/노지캠핑"]
        camping_type = next((cat for cat in categories if cat in text), "유료캠핑장")
        return {"camping_type_preference": camping_type}
    except Exception as e:
        print(f"캠핑유형 분류 오류: {e}")
        return {"camping_type_preference": "유료캠핑장", "error_message": str(e)}

def _rollup_query():
    """GraphRAG 검색을 위한 Cypher 쿼리"""
    return """
    CALL {
      CALL db.index.vector.queryNodes('camp_embedding_index', $topk_camp, $q) YIELD node, score
      WITH node AS camp, score * $w_camp AS s, 'camp' AS src
      WHERE camp.type = $camping_type
      RETURN camp, s, src
      UNION
      CALL db.index.vector.queryNodes('attr_embedding_index', $topk_attr, $q) YIELD node, score
      MATCH (camp:Camp)-[:HAS_ATTRIBUTE]->(node)
      WITH DISTINCT camp, score * $w_attr AS s, 'attr' AS src
      WHERE camp.type = $camping_type
      RETURN camp, s, src
      UNION
      CALL db.index.vector.queryNodes('summary_embedding_index', $topk_sum, $q) YIELD node, score
      MATCH (camp:Camp)-[:HAS_SUMMARY]->(node)
      WITH DISTINCT camp, score * $w_sum AS s, 'summary' AS src
      WHERE camp.type = $camping_type
      RETURN camp, s, src
    }
    WITH camp, collect({src:src, score:s}) AS parts, reduce(total=0.0, x IN collect(s) | total + x) AS totalScore
    ORDER BY totalScore DESC
    LIMIT $rollup_limit
    OPTIONAL MATCH (camp)-[ra:HAS_ATTRIBUTE]->(a:Attribute)
    WITH camp, parts, totalScore, collect({type: ra.type, text: a.text}) AS attributes
    OPTIONAL MATCH (camp)-[:HAS_SUMMARY]->(s:Summary)
    WITH camp, parts, totalScore, attributes, collect(s.text) AS summaries
    RETURN camp, parts, totalScore, attributes, summaries
    ORDER BY totalScore DESC
    """

def _camp_to_location_dict(record):
    """Neo4j 검색 결과를 LangChain Document 형식으로 변환하는 함수"""
    camp_node = record["camp"]
    score = float(record.get("totalScore", 0.0))
    camp_props = dict(camp_node)
    
    try:
        camp_meta = json.loads(camp_props.get("meta", "{}"))
        lat = camp_meta.get("캠핑장 위도")
        lon = camp_meta.get("캠핑장 경도")
    except (json.JSONDecodeError, AttributeError):
        lat, lon = None, None

    meta_for_doc = {
        "캠핑장이름": camp_props.get("name", ""), "운영상태": camp_props.get("status"),
        "캠핑장주소": camp_props.get("address"), "캠핑유형": camp_props.get("type"),
        "캠핑장시설": camp_props.get("facilities"), "즐길거리": camp_props.get("activities"),
        "캠핑장 위도": lat, "캠핑장 경도": lon
    }
    
    combined_context = f"## CAMP META\n{json.dumps(meta_for_doc, ensure_ascii=False)}"
    doc = Document(page_content=combined_context, metadata=meta_for_doc)

    return {
        "local_document": {"metadata": meta_for_doc, "content": combined_context, "score": score},
        "doc_for_web": doc, "web_snippet": None
    }

async def search_camping(state: dict, camping_type: str) -> dict:
    """GraphRAG 검색을 수행하는 메인 노드"""
    search_query = (f"사용자 원래 질문: {state.get('original_question', '')}\n"
                    f"사용자 캠핑 유형 답변: {state.get('question', '')}").strip()
    task_id = state.get('original_question', f'task_{int(time.time())}')
    print(f"\n[TASK: {task_id}] ➡️ GraphRAG 검색 시작 (필터: {camping_type})")
    
    try:
        # 상태(state)에서 embedder를 가져와 사용
        unified_embedder = state["unified_embedder"]
        query_vecs = await unified_embedder.encode_for_vector_db([search_query], task_id=task_id)
        query_vec = query_vecs[0]

        async with driver.session() as session:
            records = await session.run(
                _rollup_query(), q=query_vec, camping_type=camping_type,
                topk_camp=config.TOPK_CAMP, topk_attr=config.TOPK_ATTR, topk_sum=config.TOPK_SUM,
                w_camp=config.WEIGHT_CAMP, w_attr=config.WEIGHT_ATTR, w_sum=config.WEIGHT_SUM,
                rollup_limit=config.ROLLUP_LIMIT
            )
            rows = [record async for record in records]

        if not rows:
            return {"locations": []}

        locations = [_camp_to_location_dict(rec) for rec in rows[:config.FINAL_TOPN]]
        
        if camping_type != "오지/노지캠핑" and locations:
            docs_with_metadata = [(loc["doc_for_web"], float(loc["local_document"]["score"])) for loc in locations]
            try:
                snippets = await build_snippet_per_doc(docs_with_metadata=docs_with_metadata, per_type_display=20, fetch_timeout=8, max_chars=2000)
                snippet_map = {s["장소이름"]: s for s in snippets}
                for loc in locations:
                    place_name = loc["local_document"]["metadata"].get("캠핑장이름", "")
                    if place_name in snippet_map:
                        loc["web_snippet"] = snippet_map[place_name]
            except Exception as e:
                print(f"[웹 스니펫 오류] {e}")

        for loc in locations:
            loc.pop("doc_for_web", None)

        return {"locations": locations}
    except Exception as e:
        print(f"❌ [TASK: {task_id}] Neo4j GraphRAG 검색 실패: {e}")
        return {"locations": []}

async def search_paid_camping(state): return await search_camping(state, "유료캠핑장")
async def search_glamping_caravan(state): return await search_camping(state, "글램핑/카라반")
async def search_ojee_camping(state): return await search_camping(state, "오지/노지캠핑")

async def generate_location_answer(state: GraphState) -> dict:
    """검색된 장소 정보를 바탕으로 최종 추천 답변을 생성하는 노드"""
    locations = state.get("locations", [])
    context_strs = []
    final_locations = []
    
    for loc in locations:
        local_meta = loc.get("local_document", {}).get("metadata", {})
        snippet = loc.get("web_snippet", {})
        
        location_data = {
            "name": local_meta.get('캠핑장이름'), "address": local_meta.get('캠핑장주소'),
            "latitude": local_meta.get('캠핑장 위도'), "longitude": local_meta.get('캠핑장 경도'),
            "local_meta": local_meta,
        }
        final_locations.append(location_data)
        
        context_strs.append(f"메타데이터: {json.dumps(location_data, ensure_ascii=False)}\n본문: {loc.get('local_document', {}).get('content', '')}\n네이버 정보: {snippet.get('snippet', '') if snippet else ''}")

    prompt = (
        f"당신은 캠핑에이전트 챗봇입니다. 아래 문맥을 참고하여 '{state.get('camping_type_preference', '')}' 유형에 맞는 장소를 추천해주세요. 최대한 친절하게 설명하세요.\n"
        "추천 시에는 반드시 문맥 내 메타데이터와 최신 네이버 정보를 참고하여 답변하세요.\n"
        "사이트나 전화번호를 말해줄때는 \"모든 정보는 최신이 아닐 수 있으니 공식 사이트/전화로 재확인 바랍니다.\"라고 덧붙여주세요.\n\n"
        f"첫 번째 질문: {state.get('original_question', '')}\n"
        f"두 번째 질문 및 사용자의 캠핑 유형 답변: {state.get('question', '')}\n\n"
        f"문맥:\n{'---'.join(context_strs)}\n\n"
        "위 내용을 고려하여 줄바꿈을 이용하여 가독성 좋게 답변을 작성해 주세요.\n답변:"
    )
    try:
        answer = await oai_text(prompt)
        return {"final_answer": answer, "locations": final_locations}
    except Exception as e:
        print(f"❌ 답변 생성 오류: {e}")
        return {"final_answer": "답변 생성 중 오류가 발생했습니다.", "error_message": str(e), "locations": final_locations}


# --- 4. LangGraph Workflow Builder ---

def build_workflows():
    """두 개의 LangGraph 워크플로우(메인, 후속)를 빌드하고 반환합니다."""
    
    # 1. 메인 워크플로우: 질문 유형 분류 및 첫 응답
    workflow = StateGraph(GraphState)
    workflow.add_node("classify_type", classify_question_type)
    workflow.add_node("generate_general", generate_general_answer)
    workflow.add_node("ask_preference", ask_camping_preference)
    workflow.set_entry_point("classify_type")
    workflow.add_conditional_edges(
        "classify_type",
        lambda state: "general" if state.get("classification") == "일반 캠핑" else "ask_preference",
        {"general": "generate_general", "ask_preference": "ask_preference"}
    )
    workflow.add_edge("generate_general", END)
    workflow.add_edge("ask_preference", END)
    main_app = workflow.compile()

    # 2. 후속 워크플로우: 캠핑 유형에 따른 장소 검색 및 추천
    continuation = StateGraph(GraphState)
    continuation.add_node("classify_camping", classify_camping_type)
    continuation.add_node("search_paid", search_paid_camping)
    continuation.add_node("search_glamping", search_glamping_caravan)
    continuation.add_node("search_ojee", search_ojee_camping)
    continuation.add_node("generate_location", generate_location_answer)
    continuation.set_entry_point("classify_camping")
    continuation.add_conditional_edges(
        "classify_camping",
        lambda state: {"유료캠핑장": "paid", "글램핑/카라반": "glamping", "오지/노지캠핑": "ojee"}.get(state.get("camping_type_preference", "유료캠핑장")),
        {"paid": "search_paid", "glamping": "search_glamping", "ojee": "search_ojee"}
    )
    continuation.add_edge("search_paid", "generate_location")
    continuation.add_edge("search_glamping", "generate_location")
    continuation.add_edge("search_ojee", "generate_location")
    continuation.add_edge("generate_location", END)
    continuation_app = continuation.compile()

    return main_app, continuation_app