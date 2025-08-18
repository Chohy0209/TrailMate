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
from dotenv import load_dotenv

load_dotenv()

# ===== 환경 설정 =====
# ⚠️ 키는 환경변수로 읽습니다: export OPENAI_API_KEY="sk-..."
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,
)

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CHROMA_DB_PATH = "./data/camp_vectorDB"
MAX_LOOPS = 2

GPT_MODEL = "ft:gpt-4.1-2025-04-14:ailab::C2eTINXG:ckpt-step-357"

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
print("--- 임베딩 모델 및 ChromaDB 로딩 중 ---")
start_time = time.perf_counter()
embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory = CHROMA_DB_PATH, embedding_function = embeddings)
end_time = time.perf_counter()
print(f"--- 로딩 완료 (소요 시간: {end_time - start_time:.2f}초) ---")

# --- 1차 분류 노드 ---

def classify_question_type(state: GraphState) -> dict:
    t0 = time.perf_counter()
    question = state["question"]
    prompt = (
        "당신은 질문을 분류하는 AI 어시스턴트입니다.\n"
        "다음 질문을 읽고 '일반 캠핑' 또는 '장소 추천' 중 하나의 단어로만 답변하세요.\n\n"
        f"질문: {question}\n"
        "분류:"
    )
    try:
        out = oai_text(prompt)
        classification = (out["text"].splitlines() or [""])[0].strip()
        if classification not in ["일반 캠핑", "장소 추천"]:
            classification = "일반 캠핑"
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
    user_input = state["question"]  # 두 번째 질문, 캠핑 유형에 대한 답변
    original_question = state.get("original_question", "")
    prompt = (
        "당신은 사용자의 캠핑 유형 선호도를 분류하는 AI 어시스턴트입니다.\n"
        "사용자의 응답과 원래 질문을 읽고 '유료캠핑장', '글램핑/카라반', '오지/노지캠핑' 중 하나로 답변하세요.\n\n"
        f"원래 질문: {original_question}\n"
        f"사용자 응답: {user_input}\n"
        "분류:"
    )
    try:
        out = oai_text(prompt)
        camping_type = (out["text"].splitlines() or [""])[0].strip()
        if camping_type not in ["유료캠핑장", "글램핑/카라반", "오지/노지캠핑"]:
            camping_type = "유료캠핑장"
        print(f"캠핑유형: {camping_type} ({time.perf_counter() - t0:.2f}s)")
        return {"camping_type_preference": camping_type}
    except Exception as e:
        print(f"캠핑유형 분류 오류: {e} ({time.perf_counter() - t0:.2f}s)")
        return {"camping_type_preference": "유료캠핑장", "error_message": str(e)}

# --- RAG + 웹검색 노드 (웹검색은 naver api) ---
async def search_camping(state: GraphState, camping_type: str) -> dict:
    """RAG로 문서 2개 뽑고 → 네이버 API로 각 문서당 최신글 1개 본문 파싱 → 키워드 문장만 묶어서 컨텍스트에 추가"""
    question = state.get("original_question", state["question"])
    print(f"\n--- 🔍 {camping_type} 유형으로 벡터DB 검색 시작 ---")

    try:
        # RAG: 캠핑유형 필터 지원 여부에 따라 분기
        try:
            docs_with_metadata = await asyncio.to_thread(
                vectordb.similarity_search_with_score,
                query=question,
                k=2,
                filter={"캠핑유형": camping_type},
            )
        except TypeError:
            docs_with_metadata = await asyncio.to_thread(
                vectordb.similarity_search_with_score,
                query=question,
                k=2,
            )

        locations_data: List[Dict[str, Any]] = []
        unique_names = set()
        
        # STEP 1: 네이버 검색 결과를 먼저 가져와서 딕셔너리로 저장
        web_snippets_map = {}
        if camping_type != "오지/노지캠핑" and docs_with_metadata:
            print("--- 🌐 네이버 검색 + 본문 파싱 시작 ---")
            snippets = await build_snippet_per_doc(
                docs_with_metadata=docs_with_metadata,
                per_type_display=20,
                fetch_timeout=8,
                max_chars=2000,
                enforce_name_in_title=True,
                include_when_no_local_modified=True,
            )
            web_snippets_map = {s['장소이름']: s for s in snippets}
            if web_snippets_map:
                print(f"✔️ 네이버 스니펫 {len(snippets)}개를 가져왔습니다.")
            else:
                print("⚠️ 네이버 스니펫을 만들 수 있는 최신 글이 없습니다.")

        # STEP 2: 로컬 문서와 네이버 검색 결과를 장소별로 묶어서 하나의 딕셔너리에 담기
        if docs_with_metadata:
            print("✔️ RAG 검색 결과가 발견되었습니다.")
            for i, (doc, score) in enumerate(docs_with_metadata):
                meta = getattr(doc, "metadata", {}) or {}
                location_name = meta.get("캠핑장이름", "이름 정보 없음")
                
                if location_name in unique_names:
                    continue
                unique_names.add(location_name)

                print(f"  [{i+1}] 문서 유사도: {score:.4f} | 메타: {meta}")

                # 각 장소별 정보를 담을 딕셔너리 생성
                location_info = {
                    "local_document": {
                        "metadata": meta,
                        "content": doc.page_content,
                        "score": score
                    },
                    "web_snippet": web_snippets_map.get(location_name)
                }
                
                locations_data.append(location_info)
        else:
            print(f"⚠️ {camping_type} 유형에 대한 문서가 벡터DB에서 검색되지 않았습니다.")

        # 최종 반환할 context와 locations 리스트 생성
        final_context = []
        final_locations = []
        for i, loc_data in enumerate(locations_data):
            loc_name = loc_data['local_document']['metadata'].get('캠핑장이름')
            loc_address = loc_data['local_document']['metadata'].get('캠핑장주소')
            
            context_str = f"[LOCAL{i+1}] 문서 메타데이터:\n{loc_data['local_document']['metadata']}\n\n" \
                          f"문서 내용: {loc_data['local_document']['content'][:]}...\n"

            if loc_data.get('web_snippet'):
                context_str += (
                    f"\n[네이버 웹 검색 결과]\n"
                    f"- 링크: {loc_data['web_snippet'].get('링크주소', '')}\n"
                    f"- 키워드 문장: {loc_data['web_snippet'].get('본문내용', '')}"
                )
            
            final_context.append(context_str)
            final_locations.append({
                "name": loc_name,
                "address": loc_address,
                "latitude": loc_data['local_document']['metadata'].get('캠핑장 위도'),
                "longitude": loc_data['local_document']['metadata'].get('캠핑장 경도'),
            })

        final_context_str = "\n\n" + "\n\n--- [다음 장소] ---\n\n".join(final_context)

        print(f"[DEBUG] search_camping() 반환 locations: {len(final_locations)}개")
        # 'context' 키의 값으로 리스트 대신 하나의 구분된 문자열을 반환
        return {"context": [final_context_str], "locations": final_locations, "search_attempted": True}

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        return {
            "context": ["검색 오류 발생"],
            "locations": [],
            "search_attempted": True,
            "error_message": str(e),
        }
async def search_paid_camping(state):
    return await search_camping(state, "유료캠핑장")


async def search_glamping_caravan(state):
    return await search_camping(state, "글램핑/카라반")


async def search_ojee_camping(state):
    return await search_camping(state, "오지/노지캠핑")


# --- 장소 추천 최종 답변 생성 ---

def generate_location_answer(state: GraphState) -> dict:
    t0 = time.perf_counter()
    
    original_question = state.get("original_question", "")
    second_question = state.get("question", "")
    camping_type = state.get("camping_type_preference", "")
    context = state.get("context", [])
    locations = state.get("locations", [])
    print(f"[DEBUG] 최종 반환될 장소 개수: {len(locations)}개")
    
    context_str = "\n".join(str(ctx) for ctx in context[:5])

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
        print(f"✅ 장소 추천 답변 생성 완료 ({time.perf_counter() - t0:.2f}s | req={out['request_id']})")
        return {"final_answer": answer, "locations": locations}
    except Exception as e:
        print(f"❌ 답변 생성 오류: {e} ({time.perf_counter() - t0:.2f}s)")
        return {"final_answer": "답변 생성 중 오류가 발생했습니다.", "error_message": str(e), "locations": locations}


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
