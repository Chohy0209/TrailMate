import os
import time
import asyncio
import random
import operator
from typing import TypedDict, Annotated, List, Any, Dict

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
CHROMA_DB_PATH = "./data/camp_chroma_store"
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
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
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

# --- RAG + 웹검색 노드 (웹검색은 mock) ---

async def general_web_search_async(query: str, camping_type: str) -> List[str]:
    await asyncio.sleep(0.2)  # 실제 API 대기 시뮬레이션
    mock_results = {
        "유료캠핑장": [
            "[웹] 속초 오토캠핑장 후기",
            "[웹] 강릉 해변 캠핑장 추천",
            "[웹] 양양 캠핑장 예약 팁",
        ],
        "글램핑/카라반": [
            "[웹] 가평 카라반 숙소 리뷰",
            "[웹] 제주 글램핑 인기 장소",
            "[웹] 남해 글램핑 시설 안내",
        ],
        "오지/노지캠핑": [
            "[웹] 제주도 노지캠핑 스팟",
            "[웹] 강원도 오지캠핑 금지 구역",
            "[웹] 차박 성지 베스트 5",
        ],
    }
    pool = mock_results.get(camping_type, [])
    k = min(3, len(pool))
    return random.sample(pool, k=k) if k > 0 else []


async def search_camping(state: GraphState, camping_type: str) -> dict:
    question = state.get("original_question", state["question"])
    print(f"\n--- 🔍 {camping_type} 유형으로 벡터DB 검색 시작 ---")
    try:
        # 벡터DB에서 유사도 검색 시 메타필터 지원 유무에 따라 분기
        try:
            docs_with_metadata = await asyncio.to_thread(
                vectordb.similarity_search_with_score,
                query=question,
                k=2,
                filter={"캠핑유형": camping_type},
            )
        except TypeError:
            # langchain/chroma 버전에 따라 filter 인자 미지원 가능 → 필터 없이 검색
            docs_with_metadata = await asyncio.to_thread(
                vectordb.similarity_search_with_score,
                query=question,
                k=2,
            )

        context: List[str] = []
        locations: List[dict] = []
        unique_names = set()
        if docs_with_metadata:
            print("✔️ RAG 검색 결과가 발견되었습니다.")
            for i, (doc, score) in enumerate(docs_with_metadata):
                location_name = doc.metadata.get("캠핑장이름", "이름 정보 없음")
                if location_name not in unique_names:
                    unique_names.add(location_name)
                    
                    metadata_str = f"메타데이터: {getattr(doc, 'metadata', {})}"
                    content_with_metadata = f"문서 내용: {doc.page_content}\n{metadata_str}"
                    context.append(content_with_metadata)
                    
                    location_info = {
                        "name": location_name,
                        "address": doc.metadata.get("캠핑장주소", "주소 정보 없음"),
                        "latitude": doc.metadata.get("위도", None),
                        "longitude": doc.metadata.get("경도", None)
                    }
                    locations.append(location_info)
                    
                    print(
                        f"  [{i+1}] 문서: {doc.page_content[:40]}... | 유사도: {score:.4f} | 메타데이터: {getattr(doc, 'metadata', {})}"
                    )
        else:
            print(f"⚠️ {camping_type} 유형에 대한 문서가 벡터DB에서 검색되지 않았습니다.")

        # 오지/노지캠핑은 웹 검색 생략
        if camping_type != "오지/노지캠핑":
            print("--- 🌐 웹 검색 시작 ---")
            web_results = await general_web_search_async(question, camping_type)
            if web_results:
                print("✔️ 웹 검색 결과가 추가되었습니다.")
                for i, res in enumerate(web_results):
                    context.append(res)
                    print(f"  [웹{i+1}] {res}")
            else:
                print("⚠️ 웹 검색 결과가 없습니다.")
        else:
            print("🌐 '오지/노지캠핑' 유형은 웹 검색을 생략합니다.")

        print(f"[DEBUG] search_camping 함수에서 반환될 locations: {locations}")
        return {"context": context, "locations": locations, "search_attempted": True}
    
    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        return {"context": ["검색 오류 발생"], "locations": [], "search_attempted": True, "error_message": str(e)}


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
        "추천 시에는 반드시 문맥 내 메타데이터를 참고하여 답변하세요.\n"
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
