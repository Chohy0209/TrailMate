import operator
from typing import TypedDict, Annotated, List, Any
import asyncio
import requests
import time
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langgraph.graph import StateGraph, END
from langchain.schema import Document

# 설정
MAX_LOOPS = 2
LLM_TIMEOUT = 30
MODEL_DIR = "./model_ax_merge4"
CLASSIFIER_MODEL_NAME = "./model_ax_merge4"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CHROMA_DB_PATH = './data/camp_chroma_store'

class GraphState(TypedDict):
    question: str
    original_question: str
    classification: str
    camping_type_preference: str
    context: Annotated[List[Any], operator.add]
    locations: Annotated[List[dict], operator.add] # locations 추가
    final_answer: str
    loop_count: int
    search_attempted: bool
    error_message: str

# 모델 로딩 (기존 코드 유지)
print("--- LLM 모델 로딩 중 ---")
start_time = time.perf_counter()

model_main = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.float16)
tokenizer_main = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer_main.pad_token is None:
    tokenizer_main.pad_token = tokenizer_main.eos_token
tokenizer_main.pad_token = tokenizer_main.eos_token

llm_pipeline_main = pipeline("text-generation", model=model_main, tokenizer=tokenizer_main, max_new_tokens=1024, do_sample=True, temperature=0.7, pad_token_id=tokenizer_main.eos_token_id)
llm_main = HuggingFacePipeline(pipeline=llm_pipeline_main)

model_classifier = AutoModelForCausalLM.from_pretrained(CLASSIFIER_MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
tokenizer_classifier = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
if tokenizer_classifier.pad_token is None:
    tokenizer_classifier.pad_token = tokenizer_classifier.eos_token

llm_pipeline_classifier = pipeline("text-generation", model=model_classifier, tokenizer=tokenizer_classifier, max_new_tokens=100, do_sample=False, temperature=0.0, pad_token_id=tokenizer_classifier.eos_token_id)
llm_classifier = HuggingFacePipeline(pipeline=llm_pipeline_classifier)
end_time = time.perf_counter()
print(f"--- 모델 로딩 완료 (소요 시간: {end_time - start_time:.2f}초) ---")

print("--- 임베딩 모델 및 ChromaDB 로딩 중 ---")
start_time = time.perf_counter()
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
end_time = time.perf_counter()
print(f"--- 임베딩 모델 및 ChromaDB 로딩 완료 (소요 시간: {end_time - start_time:.2f}초) ---")

def extract_llm_response(raw_response: str, prompt: str) -> str:
    response_part = raw_response.split(prompt)[-1].strip()
    first_line = response_part.split('\n')[0].strip()
    return first_line

def extract_locations_from_docs(docs: List[Document]) -> List[dict]:
    locations = []
    for doc in docs:
        if hasattr(doc, 'metadata') and '위도' in doc.metadata and '경도' in doc.metadata:
            name = doc.page_content.split('\n')[0].strip()
            locations.append({
                "name": name,
                "lat": doc.metadata['위도'],
                "lon": doc.metadata['경도']
            })
    return locations

# ===== 노드 1: 1차 분류 (일반 vs 장소추천) =====
def classify_question_type(state: GraphState) -> dict:
    print("--- [노드1] 질문 유형 분류 (일반 vs 장소추천) ---")
    question = state['question']
    prompt = f"""당신은 질문을 분류하는 AI 어시스턴트입니다.
다음 질문을 읽고 '일반 캠핑' 또는 '장소 추천' 중 하나의 단어로만 답변하세요.

<분류 기준>
- 일반 캠핑: 캠핑 준비물, 장비, 팁, 방법 등 캠핑 자체에 대한 질문
- 장소 추천: 장소를 추천해달라는 질문이나 장소를 물어보는 질문

질문: {question}
분류:"""
    
    try:
        raw_response = llm_classifier.invoke(prompt)
        classification = extract_llm_response(raw_response, prompt)
        if classification not in ['일반 캠핑', '장소 추천']:
            classification = '일반 캠핑'
            print("⚠️ 분류 결과가 불분명하여 '일반 캠핑'으로 기본 처리")
        print(f"분류 결과: {classification}")
        return {
            "classification": classification,
            "original_question": question
        }
    except Exception as e:
        print(f"분류 오류: {e}. '일반 캠핑'으로 처리합니다.")
        return {
            "classification": "일반 캠핑",
            "error_message": str(e),
            "original_question": question
        }

# ===== 노드 2: 일반 질문 답변 생성 =====
def generate_general_answer(state: GraphState) -> dict:
    print("--- [노드2] 일반 캠핑 질문 답변 생성 ---")
    question = state['question']
    prompt = f"""당신은 캠핑 전문가입니다. 다음 캠핑 관련 질문에 답변해주세요.

질문: {question}
답변:"""
    
    try:
        response = llm_pipeline_main(prompt, max_new_tokens=512)[0]['generated_text']
        final_answer = response.replace(prompt, "").strip()
        return {"final_answer": final_answer}
    except Exception as e:
        return {"final_answer": "답변 생성 중 오류가 발생했습니다.", "error_message": str(e)}

# ===== 노드 3: 캠핑 유형 선택 요청 (3가지로 세분화) =====
def ask_camping_preference(state: GraphState) -> dict:
    print("--- [노드3] 캠핑 유형 선택 요청 ---")
    clarification_message = """🏕️ 장소를 추천해드릴게요! 
어떤 스타일의 캠핑을 원하시나요?

ex)

1️⃣ **유료캠핑장** - 오토캠핑장, 일반 캠핑장, 편의시설 완비
2️⃣ **글램핑/카라반** - 글램핑장, 카라반, 펜션형 캠핑 
3️⃣ **오지/노지캠핑** - 자연 속 야생캠핑, 백패킹, 무료 캠핑
"""
    return {"final_answer": clarification_message}

# ===== 노드 4: 캠핑 유형 분류 (3가지로 세분화) =====
def classify_camping_type(state: GraphState) -> dict:
    print("--- [노드4] 캠핑 유형 분류 (유료/글램핑/오지) ---")
    user_input = state['question']
    prompt = f"""당신은 사용자의 캠핑 유형 선호도를 분류하는 AI 어시스턴트입니다.
사용자의 응답을 읽고 '유료캠핑장', '글램핑카라반', '오지노지캠핑' 중 하나의 단어로만 답변하세요.

<분류 기준>
- 유료캠핑장: 1, 1번, 유료, 오토캠핑장, 일반캠핑장, 편의시설을 원하는 경우
- 글램핑카라반: 2, 2번, 글램핑, 카라반, 펜션, 럭셔리, 편안한 캠핑,캠핑카를를 원하는 경우  
- 오지노지캠핑: 3, 3번, 오지, 노지, 야생캠핑, 백패킹, 자연 속 캠핑, 무료를 원하는 경우

사용자 응답: {user_input}
분류:"""
    
    try:
        raw_response = llm_classifier.invoke(prompt)
        camping_type = extract_llm_response(raw_response, prompt)
        if camping_type not in ['유료캠핑장', '글램핑카라반', '오지노지캠핑']:
            camping_type = '유료캠핑장'
            print("⚠️ 분류 결과가 불분명하여 '유료캠핑장'으로 기본 처리")
        print(f"캠핑 유형 분류 결과: {camping_type}")
        return {"camping_type_preference": camping_type}
    except Exception as e:
        print(f"캠핑 유형 분류 오류: {e}. '유료캠핑장'으로 처리합니다.")
        return {
            "camping_type_preference": "유료캠핑장",
            "error_message": str(e)
        }

# ===== 노드 5: 유료캠핑장 검색 =====
async def search_paid_camping(state: GraphState) -> dict:
    print("--- [노드5] 유료캠핑장 RAG+웹검색 ---")
    original_question = state.get('original_question', state['question'])
    try:
        docs = await asyncio.to_thread(
            vectordb.similarity_search,
            query=original_question,
            k=2,
            filter={"캠핑유형": '유료캠핑장'}
        )
        context = docs.copy()
        locations = extract_locations_from_docs(docs)
        print(f"유료캠핑장 검색 결과: {len(docs)}개 문서, 위치 {len(locations)}개")

        web_results = await general_web_search_async(original_question, "유료캠핑장")
        context.extend(web_results)
        print(f"웹검색 결과: {len(web_results)}개 추가")

        return {"context": context, "locations": locations, "search_attempted": True}
    except Exception as e:
        print(f"유료캠핑장 검색 오류: {e}")
        return {
            "context": ["유료캠핑장 검색 중 오류 발생"],
            "locations": [],
            "error_message": str(e),
            "search_attempted": True
        }

# ===== 노드 6: 글램핑/카라반 웹검색+RAG =====
async def search_glamping_caravan(state: GraphState) -> dict:
    print("--- [노드6] 글램핑/카라반 RAG+웹검색 ---")
    original_question = state.get('original_question', state['question'])
    try:
        docs = await asyncio.to_thread(
            vectordb.similarity_search,
            query=original_question,
            k=5,
            filter={"캠핑유형": '글램핑/카라반'}
        )
        context = docs.copy()
        locations = extract_locations_from_docs(docs)
        print(f"글램핑/카라반 RAG 검색 결과: {len(docs)}개 문서, 위치 {len(locations)}개")

        web_results = await general_web_search_async(original_question, "글램핑/카라반")
        context.extend(web_results)
        print(f"웹검색 결과: {len(web_results)}개 추가")

        return {"context": context, "locations": locations, "search_attempted": True}
    except Exception as e:
        print(f"글램핑/카라반 검색 오류: {e}")
        return {
            "context": ["글램핑/카라반 검색 중 오류 발생"],
            "locations": [],
            "error_message": str(e),
            "search_attempted": True
        }

# ===== 노드 7: 오지/노지캠핑 벡터DB 검색 =====
async def search_ojee_camping(state: GraphState) -> dict:
    print("--- [노드7] 오지/노지캠핑 RAG 검색 ---")
    original_question = state.get('original_question', state['question'])
    try:
        docs = await asyncio.to_thread(
            vectordb.similarity_search,
            query=original_question,
            k=5,
            filter={"캠핑유형": '오지/노지캠핑'}
        )
        locations = extract_locations_from_docs(docs)
        print(f"오지/노지캠핑 검색 결과: {len(docs)}개 문서, 위치 {len(locations)}개")
        return {"context": docs.copy(), "locations": locations, "search_attempted": True}
    except Exception as e:
        print(f"오지/노지캠핑 검색 오류: {e}")
        return {
            "context": ["오지/노지캠핑 검색 중 오류 발생"],
            "locations": [],
            "error_message": str(e),
            "search_attempted": True
        }

# 웹검색 함수들
async def general_web_search_async(query: str, camping_type: str) -> List[str]:
    await asyncio.sleep(0.5)  # 모의 API 대기시간

    mock_results = {
        "유료캠핑장": [
            "[웹검색] 속초 오토캠핑장 - 바닷가 근처, 전기시설 있음",
            "[웹검색] 강릉 해변 캠핑장 - 샤워장, 매점 완비",
            "[웹검색] 양양 일반 캠핑장 - 예약 가능, 화장실 있음"
        ],
        "글램핑/카라반": [
            "[웹검색] 가평 카라반 캠핑장 - 바베큐, 침대 완비",
            "[웹검색] 제주 글램핑 빌리지 - 오션뷰, 온수샤워",
            "[웹검색] 남해 글램핑 - 해안절경, 럭셔리 텐트"
        ]
    }

    return random.sample(mock_results.get(camping_type, []), k=3)

# ===== 노드 8: 장소추천 답변 생성 =====
def generate_location_answer(state: GraphState) -> dict:
    print("--- [노드8] 장소추천 답변 생성 ---")
    original_question = state.get('original_question', state['question'])
    camping_type = state.get('camping_type_preference', '')
    context = state.get('context', [])
    context_str = "\n".join(
        ctx.page_content if hasattr(ctx, 'page_content') else str(ctx) for ctx in context[:5]
    )
    
    prompt = f"""당신은 캠핑 전문가입니다. 다음 문맥을 참고하여 {camping_type} 추천 질문에 답해주세요.

캠핑 유형: {camping_type}
문맥:
{context_str}

질문: {original_question}
답변:"""
    
    try:
        response = llm_pipeline_main(prompt, max_new_tokens=512)[0]['generated_text']
        final_answer = response.replace(prompt, "").strip()
        # locations는 이미 state에 있으므로 그대로 전달됨
        return {"final_answer": final_answer}
    except Exception as e:
        return {"final_answer": "장소 추천 답변 생성 중 오류가 발생했습니다.", "error_message": str(e)}

# ===== 라우팅 함수들 =====
def route_by_classification(state: GraphState) -> str:
    """1차 분류 결과에 따른 라우팅"""
    if state['classification'] == '일반 캠핑':
        return 'general'
    else:
        return 'location'

def route_by_camping_type(state: GraphState) -> str:
    """캠핑 유형에 따른 라우팅 (3가지)"""
    camping_type = state['camping_type_preference']
    if camping_type == '유료캠핑장':
        return 'paid'
    elif camping_type == '글램핑카라반':
        return 'glamping'
    else:  # 오지노지캠핑
        return 'ojee'

# ===== LangGraph 워크플로우 구성 =====
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("classify_type", classify_question_type)           # 노드1: 1차 분류
workflow.add_node("generate_general", generate_general_answer)       # 노드2: 일반 답변
workflow.add_node("ask_preference", ask_camping_preference)          # 노드3: 선택 요청 (3가지)

# 시작점 설정
workflow.set_entry_point("classify_type")

# 엣지 연결
workflow.add_conditional_edges("classify_type", route_by_classification, {
    "general": "generate_general",
    "location": "ask_preference"
})

workflow.add_edge("generate_general", END)
workflow.add_edge("ask_preference", END)

# 별도의 continuation 워크플로우 (사용자 응답 처리용)
continuation_workflow = StateGraph(GraphState)
continuation_workflow.add_node("classify_camping", classify_camping_type)
continuation_workflow.add_node("search_paid", search_paid_camping)
continuation_workflow.add_node("search_glamping", search_glamping_caravan)
continuation_workflow.add_node("search_ojee", search_ojee_camping)
continuation_workflow.add_node("generate_location", generate_location_answer)

continuation_workflow.set_entry_point("classify_camping")
continuation_workflow.add_conditional_edges("classify_camping", route_by_camping_type, {
    "paid": "search_paid",
    "glamping": "search_glamping", 
    "ojee": "search_ojee"
})
continuation_workflow.add_edge("search_paid", "generate_location")
continuation_workflow.add_edge("search_glamping", "generate_location")
continuation_workflow.add_edge("search_ojee", "generate_location")
continuation_workflow.add_edge("generate_location", END)

# 앱 컴파일
main_app = workflow.compile()
continuation_app = continuation_workflow.compile()

# ===== 메인 실행 함수 =====
def main():
    print("\n" + "=" * 60)
    print("🏕️ 캠핑 챗봇에 오신 것을 환영합니다! '종료' 입력 시 종료됩니다.")
    print("=" * 60)
    
    waiting_for_camping_choice = False
    last_state = None
    
    while True:
        user_input = input("\n❓ 사용자 질문: ").strip()
        if user_input.lower() in ["종료", "quit", "exit"]:
            print("👋 챗봇을 종료합니다.")
            break
        
        if waiting_for_camping_choice:
            # 캠핑 유형 선택 처리
            state = {
                "question": user_input,
                "original_question": last_state.get('original_question', user_input),
                "context": [],
                "locations": [], # locations 초기화
                "search_attempted": False
            }
            result = asyncio.run(continuation_app.ainvoke(state))
            waiting_for_camping_choice = False
            last_state = None
        else:
            # 새로운 질문 처리
            state = {
                "question": user_input,
                "context": [],
                "locations": [], # locations 초기화
                "search_attempted": False,
                "loop_count": 0
            }
            result = asyncio.run(main_app.ainvoke(state))
            
            # 캠핑 유형 선택 요청이면 대기 상태로 전환
            if "어떤 스타일의 캠핑을 원하시나요?" in result.get('final_answer', ''):
                waiting_for_camping_choice = True
                last_state = result
        
        print(f"\n📝 답변: {result.get('final_answer')}\n")
        if result.get('locations'):
            print(f"📍 추천 장소: {result['locations']}")

if __name__ == "__main__":
    main()
