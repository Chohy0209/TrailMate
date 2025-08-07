import operator
from typing import TypedDict, Annotated, List, Any
import asyncio
import requests
import time
import random
import torch

# LangChain 및 Transformers 라이브러리
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.graph import StateGraph, END, START
from langchain.schema import Document

# ==============================================================================
# 1. 설정 및 상수
# ==============================================================================
# LangGraph 루프 최대 횟수 (무한 루프 방지)
MAX_LOOPS = 2
# LLM 호출 시 타임아웃 (초 단위)
LLM_TIMEOUT = 30
# 모델 및 DB 경로
MODEL_DIR = "merged_model"
CLASSIFIER_MODEL_NAME = "merged_model"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CHROMA_DB_PATH = './data/camp_chroma_store'

# ==============================================================================
# 2. 상태 정의 (LangGraph State)
# ==============================================================================
class GraphState(TypedDict):
    """
    LangGraph 상태를 정의하는 TypedDict
    """
    question: str
    classification: str
    context: Annotated[List[Any], operator.add]
    locations: Annotated[List[dict], operator.add]
    final_answer: str
    route_decision: str
    loop_count: int
    search_attempted: bool
    error_message: str

# ==============================================================================
# 3. 모델 및 벡터 DB 로딩
# ==============================================================================
print("--- LLM 모델 로딩 중 ---")
start_time = time.perf_counter()

# 메인 LLM 모델
model_main = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.float16)
tokenizer_main = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer_main.pad_token is None:
    tokenizer_main.pad_token = tokenizer_main.eos_token
llm_pipeline_main = pipeline("text-generation", model=model_main, tokenizer=tokenizer_main, max_new_tokens=1024, do_sample=True, temperature=0.7, pad_token_id=tokenizer_main.eos_token_id)
llm_main = HuggingFacePipeline(pipeline=llm_pipeline_main)

# 분류용 경량 모델
model_classifier = AutoModelForCausalLM.from_pretrained(CLASSIFIER_MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
tokenizer_classifier = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
if tokenizer_classifier.pad_token is None:
    tokenizer_classifier.pad_token = tokenizer_classifier.eos_token
llm_pipeline_classifier = pipeline("text-generation", model=model_classifier, tokenizer=tokenizer_classifier, max_new_tokens=100, do_sample=True, temperature=0.3, pad_token_id=tokenizer_classifier.eos_token_id)
llm_classifier = HuggingFacePipeline(pipeline=llm_pipeline_classifier)

end_time = time.perf_counter()
print(f"--- 모델 로딩 완료 (소요 시간: {end_time - start_time:.2f}초) ---")

print("--- 임베딩 모델 및 ChromaDB 로딩 중 ---")
start_time = time.perf_counter()
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
end_time = time.perf_counter()
print(f"--- 임베딩 모델 및 ChromaDB 로딩 완료 (소요 시간: {end_time - start_time:.2f}초) ---")

# ==============================================================================
# 4. 헬퍼 함수
# ==============================================================================
def extract_llm_response(raw_response: str, prompt: str) -> str:
    """LLM 응답에서 실제 답변 부분만 추출"""
    response_part = raw_response.split(prompt)[-1].strip()
    first_line = response_part.split('\n')[0].strip()
    return first_line

# ==============================================================================
# 5. LangGraph 노드 함수 (Graph Nodes)
# ==============================================================================
def primary_classify_question(state: GraphState) -> dict:
    """
    질문을 '일반 캠핑' 또는 '장소 추천'으로 1차 분류
    """
    print("--- 1차 분류 중 (일반 vs 장소) ---")
    question = state['question']
    prompt = f"""당신은 질문을 1차적으로 분류하는 AI 어시스턴트입니다.
다음 질문을 읽고 '일반 캠핑' 또는 '장소 추천' 중 하나의 단어로만 답변하세요.

<분류 기준>
- 일반 캠핑: 캠핑 준비물, 장비, 팁, 방법 등 캠핑 자체에 대한 질문
- 장소 추천: 특정 지역의 캠핑 장소를 추천해달라는 질문

질문: {question}
분류:"""
    try:
        raw_response = llm_classifier.invoke(prompt)
        classification = extract_llm_response(raw_response, prompt)
        
        if classification not in ['일반 캠핑', '장소 추천']:
            classification = '일반 캠핑'
            print("⚠️ 1차 분류 결과가 불분명하여 '일반 캠핑'으로 기본 처리")

        print(f"1차 분류 LLM 응답: {classification}")
        return {"classification": classification}

    except Exception as e:
        print(f"1차 분류 오류: {e}. '일반 캠핑'으로 처리합니다.")
        return {"classification": "일반 캠핑", "error_message": str(e)}

async def retrieve_context_async(state: GraphState) -> dict:
    """
    '장소 추천' 질문에 대해 비동기적으로 문맥(context)과 위치 정보를 검색
    """
    print("--- 장소 추천 질문에 대해 정보 검색 시작 ---")
    question = state['question']
    locations = []  # 위치 정보를 저장할 리스트

    try:
        # '장소 추천' 질문은 일단 RAG 검색을 먼저 수행
        docs = await asyncio.to_thread(retriever.get_relevant_documents, question)
        context = [f"[벡터DB] {doc.page_content}" for doc in docs]

        # 메타데이터에서 위치 정보 추출
        for doc in docs:
            if '위도' in doc.metadata and '경도' in doc.metadata:
                # 캠핑장 이름은 page_content의 첫 줄로 가정
                name = doc.page_content.split('\n')[0].strip()
                locations.append({
                    "name": name,
                    "lat": doc.metadata['위도'],
                    "lon": doc.metadata['경도']
                })
        
        if locations:
            print(f"     -> {len(locations)}개의 위치 정보를 찾았습니다.")

        # RAG 검색 결과의 메타데이터를 확인하여 웹 검색이 필요한지 판단
        requires_web_search = False
        for doc in docs:
            # 메타데이터에 '유료' 또는 '글램핑', '카라반' 등의 키워드가 포함되어 있는지 확인
            if '유료' in doc.metadata.get('type', '') or \
               '글램핑' in doc.page_content or \
               '카라반' in doc.page_content:
                requires_web_search = True
                break
        
        if requires_web_search:
            print("     -> 검색 결과에 유료/시설 관련 정보가 포함되어 있어 웹 검색을 추가 진행합니다.")
            web_results = await web_search_real_async(question)
            context.extend(web_results)

        print(f"총 검색 결과: {len(context)}개")
        return {"context": context, "locations": locations, "search_attempted": True}
    
    except Exception as e:
        print(f"정보 검색 오류: {e}")
        return {"context": ["검색 중 오류가 발생했습니다."], "locations": [], "error_message": str(e), "search_attempted": True}

async def web_search_real_async(query: str) -> List[str]:
    """유료 캠핑장 전용 웹 검색 시뮬레이션 (더미 데이터)를 비동기적으로 실행"""
    print(f"     -> 웹 검색 중: '{query}'")
    await asyncio.sleep(0.5) # 실제 API 호출 시뮬레이션
    paid_camping_data = [
        "[웹검색] 가평 별빛글램핑파크 - 예약 가능, 1박 15만원, 바베큐시설 완비",
        "[웹검색] 춘천 남이섬글램핑 - 주말 예약 마감, 평일 예약 가능, 호수뷰",
        "[웹검색] 양평 들꽃수목원 오토캠핑장 - 전기/수도 시설, 1박 3만원",
    ]
    regional_keywords = {"강원": ["춘천"], "경기": ["가평", "양평"]}
    region_specific = []
    if any(city in query for city in regional_keywords.get("경기", [])):
        region_specific.extend(["[웹검색] 경기 지역 인기 글램핑장 TOP 5"])
    
    all_data = paid_camping_data + region_specific
    num_results = random.randint(2, 4)
    selected_results = random.sample(all_data, min(num_results, len(all_data)))
    
    print(f"     -> 웹 검색 완료: {len(selected_results)}개 결과")
    return selected_results

def generate_final_answer(state: GraphState) -> dict:
    """
    주어진 문맥을 바탕으로 최종 답변 생성 (문맥이 없으면 자체 생성)
    """
    print("--- 최종 답변 생성 중 ---")
    start_time = time.perf_counter()
    
    question = state['question']
    context = state.get('context', [])
    
    context_str = ""
    if context:
        context_str = "\n".join([ctx.replace("[웹검색]", "").replace("[벡터DB]", "").strip() for ctx in context[:5]])
    
    prompt = f"""당신은 친절하고 능숙한 캠핑 전문가 챗봇 입니다.
{'주어진 문맥(context)을 바탕으로' if context else ''} 사용자의 질문에 답변해주세요.
만약 주어진 문맥만으로는 답변하기 어렵다면, "죄송합니다. 제공된 정보로는 답변하기 어렵습니다."라고 말해주세요.
답변은 구체적이고 도움이 되는 정보를 포함해야 합니다.

{f"문맥: {context_str}" if context_str else ""}
질문: {question}

답변:"""

    try:
        response = llm_pipeline_main(prompt, 
        max_new_tokens=512, 
        do_sample=True, 
        temperature=0.6,
        repetition_penalty=1.2,

        pad_token_id=tokenizer_main.eos_token_id, 
        eos_token_id=tokenizer_main.eos_token_id)
        final_answer = response[0]['generated_text'].replace(prompt, "").strip().replace("답변:", "").strip()
        
        if not final_answer or len(final_answer.strip()) < 10:
            if context_str:
                final_answer = f"""죄송합니다. 구체적인 답변을 생성하지 못했습니다.
다음 정보를 참고해 주세요:\n{context_str[:500]}"""
            else:
                 final_answer = "죄송합니다. 구체적인 답변을 생성하지 못했습니다. 다시 시도해주세요."

        end_time = time.perf_counter()
        print(f"--- 최종 답변 생성 완료 (소요 시간: {end_time - start_time:.2f}초) ---")
        return {"final_answer": final_answer, "loop_count": state.get('loop_count', 0) + 1}

    except Exception as e:
        print(f"답변 생성 오류: {e}")
        return {"final_answer": "답변 생성 중 오류가 발생했습니다. 다시 시도해주세요.", "error_message": str(e), "loop_count": state.get('loop_count', 0) + 1}


def check_answer_quality(state: GraphState) -> dict:
    """
    생성된 답변의 품질을 검증하고, 필요시 추가 검색을 결정
    """
    print("--- 답변 품질 검증 중 ---")
    final_answer = state['final_answer']
    context = state.get('context', [])
    current_loop = state.get('loop_count', 0)
    
    # RAG가 필요한 질문에만 재시도 로직 적용
    if state['classification'] == '장소 추천':
        insufficient_indicators = ["답변 생성 중 오류", "제공된 정보가 부족", "찾을 수 없습니다", "답변하기 어렵습니다"]
        is_insufficient = any(indicator in final_answer for indicator in insufficient_indicators)
        
        if is_insufficient and current_loop < MAX_LOOPS:
            print(f"     -> 답변이 부족합니다. (현재 루프: {current_loop}/{MAX_LOOPS}) 추가 검색을 시도합니다.")
            return {"route_decision": "continue"}
        else:
            print("     -> 답변을 완료하거나 재시도 횟수를 초과했습니다.")
            return {"route_decision": "end"}
    else:
        # 일반 캠핑 질문은 재시도 없이 바로 종료
        print("     -> 일반 캠핑 질문이므로 바로 종료합니다.")
        return {"route_decision": "end"}


def route_primary_classification(state: GraphState):
    """
    1차 분류 결과에 따라 다음 노드를 결정
    """
    if state['classification'] == '일반 캠핑':
        # 일반 캠핑 질문은 검색 없이 바로 답변 생성
        return 'generate'
    elif state['classification'] == '장소 추천':
        # 장소 추천 질문은 검색 노드를 거침
        return 'retrieve'
    else: # 예외 상황
        return 'generate'


# ==============================================================================
# 6. LangGraph 그래프 구성
# ==============================================================================
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("primary_classify", primary_classify_question)
# 'retrieve' 노드 이름을 변경하여 의미를 명확하게 함
workflow.add_node("retrieve", retrieve_context_async)
workflow.add_node("generate", generate_final_answer)
workflow.add_node("check_quality", check_answer_quality)

# 시작점 설정
workflow.set_entry_point("primary_classify")

# 1차 분류 결과에 따라 다음 노드 결정
workflow.add_conditional_edges(
    "primary_classify",
    route_primary_classification,
    {
        "generate": "generate",
        "retrieve": "retrieve"
    }
)

# 'retrieve' 노드에서 'generate' 노드로 이동
workflow.add_edge("retrieve", "generate")

# 답변 생성 후 품질 검사
workflow.add_edge("generate", "check_quality")

# 품질 검사 결과에 따라 추가 검색 또는 종료
workflow.add_conditional_edges(
    "check_quality",
    lambda state: state['route_decision'],
    {"continue": "retrieve", "end": END}
)

app = workflow.compile()

# ==============================================================================
# 7. 테스트 함수 및 메인 실행
# ==============================================================================
def test_camping_qa(question: str):
    print("="*60)
    print(f"질문: {question}")
    print("="*60)
    
    start_time = time.perf_counter()
    
    try:
        # 초기 상태에 locations 필드 추가
        initial_state = {
            "question": question, 
            "loop_count": 0, 
            "context": [], 
            "locations": [],  # <--- 초기화
            "search_attempted": False
        }
        result = asyncio.run(app.ainvoke(initial_state))
        
        end_time = time.perf_counter()
        
        print(f"\n🎯 최종 분류: {result.get('classification', 'N/A')}")
        
        # 위치 정보 출력 로직 추가
        if result.get('locations'):
            print(f"\n📍 추천 장소 (지도 표시용):")
            for loc in result['locations']:
                print(f"     - {loc['name']} (위도: {loc['lat']}, 경도: {loc['lon']})")

        print(f"\n📝 최종 답변:")
        print("-" * 40)
        print(result['final_answer'])
        print("-" * 40)
        
        print(f"\n📊 실행 정보:")
        print(f"     - 전체 실행 시간: {end_time - start_time:.2f}초")
        print(f"     - 사용된 정보 소스 수: {len(result.get('context', []))}")
        print(f"     - 루프 횟수: {result.get('loop_count', 0)}")
        
        if result.get('context'):
            print(f"\n📚 검색된 정보 소스:")
            for i, doc in enumerate(result['context'][:3], 1):
                print(f"     {i}. {doc[:100]}..." if len(doc) > 100 else f"     {i}. {doc}")
        
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏕️ 개선된 캠핑 Q&A 시스템 시작")
    print("="*60)
    
    test_questions = [
        "캠핑할 때 필요한 준비물이 뭐야?",
        "강원도에서 노지 캠핑 할 수 있는 곳 추천해줘",
        "가평 글램핑 예약 가능한 곳 있어?",
        "부산 근처 카라반 캠핑장 어디가 좋아?",
        "겨울 캠핑 텐트 고르는 방법 알려줘",
        "제주도 글램핑 추천해줘",
        "오지 캠핑 갈 때 주의사항은?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        test_camping_qa(question)
        if i < len(test_questions):
            print("\n" + "="*60 + "\n")
    
    print("\n🎉 모든 테스트 완료!")
