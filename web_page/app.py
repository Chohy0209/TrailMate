# app.py
import asyncio
import httpx
from quart import Quart, jsonify, render_template, request, session
from quart_cors import cors

import config
from services import driver
from rag_components import UnifiedBGEM3Embedder, build_workflows

# --- Quart App Initialization ---
app = Quart(__name__)
app = cors(app, allow_origin="*") # 모든 출처에서의 요청 허용 (개발용)
app.secret_key = config.FLASK_SECRET_KEY

# --- Web Server Routes ---
@app.route("/")
async def index():
    """메인 페이지 렌더링"""
    session.clear()
    return await render_template(
        "trailmate_main_page.html", 
        tmap_api_key=config.TMAP_API_KEY, 
        openweathermap_api_key=config.OPENWEATHERMAP_API_KEY
    )

@app.route("/camptory_chat", methods=["GET", "POST"])
async def camptory_chat():
    """채팅 페이지 렌더링"""
    initial_message = (await request.form).get("message") if request.method == "POST" else None
    return await render_template(
        "trailmate_chatting_page.html", 
        tmap_api_key=config.TMAP_API_KEY, 
        openweathermap_api_key=config.OPENWEATHERMAP_API_KEY, 
        initial_message=initial_message
    )

@app.route("/get_tmap_route", methods=["POST"])
async def get_tmap_route():
    """TMAP API를 호출하여 경로 정보 반환"""
    data = await request.get_json()
    if not all(k in data for k in ["startX", "startY", "endX", "endY"]):
        return jsonify({"error": "필수 파라미터가 누락되었습니다."}), 400

    payload = {
        "startX": data["startX"], "startY": data["startY"],
        "endX": data["endX"], "endY": data["endY"],
        "startName": "현재 위치", "endName": data.get("endName", "목적지"),
        "searchOption": 0, "trafficInfo": "Y"
    }
    url = "https://apis.openapi.sk.com/tmap/routes?version=1"
    headers = {"appKey": config.TMAP_API_KEY, "content-type": "application/json"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return jsonify(response.json())
    except httpx.HTTPStatusError as e:
        print(f"TMAP API 오류: {e.response.status_code} - {e.response.text}")
        return jsonify({"error": f"TMAP API 오류: {e.response.status_code}"}), e.response.status_code
    except Exception as e:
        print(f"경로 데이터 처리 오류: {e}")
        return jsonify({"error": "경로 데이터를 처리하는 중 내부 오류가 발생했습니다."}), 500

@app.route("/chat", methods=["POST"])
async def chat():
    """챗봇 응답을 처리하는 메인 엔드포인트"""
    data = await request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"answer": "메시지를 입력해주세요.", "locations": []})

    conversation_state = session.get('state', 'new_conversation')
    
    initial_state = {
        "question": message,
        "original_question": session.get('original_question', message),
        "unified_embedder": app.unified_embedder 
    }

    try:
        workflow = app.continuation_workflow if conversation_state == 'awaiting_camping_type' else app.main_workflow
        result = await workflow.ainvoke(initial_state)

        if "어떤 스타일의 캠핑을 원하시나요?" in result.get('final_answer', ''):
            session['state'] = 'awaiting_camping_type'
            session['original_question'] = message
        else:
            session.clear()

        raw_locations = result.get("locations", [])
        unique_locations = list({loc['name']: loc for loc in raw_locations if loc.get('name')}.values())
        return jsonify({"answer": result.get("final_answer", "답변 생성 실패"), "locations": unique_locations})
    except Exception as e:
        print(f"chat - LangGraph 호출 오류: {e}")
        session.clear()
        return jsonify({"answer": "죄송합니다. 처리 중 오류가 발생했습니다.", "locations": []}), 500


# --- Application Startup ---
@app.before_serving
async def initialize_app():
    """Uvicorn/Hypercorn 서버가 실행된 후, 요청을 받기 전에 AI 컴포넌트를 초기화합니다."""
    print("🚀 Quart 애플리케이션 시작... AI 컴포넌트를 초기화합니다.")
    
    app.unified_embedder = await UnifiedBGEM3Embedder.create()
    app.main_workflow, app.continuation_workflow = build_workflows()
    
    async with driver.session() as session:
        try:
            await session.run(f"CREATE VECTOR INDEX camp_embedding_index IF NOT EXISTS FOR (c:Camp) ON (c.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {config.VECTOR_DIM}, `vector.similarity_function`: '{config.SIM_FUNC}'}}}}")
        except Exception as e:
            print(f"인덱스 생성 중 경고 발생 (이미 존재할 수 있음): {e}")

    print("✅ AI 컴포넌트 초기화 완료. 서버가 요청을 받을 준비가 되었습니다.")


if __name__ == "__main__":
    print("=" * 50)
    print("⚠️  이 파일은 직접 실행하는 것이 아닙니다.")
    print("👇 아래 명령어를 터미널에 입력하여 서버를 실행하세요:")
    print("uvicorn app:app --host 0.0.0.0 --port 5000")
    print("=" * 50)