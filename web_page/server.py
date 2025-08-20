# server.py
from flask import Flask, render_template, request, jsonify, session
import asyncio
import os
from dotenv import load_dotenv
import requests

# .env 파일에서 환경 변수 로드
load_dotenv()

DEBUG = False

if DEBUG:
    print("loading local model")
    from filterrag_lang import main_app, continuation_app

else:
    # model_gpt.py에서 main_app과 continuation_app을 임포트
    print("loading gpt model")
    from graphrag import main_app, continuation_app
    

app = Flask(__name__)
# Flask 세션을 사용하기 위해 시크릿 키 설정
# 실제 운영 환경에서는 환경 변수 등을 통해 관리되는 강력한 키로 변경해야 합니다.
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

@app.route("/")
def index():
    """
    메인 페이지를 렌더링합니다.
    세션을 초기화하고 TMAP API 키를 템플릿에 전달합니다.
    """
    session.clear()  # 새 방문 시 세션 초기화
    tmap_api_key = os.getenv("TMAP_API_KEY")
    openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    
    # print(f"tmap api : {tmap_api_key}, weather api : {openweathermap_api_key}")
    
    return render_template("trailmate_main_page.html", tmap_api_key=tmap_api_key, openweathermap_api_key=openweathermap_api_key)

@app.route("/camptory_chat", methods=["GET", "POST"])
def camptory_chat():
    """
    채팅 페이지를 렌더링합니다.
    TMAP API 키를 템플릿에 전달합니다.
    POST 요청 시 메시지를 받아 템플릿에 전달합니다.
    """
    tmap_api_key = os.getenv("TMAP_API_KEY")
    openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    
    initial_message = None
    if request.method == "POST":
        initial_message = request.form.get("message")
        
    return render_template("trailmate_chatting_page.html", tmap_api_key=tmap_api_key, openweathermap_api_key=openweathermap_api_key, initial_message=initial_message)

@app.route("/get_tmap_route", methods=["POST"])
def get_tmap_route():
    """
    TMAP API를 호출하여 자동차 경로 정보를 가져옵니다.
    요청 본문에서 출발지와 목적지 좌표를 받아 처리합니다.
    """
    data = request.get_json()
    
    print(f"넘어 온 데이터 : {data} \n")
    
    if not all(k in data for k in ["startX", "startY", "endX", "endY"]):
        return jsonify({"error": "필수 파라미터가 누락되었습니다."}), 400

    tmap_api_key = os.getenv("TMAP_API_KEY")
    
    headers = {
        "appKey": tmap_api_key,
        "content-type": "application/json"
    }
    
    payload = {
        "startX": data["startX"],
        "startY": data["startY"],
        "endX": data["endX"],
        "endY": data["endY"],
        "startName": "현재 위치",
        "endName": data.get("endName", "목적지"),
        "searchOption": 0,  # 추천+실시간교통정보
        "trafficInfo": "Y"    # 실시간 교통정보 포함
    }
    
    print(f"payload : {payload}")
    
    url = "https://apis.openapi.sk.com/tmap/routes?version=1"
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        route_data = response.json()
        
        return jsonify(route_data)

    except requests.exceptions.RequestException as e:
        print(f"TMAP API 요청 오류: {e} \n")
        return jsonify({"error": "TMAP API 연동 중 오류가 발생했습니다."}), 502
    except (IndexError, KeyError) as e:
        print(f"경로 데이터 파싱 오류: {e} \n")
        return jsonify({"error": "TMAP API 응답 데이터를 처리하는 중 오류가 발생했습니다."}), 500
    except Exception as e:
        print(f"경로 데이터 처리 오류: {e} \n")
        return jsonify({"error": "경로 데이터를 처리하는 중 내부 오류가 발생했습니다."}), 500

@app.route("/chat", methods=["POST"])
async def chat():
    """
    사용자 메시지를 받아 챗봇 응답을 반환합니다.
    세션을 이용해 다중 턴 대화 상태를 관리합니다.
    """
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"answer": "메시지를 입력해주세요.", "locations": []})

    # 세션에서 대화 상태 확인
    conversation_state = session.get('state', 'new_conversation')

    try:
        if conversation_state == 'awaiting_camping_type':
            # 상태 2: 사용자가 캠핑 유형을 선택한 후의 응답 처리
            initial_state = {
                "question": message,
                "original_question": session.get('original_question', ''),
                "context": [],
                "locations": [],
                "search_attempted": False
            }
            result = await continuation_app.ainvoke(initial_state)
            session.clear()  # 대화 종료 후 세션 초기화
        else:
            # 상태 1: 새로운 질문 처리
            initial_state = {
                "question": message,
                "context": [],
                "locations": [],
                "search_attempted": False,
                "loop_count": 0
            }
            result = await main_app.ainvoke(initial_state)

            # 봇이 캠핑 유형을 물어봤는지 확인하여 대화 상태 변경
            if "어떤 스타일의 캠핑을 원하시나요?" in result.get('final_answer', ''):
                session['state'] = 'awaiting_camping_type'
                session['original_question'] = message
            else:
                session.clear()  # 일반 답변 후 세션 초기화

        final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")
        
        # Debugging: Log locations received from model_gpt.py
        raw_locations_from_model = result.get("locations", [])
        print(f"[DEBUG] server.py: model_gpt.py로부터 받은 raw locations: {raw_locations_from_model} \n")

        locations = raw_locations_from_model

        # Filter out locations without a name first
        named_locations = [loc for loc in locations if loc.get('name')]

        unique_locations = []
        seen_names = set()
        for loc in named_locations:
            if loc.get('name') not in seen_names:
                unique_locations.append(loc)
                seen_names.add(loc.get('name'))
        locations = unique_locations
        
        # Debugging: Log final locations sent to frontend
        print(f"[DEBUG] server.py: frontend로 보낼 최종 locations (중복 제거 후): {locations} \n")

        return jsonify({
            "answer": final_answer,
            "locations": locations
        })

    except Exception as e:
        print(f"chat - LangGraph 호출 오류: {e} \n")
        session.clear()  # 오류 발생 시 세션 초기화
        return jsonify({"answer": "죄송합니다. 처리 중 오류가 발생했습니다.", "locations": []}), 500

if __name__ == "__main__":
    # use_reloader=False 옵션은 모델 이중 로딩을 방지하기 위해 필요합니다.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
