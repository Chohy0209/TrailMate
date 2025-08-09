# server.py
from flask import Flask, render_template, request, jsonify, session
import asyncio
import os
from dotenv import load_dotenv
import requests

# .env 파일에서 환경 변수 로드
load_dotenv()

# filterrag_lang.py에서 main_app과 continuation_app을 임포트
from filterrag_lang import main_app, continuation_app

app = Flask(__name__)
# Flask 세션을 사용하기 위해 시크릿 키 설정 (실제 운영 환경에서는 강력한 키로 변경해야 함)
app.secret_key = os.urandom(24)

@app.route("/")
def index():
    # 세션 초기화
    session.clear()
    # .env 파일에서 TMAP API 키를 가져와 템플릿에 전달
    tmap_api_key = os.getenv("TMAP_API_KEY")
    return render_template("index.html", tmap_api_key=tmap_api_key)

@app.route("/get_tmap_route", methods=["POST"])
def get_tmap_route():
    data = request.get_json()
    tmap_api_key = os.getenv("TMAP_API_KEY")
    
    headers = {
        "appKey": tmap_api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "startX": data["startX"],
        "startY": data["startY"],
        "endX": data["endX"],
        "endY": data["endY"],
        "startName": "현재 위치",
        "endName": "목적지",
        "searchOption" : "0", # 추천 경로
        "trafficInfo" : "Y"   # 교통정보 포함
    }
    
    url = "https://apis.openapi.sk.com/tmap/routes?version=1"
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        route_data = response.json()
        
        traffic_segments = []
        for feature in route_data.get("features", []):
            if feature["geometry"]["type"] == "LineString":
                segment_coords = feature["geometry"]["coordinates"]
                traffic_info = feature.get("properties", {}).get("traffic")
                traffic_segments.append({
                    "coordinates": segment_coords,
                    "traffic": traffic_info
                })

        return jsonify({"traffic_segments": traffic_segments})

    except requests.exceptions.RequestException as e:
        print(f"TMAP API 요청 오류: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"경로 데이터 처리 오류: {e}")
        return jsonify({"error": "경로 데이터를 처리하는 중 오류가 발생했습니다."}), 500

@app.route("/chat", methods=["POST"])
async def chat():
    data = request.get_json()
    message = data.get("message", "")
    conversation_state = session.get('state', 'new_conversation')

    try:
        if conversation_state == 'awaiting_camping_type':
            # 사용자가 캠핑 유형을 선택한 경우
            initial_state = {
                "question": message,
                "original_question": session.get('original_question', ''),
                "context": [],
                "locations": [],
                "search_attempted": False
            }
            result = await continuation_app.ainvoke(initial_state)
            # 대화 상태 초기화
            session.clear()
        else:
            # 새로운 질문인 경우
            initial_state = {
                "question": message,
                "context": [],
                "locations": [],
                "search_attempted": False,
                "loop_count": 0
            }
            result = await main_app.ainvoke(initial_state)

            # 봇이 캠핑 유형을 물어봤는지 확인
            if "어떤 스타일의 캠핑을 원하시나요?" in result.get('final_answer', ''):
                # 대화 상태를 '캠핑 유형 대기'로 변경
                session['state'] = 'awaiting_camping_type'
                # 원래 질문을 세션에 저장
                session['original_question'] = message
            else:
                # 일반 답변이므로 세션 초기화
                session.clear()

        final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")
        locations = result.get("locations", [])

        return jsonify({
            "answer": final_answer,
            "locations": locations
        })

    except Exception as e:
        print(f"Error during langGraph invocation: {e}")
        session.clear() # 오류 발생 시 세션 초기화
        return jsonify({"answer": "죄송합니다. 처리 중 오류가 발생했습니다.", "locations": []}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)