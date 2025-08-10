# server.py
from flask import Flask, render_template, request, jsonify, session
import asyncio
import os
from dotenv import load_dotenv
import requests

# .env 파일에서 환경 변수 로드
load_dotenv()

# model_gpt.py에서 main_app과 continuation_app을 임포트
from model_gpt import main_app, continuation_app

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
    return render_template("index.html", tmap_api_key=tmap_api_key)

@app.route("/get_tmap_route", methods=["POST"])
def get_tmap_route():
    """
    TMAP API를 호출하여 자동차 경로 정보를 가져옵니다.
    요청 본문에서 출발지와 목적지 좌표를 받아 처리합니다.
    """
    data = request.get_json()
    if not all(k in data for k in ["startX", "startY", "endX", "endY"]):
        return jsonify({"error": "필수 파라미터가 누락되었습니다."}), 400

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
        "endName": data.get("endName", "목적지"),
        "searchOption": "2",  # 추천 경로
        "trafficInfo": "Y"    # 실시간 교통정보 포함
    }
    
    url = "https://apis.openapi.sk.com/tmap/routes?version=1"
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        route_data = response.json()
        
        features = route_data.get("features", [])
        
        # 경로 요약 정보 추출
        summary = features[0].get("properties", {})
        total_time = summary.get("totalTime", 0)
        total_toll = summary.get("totalToll", 0)

        # 경로 데이터에서 경유지 정보 추출
        waypoints = []
        for feature in features:
            if feature.get("geometry", {}).get("type") == "Point":
                point_name = feature.get("properties", {}).get("name", "")
                if point_name and point_name not in ["출발", "도착"]:
                    waypoints.append(point_name)
        
        # Build route summary
        start_name = payload['startName']
        end_name = payload['endName']
        if waypoints:
            route_summary = f"{start_name} → {' → '.join(waypoints)} → {end_name}"
        else:
            route_summary = f"{start_name} → {end_name}"

        # 경로 데이터에서 교통정보 세그먼트 추출
        traffic_segments = []
        for feature in features:
            if feature.get("geometry", {}).get("type") == "LineString":
                segment_coords = feature["geometry"]["coordinates"]
                traffic_info = feature.get("properties", {}).get("traffic")
                traffic_segments.append({
                    "coordinates": segment_coords,
                    "traffic": traffic_info
                })

        return jsonify({
            "traffic_segments": traffic_segments,
            "total_time": total_time,
            "total_toll": total_toll,
            "route_summary": route_summary
        })

    except requests.exceptions.RequestException as e:
        print(f"TMAP API 요청 오류: {e}")
        return jsonify({"error": "TMAP API 연동 중 오류가 발생했습니다."}), 502
    except (IndexError, KeyError) as e:
        print(f"경로 데이터 파싱 오류: {e}")
        return jsonify({"error": "TMAP API 응답 데이터를 처리하는 중 오류가 발생했습니다."}), 500
    except Exception as e:
        print(f"경로 데이터 처리 오류: {e}")
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
        locations = result.get("locations", [])
        print(f"[DEBUG] server.py에서 model_gpt.py로부터 받은 locations: {locations}")

        # Ensure locations is a list and contains unique items if necessary
        # This part is crucial for preventing frontend duplication if backend sends duplicates
        unique_locations = []
        seen_names = set()
        for loc in locations:
            if loc.get('name') not in seen_names:
                unique_locations.append(loc)
                seen_names.add(loc.get('name'))
        locations = unique_locations
        print(f"[DEBUG] server.py에서 frontend로 보낼 최종 locations (중복 제거 후): {locations}")

        return jsonify({
            "answer": final_answer,
            "locations": locations
        })

    except Exception as e:
        print(f"LangGraph 호출 오류: {e}")
        session.clear()  # 오류 발생 시 세션 초기화
        return jsonify({"answer": "죄송합니다. 처리 중 오류가 발생했습니다.", "locations": []}), 500

if __name__ == "__main__":
    # use_reloader=False 옵션은 모델 이중 로딩을 방지하기 위해 필요합니다.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
