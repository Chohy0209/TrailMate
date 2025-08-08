# server.py
from flask import Flask, render_template, request, jsonify, session
import asyncio
import os

# filterrag_lang.py에서 main_app과 continuation_app을 임포트
from filterrag_lang import main_app, continuation_app

app = Flask(__name__)
# Flask 세션을 사용하기 위해 시크릿 키 설정 (실제 운영 환경에서는 강력한 키로 변경해야 함)
app.secret_key = os.urandom(24)

@app.route("/")
def index():
    # 세션 초기화
    session.clear()
    return render_template("index.html")

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