# server.py
from flask import Flask, render_template, request, jsonify
import asyncio

# langGraph.py에서 app 객체를 임포트
# langGraph.py 파일이 web_page 디렉토리에 있다고 가정합니다.
from langGraph import app as langgraph_app

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
async def chat():
    data = request.get_json()
    message = data.get("message", "")
    
    # langGraph를 실행하기 위한 초기 상태 설정
    initial_state = {
        "question": message,
        "loop_count": 0,
        "context": [],
        "locations": [],
        "search_attempted": False
    }

    try:
        # langGraph 워크플로우 실행
        result = await langgraph_app.ainvoke(initial_state)
        
        final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")
        locations = result.get("locations", [])

        # 클라이언트에 텍스트 응답과 위치 데이터를 함께 JSON으로 전송
        return jsonify({
            "answer": final_answer,
            "locations": locations
        })

    except Exception as e:
        print(f"Error during langGraph invocation: {e}")
        return jsonify({"answer": "죄송합니다. 처리 중 오류가 발생했습니다.", "locations": []}), 500

if __name__ == "__main__":
    # Flask 앱을 비동기적으로 실행하기 위해 event loop를 사용합니다.
    # debug=True는 개발 시에만 사용하고, 프로덕션에서는 제거해야 합니다.
    app.run(host="0.0.0.0", port=5000, debug=True)