# 캠토리 (Camptory) - 캠핑장 추천 챗봇

## 🏕️ 프로젝트 소개

**캠토리(Camptory)**는 사용자의 질문에 맞춰 캠핑장 정보를 제공하고 추천해주는 대화형 AI 챗봇입니다. 자연어 질문을 이해하고, 사용자의 선호도(유료 캠핑장, 글램핑, 노지 캠핑 등)를 파악하여 맞춤형 답변을 생성합니다. 추천된 장소는 Tmap API와 연동된 인터랙티브 지도에 표시되며, 사용자의 현재 위치로부터 실시간 교통 상황이 반영된 경로를 확인할 수 있습니다.

이 프로젝트는 로컬 언어 모델과 RAG(Retrieval-Augmented Generation) 기술을 기반으로 하여, 특정 데이터에 기반한 정확하고 상세한 정보 제공을 목표로 합니다.

---

## ✨ 주요 기능

- **🤖 대화형 AI 챗봇**: 자연어 질문을 이해하고 캠핑 관련 정보를 제공합니다.
- **🔍 RAG (검색 증강 생성)**: 로컬 벡터 데이터베이스(ChromaDB)를 사용하여 질문과 관련된 정확한 캠핑장 정보를 검색하고 답변을 생성합니다.
- **👍 동적 맞춤 추천**: 사용자의 선호도(유료, 글램핑, 노지 등)를 파악하여 그에 맞는 캠핑 스타일을 추천하는 2단계 대화 흐름을 갖추고 있습니다.
- **🗺️ 인터랙티브 지도**: 추천된 캠핑장의 위치를 Tmap 지도에 마커로 표시합니다.
- **🚗 실시간 교통정보 경로**: 사용자의 현재 위치(또는 기본 위치)로부터 추천된 장소까지의 자동차 경로를 실시간 교통정보에 따라 다른 색상으로 시각화합니다.
- **📱 반응형 UI**: 데스크톱(가로) 및 모바일(세로) 환경에 최적화된 UI를 제공하여 어떤 기기에서든 편리하게 사용할 수 있습니다.

---

## 🛠️ 기술 스택

- **Backend**: 
  - `Python`, `Flask`
  - `LangChain`, `LangGraph` (대화 흐름 및 AI 로직 관리)
- **Frontend**:
  - `HTML`, `CSS`, `JavaScript`
  - `Tmap API v2` (지도 및 경로 시각화)
- **AI/ML**:
  - **LLM**: 로컬 모델 (`./model_ax_merge4`)
  - **Embedding Model**: `intfloat/multilingual-e5-large-instruct` (RAG를 위한 텍스트 임베딩)
- **Database**:
  - `ChromaDB` (캠핑장 정보 저장을 위한 벡터 스토어)

---

## ⚙️ 설치 및 실행 방법

1.  **저장소 복제(Clone)**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **필요 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

3.  **환경 변수 설정**
    - 프로젝트 루트 디렉터리에 `.env` 파일을 생성합니다.
    - 파일에 Tmap API 키를 추가합니다.
      ```
      TMAP_API_KEY=YOUR_TMAP_API_KEY
      ```

4.  **데이터 및 모델 확인**
    - `./data/camp_chroma_store` 경로에 ChromaDB 데이터가 있는지 확인합니다.
    - `./model_ax_merge4` 경로에 로컬 LLM 모델 파일들이 있는지 확인합니다.

5.  **서버 실행**
    ```bash
    python web_page/server.py
    ```

6.  **서비스 접속**
    - 웹 브라우저를 열고 `http://localhost:5000` 주소로 접속합니다.

---

## 📂 프로젝트 구조

```
TrailMate/
├── .env                  # 환경 변수 파일 (TMAP API 키 등)
├── requirements.txt      # Python 라이브러리 목록
├── web_page/
│   ├── server.py         # 메인 Flask 웹 서버
│   ├── filterrag_lang.py # LangGraph를 이용한 핵심 챗봇 로직
│   ├── static/
│   │   └── style.css     # 웹 UI 스타일 시트
│   └── templates/
│       └── index.html    # 메인 프론트엔드 HTML 파일
├── data/
│   └── camp_chroma_store/  # ChromaDB 벡터 데이터 저장소
└── model_ax_merge4/        # 로컬 LLM 모델 파일
```
