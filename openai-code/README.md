# 지치니 OpenAI 버전

기존 지치니의 Streamlit 화면과 FastAPI API(`POST/PUT/DELETE /embed-user`, `POST /chat/sync`)를 OpenAI와 Pinecone으로 동작하게 만든 버전입니다.

## 설정 및 실행

```bash
cd /Users/kshi3430/jichini/finally_jichini_ai/openai-code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`에 OpenAI와 Pinecone API 키를 입력하고, 기존 사용자 벡터가 든 Pinecone 인덱스 이름을 `PINECONE_INDEX_NAME`에 지정하세요. 기존 Upstage 임베딩으로 만든 인덱스는 OpenAI 임베딩과 차원이 달라 호환되지 않을 수 있으므로, OpenAI 버전에서는 사용자 데이터를 다시 임베딩해야 합니다.

Streamlit 화면 실행:

```bash
streamlit run chat.py
```

FastAPI 실행:

```bash
python -m uvicorn main:app --reload --port 5000
```
