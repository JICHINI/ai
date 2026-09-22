"""OpenAI와 Pinecone을 사용하는 지치니 대화/매칭 로직."""

import os
import re

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(override=True)

# 프로세스 메모리 기반 세션 저장소입니다. 서버를 재시작하면 초기화됩니다.
store = {}
session_concern_store = {}
session_seen_store = {}


def get_secret(name: str) -> str:
    """로컬 .env와 Streamlit Cloud Secrets 모두에서 비밀값을 읽는다."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        value = None

    if not value:
        raise RuntimeError(
            f"{name}이 설정되지 않았습니다. Streamlit Cloud의 App settings → Secrets에 추가한 뒤 저장하세요."
        )
    return str(value)



def get_seen_ids(session_id):
    return session_seen_store.get(session_id, set())


def set_seen_ids(session_id, ids):
    session_seen_store[session_id] = ids


def get_session_concern(session_id):
    return session_concern_store.get(session_id, "")


def set_session_concern(session_id, concern):
    session_concern_store[session_id] = concern


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_llm():
    """Streamlit Secrets 또는 로컬 .env의 OpenAI 키로 채팅 모델을 생성한다."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=get_secret("OPENAI_API_KEY"),
    )


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large")


def get_vectorstore():
    """등록과 검색이 항상 같은 Pinecone 인덱스를 사용하도록 한다."""
    return PineconeVectorStore(
        index_name="jichini-openai-index",
        embedding=get_embeddings(),
        pinecone_api_key=get_secret("PINECONE_API_KEY"),
    )


def get_retriever(user_province: str, user_city: str):
    filter_dict = {}
    if user_province and user_province != "모든 지역":
        filter_dict["province"] = user_province
        if user_city:
            filter_dict["city"] = user_city

    search_kwargs = {"k": 15}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def get_history_retriever(user_province: str, user_city: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화를 참고해 현재 질문을 독립적인 검색 질문으로 바꿔라."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_history_aware_retriever(get_llm(), get_retriever(user_province, user_city), prompt)


def get_classification_chain():
    prompt = ChatPromptTemplate.from_template("""
너는 사용자 입력을 분류하는 AI '지치니'다.
반드시 아래 중 하나만 정확히 출력하라. 다른 말은 금지한다.
- 고민
- 인사
- 욕설
- 잡담

사용자 입력:
{input}
""")
    return prompt | get_llm()


def get_guide_chain():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
너는 사용자에게 고민을 자연스럽게 말하도록 돕는 AI '지치니'다.
설교하거나 메타 설명을 하지 말고, 인사·욕설·잡담에는 자연스럽게 고민을 말하도록 유도하라.
오직 사용자에게 하는 말만 출력하라.
"""),
        ("human", "입력 유형: {type}\n사용자 입력: {input}"),
    ])
    return prompt | get_llm()


def get_concern_score(llm, text: str) -> int:
    response = llm.invoke(f"""
다음 고민의 구체성을 0~10점으로 평가하라.
상황(누구와, 언제, 무엇을)과 감정이 포함될수록 높다.
의미 없는 문장은 0~2점, 애매한 고민은 3~4점, 구체적인 고민은 5~10점이다.
반드시 숫자만 출력하라.

문장: {text}
""")
    match = re.search(r"\d+", str(response.content))
    return int(match.group()) if match else 0


def is_more_request(text: str) -> bool:
    normalized = (text or "").replace(" ", "")
    return any(keyword in normalized for keyword in [
        "더보여", "다른사람", "추가", "더추천", "또있", "더있", "더없", "또없", "더보",
    ])


def string_to_stream(text):
    for line in (text or "").split("\n"):
        yield line + "\n"


def format_match_result(docs, seen_ids):
    """이미 보여 준 사용자를 제외하고 최대 세 명의 원문 고민을 반환한다."""
    result, count = "", 0
    for doc in docs:
        user_id = str(doc.metadata.get("user_id", ""))
        if not user_id or user_id in seen_ids:
            continue
        result += (
            f"- 사용자 ID: {user_id}\n"
            f"- 시/도: {doc.metadata.get('province', '')}\n"
            f"- 시/군: {doc.metadata.get('city', '')}\n"
            f"- 고민 내용: {doc.page_content}\n\n"
        )
        seen_ids.add(user_id)
        count += 1
        if count == 3:
            break
    return result, count


def get_ai_response(user_message, user_province, user_city, session_id="default"):
    """기존 Streamlit과 FastAPI가 사용하는 스트리밍 응답 인터페이스."""
    if is_more_request(user_message):
        previous_concern = get_session_concern(session_id)
        if not previous_concern:
            return string_to_stream("먼저 고민을 입력해 주세요.")

        docs = get_history_retriever(user_province, user_city).invoke({
            "input": previous_concern,
            "chat_history": [],
        })
        seen_ids = get_seen_ids(session_id) or set()
        result, count = format_match_result(docs, seen_ids)
        set_seen_ids(session_id, seen_ids)
        if not count:
            return string_to_stream("더 이상 추천할 사용자가 없어요.\n고민을 더 구체적으로 말해주면 다른 사람을 찾아볼게요!")
        return string_to_stream(result + "\n더 많은 사용자를 보고 싶다면 '더 보여줘'라고 말해 주세요.\n고민을 조금 더 자세히 말해주면 더 비슷한 사람을 찾아드릴 수 있어요.")

    category = str(get_classification_chain().invoke({"input": user_message}).content).strip()
    if category not in {"고민", "인사", "욕설", "잡담"}:
        category = "잡담"
    if category != "고민":
        return get_guide_chain().stream({"input": user_message, "type": category})

    previous_concern = get_session_concern(session_id)
    current_concern = f"{previous_concern}\n사용자 피드백: {user_message}" if previous_concern else user_message
    if not previous_concern:
        set_seen_ids(session_id, set())

    if get_concern_score(get_llm(), current_concern) < 5:
        guide_result = get_guide_chain().invoke({"input": current_concern, "type": "고민 부족"})
        return string_to_stream(str(guide_result.content))

    set_session_concern(session_id, current_concern)
    docs = get_history_retriever(user_province, user_city).invoke({
        "input": current_concern,
        "chat_history": [],
    })
    if not docs:
        return string_to_stream("현재 고민에 맞는 사용자가 없습니다.")

    seen_ids = get_seen_ids(session_id) or set()
    result, count = format_match_result(docs, seen_ids)
    if not count:
        return string_to_stream("현재 고민에 맞는 사용자가 없습니다.")
    set_seen_ids(session_id, seen_ids)
    return string_to_stream(result + "\n더 많은 사용자를 보고 싶다면 '더 보여줘'라고 말해 주세요.\n고민을 조금 더 자세히 말해주면 더 비슷한 사람을 찾아드릴 수 있어요.")
