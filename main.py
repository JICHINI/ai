from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm import get_ai_response
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_province: str
    user_city: str
    session_id: str

class UserConcernRequest(BaseModel):
    user_id: str
    province: str
    city: str
    concern: str


# Spring Boot에서 호출하는 엔드포인트
# 회원가입 시 Pinecone에 유저 임베딩 저장
@app.post("/embed-user")
async def embed_user(req: UserConcernRequest):
    try:
        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        vectorstore = PineconeVectorStore(
            index_name="jichini-user-index",
            embedding=embeddings,
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
        )
        doc = Document(
            page_content=req.concern,
            metadata={
                "user_id": req.user_id,
                "province": req.province,
                "city": req.city,
            }
        )
        vectorstore.add_documents([doc])
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat/sync")
def chat_sync(req: ChatRequest):
    try:
        stream = get_ai_response(
            user_message=req.message,
            user_province=req.user_province,
            user_city=req.user_city,
            session_id=req.session_id
        )
        result = ""
        for chunk in stream:
            try:
                result += chunk.content
            except:
                result += str(chunk)
        return {"answer": result}
    except Exception as e:
        return {"answer": f"오류가 발생했습니다: {str(e)}"}



# 내정보 수정 시 Pinecone 유저 벡터 업데이트
@app.put("/embed-user")
async def update_user(req: UserConcernRequest):
    try:
        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        vectorstore = PineconeVectorStore(
            index_name="jichini-user-index",
            embedding=embeddings,
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
        )
        # 기존 벡터 삭제
        vectorstore.delete(filter={"user_id": req.user_id})

        # 새 벡터 추가
        doc = Document(
            page_content=req.concern,
            metadata={
                "user_id": req.user_id,
                "province": req.province,
                "city": req.city,
            }
        )
        vectorstore.add_documents([doc])
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



# 회원탈퇴 시 Pinecone에서 유저 벡터 삭제
@app.delete("/embed-user/{user_id}")
async def delete_user(user_id: str):
    try:
        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        vectorstore = PineconeVectorStore(
            index_name="jichini-user-index",
            embedding=embeddings,
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
        )
        vectorstore.delete(filter={"user_id": user_id})
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}