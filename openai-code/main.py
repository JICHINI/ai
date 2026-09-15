"""지치니 OpenAI/Pinecone FastAPI 서버."""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from pydantic import BaseModel

from llm import get_ai_response, get_vectorstore

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
    detail_concern: str


def make_document(req: UserConcernRequest) -> Document:
    return Document(
        page_content=req.concern,
        metadata={
            "user_id": req.user_id,
            "province": req.province,
            "city": req.city,
            "concern": req.concern,
            "detail_concern": req.detail_concern,
        },
    )


@app.post("/embed-user")
async def embed_user(req: UserConcernRequest):
    try:
        get_vectorstore().add_documents([make_document(req)])
        return {"status": "ok"}
    except Exception as error:
        return {"status": "error", "message": str(error)}


@app.post("/chat/sync")
def chat_sync(req: ChatRequest):
    try:
        result = ""
        for chunk in get_ai_response(req.message, req.user_province, req.user_city, req.session_id):
            result += getattr(chunk, "content", str(chunk))
        return {"answer": result}
    except Exception as error:
        return {"answer": f"오류가 발생했습니다: {error}"}


@app.put("/embed-user")
async def update_user(req: UserConcernRequest):
    try:
        vectorstore = get_vectorstore()
        vectorstore.delete(filter={"user_id": req.user_id})
        vectorstore.add_documents([make_document(req)])
        return {"status": "ok"}
    except Exception as error:
        return {"status": "error", "message": str(error)}


@app.delete("/embed-user/{user_id}")
async def delete_user(user_id: str):
    try:
        get_vectorstore().delete(filter={"user_id": user_id})
        return {"status": "ok"}
    except Exception as error:
        return {"status": "error", "message": str(error)}
