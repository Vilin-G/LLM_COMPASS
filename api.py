from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from fastapi.responses import FileResponse

from app import (
    build_llm,
    build_embeddings,
    load_and_split,
    get_vector_store,
    build_graph,
)

# 服务启动时初始化一次（模块被导入时执行）
llm = build_llm()
embeddings = build_embeddings()
split_docs = load_and_split()
vector_store = get_vector_store(split_docs, embeddings)
graph = build_graph(vector_store, llm)

app = FastAPI(title="学习笔记 RAG 问答 API")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke({"question": request.question})
    return AskResponse(answer=result["answer"])
