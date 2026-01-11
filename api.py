from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import ask_rag

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "AI backend running"}

@app.post("/ask")
def ask(query: Query):
    answer, citations = ask_rag(query.question)
    return {
        "question": query.question,
        "answer": answer,
        "citations": citations
    }