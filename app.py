from fastapi import FastAPI
from pydantic import BaseModel
import os

from rag import (
    SimpleRAG,
    load_pdf_text,
    chunk_text,
    generate_answer
)

app = FastAPI()
rag = SimpleRAG()


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup_event():

    loaded = rag.load_cache()

    if loaded:
        print("✅ Loaded FAISS cache")

    else:
        print("⚠️ No cache found. Waiting for /reload or PDF.")

        pdf_path = "data/paper.pdf"

        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:

            print("📄 Building index from PDF...")

            pages = load_pdf_text(pdf_path)
            chunks = chunk_text(pages)

            rag.build_index(chunks)

            print("✅ Index built successfully")

        else:
            print("❌ No PDF found at startup")


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"message": "RAG API Running"}


# =========================
# REQUEST MODEL
# =========================
class Query(BaseModel):
    question: str


# =========================
# ASK ENDPOINT (SAFE)
# =========================
@app.post("/ask")
def ask(query: Query):

    if rag.index is None:
        return {"error": "Index not ready. Please upload PDF and reload."}

    relevant_chunks = rag.search(query.question)

    answer = generate_answer(relevant_chunks, query.question)

    return {
        "question": query.question,
        "answer": answer,
        "sources": relevant_chunks
    }


# =========================
# RELOAD ENDPOINT
# =========================
@app.post("/reload")
def reload_rag():

    pdf_path = "data/paper.pdf"

    if not os.path.exists(pdf_path):
        return {"error": "PDF not found"}

    if os.path.getsize(pdf_path) == 0:
        return {"error": "PDF is empty"}

    pages = load_pdf_text(pdf_path)
    chunks = chunk_text(pages)

    rag.build_index(chunks)

    return {
        "message": "RAG rebuilt successfully"
    }