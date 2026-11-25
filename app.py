from fastapi import FastAPI
from pydantic import BaseModel
import os
import time

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
    return {
        "message": "RAG API Running"
    }


# =========================
# REQUEST MODEL
# =========================
class Query(BaseModel):
    question: str


# =========================
# ASK ENDPOINT
# =========================
@app.post("/ask")
def ask(query: Query):

    if rag.index is None:
        return {
            "error": "Index not ready. Please upload PDF and reload."
        }

    # ---------------------------------
    # Total request timer
    # ---------------------------------
    total_start = time.perf_counter()

    # ---------------------------------
    # Retrieval
    # ---------------------------------
    retrieval_result = rag.search(query.question)

    relevant_chunks = retrieval_result["results"]

    embedding_latency = retrieval_result["embedding_latency"]
    retrieval_latency = retrieval_result["retrieval_latency"]

    # ---------------------------------
    # Generation
    # ---------------------------------
    generation_result = generate_answer(
        relevant_chunks,
        query.question
    )

    answer = generation_result["answer"]

    generation_latency = generation_result["generation_latency"]
    input_tokens = generation_result["input_tokens"]
    output_tokens = generation_result["output_tokens"]
    estimated_cost = generation_result["estimated_cost"]

    # ---------------------------------
    # Total latency
    # ---------------------------------
    total_latency = time.perf_counter() - total_start

    # ---------------------------------
    # Return complete observability data
    # ---------------------------------
    return {
        "question": query.question,

        "answer": answer,

        "sources": relevant_chunks,

        "metrics": {
            "embedding_latency_seconds": round(
                embedding_latency, 4
            ),

            "retrieval_latency_seconds": round(
                retrieval_latency, 4
            ),

            "generation_latency_seconds": round(
                generation_latency, 4
            ),

            "total_latency_seconds": round(
                total_latency, 4
            ),

            "input_tokens": input_tokens,

            "output_tokens": output_tokens,

            "total_tokens": input_tokens + output_tokens,

            "estimated_cost_usd": round(
                estimated_cost, 6
            )
        }
    }


# =========================
# RELOAD ENDPOINT
# =========================
@app.post("/reload")
def reload_rag():

    pdf_path = "data/paper.pdf"

    if not os.path.exists(pdf_path):
        return {
            "error": "PDF not found"
        }

    if os.path.getsize(pdf_path) == 0:
        return {
            "error": "PDF is empty"
        }

    pages = load_pdf_text(pdf_path)
    chunks = chunk_text(pages)

    rag.build_index(chunks)

    return {
        "message": "RAG rebuilt successfully",
        "chunks": len(chunks)
    }