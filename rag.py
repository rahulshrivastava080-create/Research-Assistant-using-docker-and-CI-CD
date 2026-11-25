from pypdf import PdfReader
import faiss
import numpy as np
import os
import pickle
import time
from openai import OpenAI
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# CACHE PATHS
# =========================
os.makedirs("cache", exist_ok=True)

FAISS_INDEX_FILE = "cache/faiss.index"
CHUNKS_FILE = "cache/chunks.pkl"


# =========================
# MODEL / PRICING
# =========================

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

# Keep pricing in one place.
# Verify these against the current OpenAI pricing before using
# the cost numbers in your README.
EMBEDDING_INPUT_PRICE_PER_1M = 0.02
LLM_INPUT_PRICE_PER_1M = 0.40
LLM_OUTPUT_PRICE_PER_1M = 1.60


# =========================
# PDF LOADER
# =========================

def load_pdf_text(file_path):

    reader = PdfReader(file_path)

    pages = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            pages.append({
                "text": text,
                "page": i + 1
            })

    return pages


# =========================
# CHUNKING
# =========================

def chunk_text(pages, chunk_size=200):

    chunks = []

    chunk_id = 0

    for page in pages:

        words = page["text"].split()

        for i in range(0, len(words), chunk_size):

            chunk = " ".join(words[i:i + chunk_size])

            if len(chunk.strip()) > 20:

                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": chunk,
                    "page": page["page"]
                })

                chunk_id += 1

    return chunks


# =========================
# SIMPLE RAG
# =========================

class SimpleRAG:

    def __init__(self):

        self.index = None
        self.chunks = []


    # =====================
    # BUILD INDEX
    # =====================

    def build_index(self, chunks):

        if not chunks:
            raise Exception("No chunks to index")

        self.chunks = chunks

        texts = [c["text"] for c in chunks]

        all_embeddings = []

        batch_size = 100

        # ---- Create embeddings ----

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )

            batch_embeddings = [
                e.embedding for e in response.data
            ]

            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(
            all_embeddings,
            dtype="float32"
        )

        # Normalize for cosine similarity

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        # Cosine similarity using inner product
        # after normalization

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        # SAVE CACHE

        faiss.write_index(
            self.index,
            FAISS_INDEX_FILE
        )

        with open(CHUNKS_FILE, "wb") as f:

            pickle.dump(
                self.chunks,
                f
            )


    # =====================
    # LOAD CACHE
    # =====================

    def load_cache(self):

        if (
            os.path.exists(FAISS_INDEX_FILE)
            and os.path.exists(CHUNKS_FILE)
        ):

            self.index = faiss.read_index(
                FAISS_INDEX_FILE
            )

            with open(CHUNKS_FILE, "rb") as f:

                self.chunks = pickle.load(f)

            return True

        return False


    # =====================
    # SEARCH
    # =====================

    def search(self, query, k=8):

        if self.index is None:

            return {
                "results": [],
                "embedding_latency": 0.0,
                "retrieval_latency": 0.0
            }


        # -------------------------
        # Query embedding timing
        # -------------------------

        embedding_start = time.perf_counter()

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )

        embedding_latency = (
            time.perf_counter() - embedding_start
        )


        query_vector = np.array(
            [response.data[0].embedding],
            dtype="float32"
        )

        # Normalize query

        faiss.normalize_L2(query_vector)


        # -------------------------
        # FAISS retrieval timing
        # -------------------------

        retrieval_start = time.perf_counter()

        distances, indices = self.index.search(
            query_vector,
            k
        )

        retrieval_latency = (
            time.perf_counter() - retrieval_start
        )


        # -------------------------
        # Build results
        # -------------------------

        results = []

        for rank, (score, i) in enumerate(
            zip(distances[0], indices[0]),
            start=1
        ):

            if i != -1 and i < len(self.chunks):

                chunk = self.chunks[i]

                results.append({

                    "chunk_id": chunk["chunk_id"],

                    "text": chunk["text"],

                    "page": chunk["page"],

                    "score": float(score),

                    "rank": rank

                })


        return {

            "results": results,

            "embedding_latency": embedding_latency,

            "retrieval_latency": retrieval_latency

        }


# =========================
# COST CALCULATION
# =========================

def calculate_llm_cost(
    input_tokens,
    output_tokens
):

    input_cost = (
        input_tokens / 1_000_000
    ) * LLM_INPUT_PRICE_PER_1M

    output_cost = (
        output_tokens / 1_000_000
    ) * LLM_OUTPUT_PRICE_PER_1M

    return input_cost + output_cost


# =========================
# ANSWER GENERATION
# =========================

def generate_answer(
    context_chunks,
    question
):

    if not context_chunks:

        return {
            "answer": "Answer not found in document.",

            "generation_latency": 0.0,

            "input_tokens": 0,

            "output_tokens": 0,

            "estimated_cost": 0.0
        }


    # -------------------------
    # Build context
    # -------------------------

    context_text = "\n\n".join(

        f"(Page {c['page']} | "
        f"Chunk {c['chunk_id']}): "
        f"{c['text']}"

        for c in context_chunks
    )


    # -------------------------
    # Prompt
    # -------------------------

    prompt = f"""
You are a helpful research assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
say exactly:

Answer not found in document.

Do not introduce information that is not supported
by the provided context.

Include the relevant page number(s) in your answer.

CONTEXT:
{context_text}

QUESTION:
{question}
"""


    # -------------------------
    # LLM timing
    # -------------------------

    generation_start = time.perf_counter()

    response = client.chat.completions.create(

        model=LLM_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    generation_latency = (
        time.perf_counter() - generation_start
    )


    # -------------------------
    # Answer
    # -------------------------

    answer = response.choices[0].message.content


    # -------------------------
    # Token usage
    # -------------------------

    usage = response.usage

    input_tokens = (
        usage.prompt_tokens
        if usage
        else 0
    )

    output_tokens = (
        usage.completion_tokens
        if usage
        else 0
    )


    # -------------------------
    # Cost
    # -------------------------

    estimated_cost = calculate_llm_cost(
        input_tokens,
        output_tokens
    )


    return {

        "answer": answer,

        "generation_latency": generation_latency,

        "input_tokens": input_tokens,

        "output_tokens": output_tokens,

        "estimated_cost": estimated_cost
    }