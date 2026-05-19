from pypdf import PdfReader
import faiss
import numpy as np
import os
import pickle
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

    for page in pages:
        words = page["text"].split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])

            if len(chunk.strip()) > 20:
                chunks.append({
                    "text": chunk,
                    "page": page["page"]
                })

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

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )

            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # SAVE CACHE
        faiss.write_index(self.index, FAISS_INDEX_FILE)

        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)

    # =====================
    # LOAD CACHE
    # =====================
    def load_cache(self):
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            self.index = faiss.read_index(FAISS_INDEX_FILE)

            with open(CHUNKS_FILE, "rb") as f:
                self.chunks = pickle.load(f)

            return True

        return False

    # =====================
    # SEARCH
    # =====================
    def search(self, query, k=3):

        if self.index is None:
            return []   # SAFE fallback instead of crash

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )

        query_vector = np.array(
            [response.data[0].embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(query_vector, k)

        results = []

        for i in indices[0]:
            if i != -1 and i < len(self.chunks):
                results.append(self.chunks[i])

        return results


# =========================
# ANSWER GENERATION
# =========================
def generate_answer(context_chunks, question):

    if not context_chunks:
        return "Answer not found in document."

    context_text = "\n\n".join(
        f"(Page {c['page']}): {c['text']}"
        for c in context_chunks
    )

    prompt = f"""
You are a helpful research assistant.

Answer ONLY using the provided context.
If not found, say: "Answer not found in document."

CONTEXT:
{context_text}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content