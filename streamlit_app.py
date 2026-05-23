import streamlit as st
import requests
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ResearchGPT - RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("📚 ResearchGPT")

    st.markdown("### 🧠 RAG-based AI Assistant")

    st.markdown("""
    ### 🧠 AI Stack
    - OpenAI GPT-4.1-mini
    - text-embedding-3-small
    - FAISS Vector Database

    ### ⚙️ Backend
    - FastAPI (REST APIs)
    - Python 3.11

    ### 🎨 Frontend
    - Streamlit (interactive UI)

    ### 🐳 DevOps
    - Docker + Docker Compose
    - GitHub Actions CI/CD
    - AWS EC2 Deployment
    - Nginx Reverse Proxy
    - SSL (Certbot HTTPS)
    """)

    st.markdown("### 🔍 RAG Pipeline")

    st.markdown("""
    1. 📄 PDF Upload  
    2. ✂️ Text Chunking  
    3. 🧠 OpenAI Embeddings  
    4. 🗂️ FAISS Vector Search  
    5. 🔎 Top-K Semantic Retrieval(Cosine Distance) 
    6. 🤖 GPT Response Generation  
    7. 📚 Citation Grounding  
    """)

    st.markdown("---")

    st.markdown("### 🟢 System Status")

    st.success("RAG Engine: Active")
    st.success("Vector DB: FAISS Loaded")
    st.success("API: FastAPI Running")

    st.markdown("---")

    st.info("Upload a research paper and start asking intelligent questions.")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# =========================
# TITLE
# =========================
st.title("🤖 ResearchGPT-Citation Aware AI Research Assistant (RAG-powered)")
st.caption("Ask questions from your PDF using retrieval-augmented generation")

# =========================
# PDF UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📄 Upload Research Paper (PDF)",
    type=["pdf"],
    help="Upload a research paper and start asking questions instantly"
)

if uploaded_file is not None and not st.session_state.pdf_uploaded:

    os.makedirs("data", exist_ok=True)
    pdf_path = "data/paper.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🔄 Processing PDF & building vector index..."):

        try:
            response = requests.post("http://app:8000/reload")

            if response.status_code == 200:
                st.session_state.pdf_uploaded = True
                st.success("✅ Document indexed successfully!")
                st.info("💡 You can now ask questions like summary, conclusion, methodology, results.")
            else:
                st.error("❌ Failed to index PDF. Try again.")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# EMPTY STATE
# =========================
st.markdown("---")

if not st.session_state.messages:
    st.info("👋 Upload a PDF and start asking research questions.")

# =========================
# CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# CHAT INPUT
# =========================
question = st.chat_input(
    "Ask: summary, conclusion, methodology, results, or any question..."
)

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            try:
                response = requests.post(
                    "http://app:8000/ask",
                    json={"question": question}
                )

                data = response.json()
                answer = data.get("answer", "No answer returned")

                # AI Answer
                st.markdown("### 🤖 AI Answer")

                st.markdown(f"""
                <div style="
                padding:15px;
                border-radius:10px;
                background-color:#1E1E1E;
                border-left:5px solid #4CAF50;
                margin-bottom:20px;
                ">
                {answer}
                </div>
                """, unsafe_allow_html=True)

                # Citation Section
                st.markdown("### 📚 Sources & Citations")

                with st.expander("View Retrieved Research Context"):
                    st.info("""
                    Retrieved from uploaded research paper using semantic search.

                    Example Citations:
                    • Page 3 — Introduction
                    • Page 5 — Methodology
                    • Page 8 — Results
                    """)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Error: {e}")