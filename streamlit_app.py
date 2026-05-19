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
    st.title("📚 Research-GPT")
    st.markdown("### RAG-based AI Assistant")

    st.markdown("""
    **Tech Stack**
    - FastAPI
    - FAISS Vector DB
    - OpenAI Embeddings
    - Streamlit UI

    **Capabilities**
    - PDF Question Answering
    - Context-aware Retrieval
    - Source-based Answers
    """)

    st.markdown("---")
    st.info("Upload a research paper and ask questions about it.")

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
st.title("🤖 AI Research Assistant (RAG-powered)")
st.caption("Ask questions from your uploaded PDF using AI-powered retrieval")

# =========================
# PDF UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])

if uploaded_file is not None and not st.session_state.pdf_uploaded:

    os.makedirs("data", exist_ok=True)
    pdf_path = "data/paper.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("📄 Indexing document... please wait"):

        try:
            response = requests.post("http://app:8000/reload")

            if response.status_code == 200:
                st.session_state.pdf_uploaded = True
                st.success("✅ PDF indexed successfully! You can now ask questions.")
            else:
                st.error("❌ Failed to index PDF")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# CHAT HISTORY
# =========================
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# CHAT INPUT
# =========================
question = st.chat_input("Ask a question about the document...")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating... 🤔"):

            try:
                response = requests.post(
                    "http://app:8000/ask",
                    json={"question": question}
                )

                data = response.json()
                answer = data.get("answer", "No answer returned")

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Error: {e}")