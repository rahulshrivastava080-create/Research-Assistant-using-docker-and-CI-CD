import streamlit as st
import requests
import os
import html


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
    5. 🔎 Top-K Semantic Retrieval
    6. 🤖 GPT Response Generation
    7. 📚 Citation Grounding
    """)

    st.markdown("---")

    st.markdown("### 🟢 System Status")

    st.success("RAG Engine: Active")
    st.success("Vector DB: FAISS Loaded")
    st.success("API: FastAPI Running")

    st.markdown("---")

    st.info(
        "Upload a research paper and start asking intelligent questions."
    )


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
st.title("🤖 ResearchGPT-Citation Aware AI Research Assistant")

st.caption(
    "Ask questions from your PDF using retrieval-augmented generation"
)


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

            response = requests.post(
                "http://app:8000/reload"
            )

            if response.status_code == 200:

                st.session_state.pdf_uploaded = True

                st.success(
                    "✅ Document indexed successfully!"
                )

                st.info(
                    "💡 You can now ask questions like "
                    "summary, conclusion, methodology, results."
                )

            else:

                st.error(
                    "❌ Failed to index PDF. Try again."
                )

        except Exception as e:

            st.error(f"Error: {e}")


# =========================
# EMPTY STATE
# =========================
st.markdown("---")

if not st.session_state.messages:

    st.info(
        "👋 Upload a PDF and start asking research questions."
    )


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

    # -------------------------
    # User message
    # -------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # -------------------------
    # Assistant response
    # -------------------------
    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            try:

                response = requests.post(
                    "http://app:8000/ask",
                    json={
                        "question": question
                    }
                )

                # Check HTTP response
                if response.status_code != 200:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    st.stop()


                data = response.json()


                # =========================
                # EXTRACT RESPONSE
                # =========================

                answer = data.get(
                    "answer",
                    "No answer returned"
                )

                sources = data.get(
                    "sources",
                    []
                )

                metrics = data.get(
                    "metrics",
                    {}
                )


                # =========================
                # AI ANSWER
                # =========================

                st.markdown("### 🤖 Answer")

                answer_safe = html.escape(answer)

                st.markdown(
                    f"""
                    <div style="
                    padding:15px;
                    border-radius:10px;
                    background-color:#FFFFFF;
                    color:#1A1A1A;
                    border:1px solid #E5E7EB;
                    border-left:5px solid #4CAF50;
                    margin-bottom:20px;
                    line-height:1.6;
                    font-size:16px;
                    box-shadow:0px 2px 6px rgba(0,0,0,0.05);
                    ">
                    {answer_safe}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # =========================
                # SOURCES
                # =========================

                st.markdown(
                    "### 📚 Sources & Retrieval"
                )

                if sources:

                    for source in sources:

                        page = source.get(
                            "page",
                            "N/A"
                        )

                        score = source.get(
                            "score",
                            0
                        )

                        rank = source.get(
                            "rank",
                            "N/A"
                        )

                        chunk_id = source.get(
                            "chunk_id",
                            "N/A"
                        )

                        text = source.get(
                            "text",
                            ""
                        )

                        with st.expander(
                            f"📄 Page {page}  |  "
                            f"Rank {rank}  |  "
                            f"Score {score:.4f}"
                        ):

                            st.markdown(
                                f"**Chunk ID:** `{chunk_id}`"
                            )

                            st.markdown(
                                "**Retrieved Context:**"
                            )

                            st.write(text)

                else:

                    st.warning(
                        "No retrieval sources returned."
                    )


                # =========================
                # PERFORMANCE
                # =========================

                st.markdown(
                    "### ⚡ Performance"
                )

                embedding_latency = metrics.get(
                    "embedding_latency_seconds",
                    0
                )

                retrieval_latency = metrics.get(
                    "retrieval_latency_seconds",
                    0
                )

                generation_latency = metrics.get(
                    "generation_latency_seconds",
                    0
                )

                total_latency = metrics.get(
                    "total_latency_seconds",
                    0
                )


                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Embedding",
                        f"{embedding_latency:.3f} s"
                    )

                with col2:

                    st.metric(
                        "Retrieval",
                        f"{retrieval_latency:.3f} s"
                    )

                with col3:

                    st.metric(
                        "Generation",
                        f"{generation_latency:.3f} s"
                    )

                with col4:

                    st.metric(
                        "Total",
                        f"{total_latency:.3f} s"
                    )


                # =========================
                # LLM USAGE
                # =========================

                st.markdown(
                    "### 💰 LLM Usage & Cost"
                )

                input_tokens = metrics.get(
                    "input_tokens",
                    0
                )

                output_tokens = metrics.get(
                    "output_tokens",
                    0
                )

                total_tokens = metrics.get(
                    "total_tokens",
                    0
                )

                estimated_cost = metrics.get(
                    "estimated_cost_usd",
                    0
                )


                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Input Tokens",
                        f"{input_tokens:,}"
                    )

                with col2:

                    st.metric(
                        "Output Tokens",
                        f"{output_tokens:,}"
                    )

                with col3:

                    st.metric(
                        "Total Tokens",
                        f"{total_tokens:,}"
                    )

                with col4:

                    st.metric(
                        "Estimated Cost",
                        f"${estimated_cost:.6f}"
                    )


                # =========================
                # SAVE CHAT HISTORY
                # =========================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI backend. "
                    "Make sure the API container is running."
                )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )