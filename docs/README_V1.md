# ResearchGPT — Citation-Aware RAG System

ResearchGPT is a Retrieval-Augmented Generation (RAG) application for asking questions about research papers and receiving context-grounded answers with supporting source citations.

**V1 focuses on retrieval quality, answer evaluation, citation support, API performance, and estimated LLM usage cost.** The project includes an interactive Streamlit interface, a FastAPI backend, a FAISS-based retrieval pipeline, and an initial evaluation benchmark.

## Project Status

**Current phase: V1 Evaluation and Benchmarking — Initial Baseline Established**

- **V1 — ResearchGPT RAG Application:** Research-paper question answering with semantic retrieval, context-grounded generation, and source citations.
- **V1 Evaluation Dashboard:** Streamlit-based interactive dashboard presenting retrieval metrics, generation-quality judgments, per-question results, and evaluation limitations.
- **Evaluation Benchmark:** An initial benchmark of eight research-paper questions with manually assigned relevance labels and generation-evaluation judgments.
- **Deployment:** Docker and Docker Compose support. AWS EC2 deployment was implemented previously, but the live AWS instance is currently unavailable.
- **V2 — Advanced Research Assistant:** Planned iterative development covering hybrid retrieval, reranking, multi-document research workflows, expanded evaluation, and scalability testing.

The reported V1 metrics represent an initial baseline.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Application Preview](#application-preview)
- [V1 Evaluation and Benchmarking](#v1-evaluation-and-benchmarking)
- [Evaluation Scripts](#evaluation-scripts)
- [Technology Stack](#technology-stack)
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)
- [Engineering Highlights](#engineering-highlights)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Author](#author)

## Overview

ResearchGPT combines semantic document retrieval with large language model (LLM) generation to support research-oriented question answering.

The application uses FAISS to retrieve relevant document chunks, OpenAI embeddings to represent document content, and GPT-4.1-mini to generate answers using retrieved context. A FastAPI backend exposes the question-answering API, while Streamlit provides the user interface.

In addition to implementing the RAG pipeline, V1 establishes a measurable evaluation baseline using retrieval-ranking metrics, generation-quality judgments, latency measurements, token usage, and estimated API cost.

The goal is to evaluate the application systematically and use the observed limitations to guide future improvements.

## Key Features

### RAG Application

- Research-paper question answering with source citations
- Semantic retrieval using FAISS
- OpenAI `text-embedding-3-small` embeddings
- Answer generation using GPT-4.1-mini
- FastAPI REST API
- Interactive Streamlit user interface
- Docker and Docker Compose containerization
- GitHub Actions CI/CD workflow

### V1 Evaluation and Benchmarking

- Initial benchmark containing eight research-paper questions
- Retrieval evaluation using Precision@5, Recall@5, MRR@5, Hit Rate@5, and NDCG@5
- Generation evaluation using correctness, completeness, faithfulness, citation correctness, and citation coverage
- Per-question evaluation and failure analysis
- Response-latency measurement
- Token-usage tracking
- Estimated API-cost tracking
- Streamlit V1 Evaluation Dashboard implemented locally

### Previous Deployment

- AWS EC2 deployment implemented previously
- Nginx reverse proxy
- Certbot HTTPS configuration

The previous AWS deployment is not currently available as a live service.

## System Architecture

<p align="center">
  <img src="screenshots/system-architecture.png" width="900" alt="ResearchGPT system architecture"/>
</p>

### RAG Workflow

1. **Document processing:** Extract text from research-paper PDFs.
2. **Chunking:** Divide extracted text into smaller sections suitable for retrieval.
3. **Embedding generation:** Convert text chunks into numerical vector representations.
4. **Vector indexing:** Store and search embeddings using FAISS.
5. **Question retrieval:** Retrieve the top-ranked document chunks relevant to the user's query.
6. **Answer generation:** Provide retrieved context to GPT-4.1-mini to generate a grounded response.
7. **Citation support:** Return source information alongside the generated answer.
8. **Evaluation:** Measure retrieval quality, answer quality, response latency, token usage, and estimated API cost.

## Application Preview

Add screenshots from your own running application to the locations below when available. These images are optional; the README remains usable without them.

### Main RAG Interface

<p align="center">
  <img src="screenshots/main-interface.png" width="850" alt="ResearchGPT main question-answering interface"/>
</p>

### Citation-Aware Responses

<p align="center">
  <img src="screenshots/citation-response.png" width="850" alt="ResearchGPT answer with supporting source citations"/>
</p>

### V1 Evaluation Dashboard

<p align="center">
  <img src="screenshots/v1-evaluation-dashboard.png" width="850" alt="ResearchGPT V1 retrieval and generation evaluation dashboard"/>
</p>

The V1 dashboard presents evaluation results, including retrieval-ranking metrics, generation-quality judgments, and per-question performance. The dashboard is currently implemented locally; verify its inclusion in the repository before relying on the screenshot link.

### Previous Docker and AWS Deployment

<p align="center">
  <img src="screenshots/docker-deployment.png" width="850" alt="ResearchGPT previous Docker deployment"/>
</p>

*Deployment screenshots document previous work and do not imply that a public live instance is currently available.*

## V1 Evaluation and Benchmarking

ResearchGPT V1 includes an initial benchmark of **eight research-paper questions** covering experimental design, ANOVA results, regression statistics, multi-objective optimization, tribological behaviour, and wear mechanisms.

The benchmark uses expected answers, expected source pages, manually assigned retrieval-relevance labels, and generation-evaluation judgments.

### Retrieval Evaluation

The following metrics assess the relevance and ranking of the retrieved top-five chunks.

| Metric | V1 baseline |
|---|---:|
| Precision@5 | 0.475 |
| Recall@5 | 0.958 |
| Mean Reciprocal Rank@5 (MRR@5) | 0.792 |
| Normalized Discounted Cumulative Gain@5 (NDCG@5) | 0.781 |
| Hit Rate@5 | 1.000 |

**Interpretation:** The initial benchmark indicates high retrieval coverage, with at least one relevant result retrieved for each of the eight questions. Precision@5 is lower, indicating that some retrieved chunks were not labelled relevant. Improving retrieval precision and ranking is an important direction for V2.

These results are specific to the current benchmark and relevance labels; they do not establish performance across all research papers or query types.

### Generation Evaluation

Generation quality was assessed using manual evaluation judgments for eight benchmark questions.

| Metric | V1 baseline |
|---|---:|
| Answer correctness | 87.5% |
| Answer completeness | 75.0% |
| Faithfulness | 100% |
| Citation correctness | 87.5% |
| Citation coverage | 100% |

**Interpretation:** The current judgments indicate that answer completeness and citation correctness have room for improvement. Faithfulness received a score of 100% on this small benchmark, but this does not guarantee that the system will never hallucinate or produce unsupported answers.

The evaluation set should be expanded, and generation evaluation should become more systematic and reproducible.

### Performance and API Cost

| Metric | V1 baseline |
|---|---:|
| Average end-to-end latency | 3.49 seconds |
| Average token usage | 2,464.1 tokens/query |
| Average estimated API cost | $0.001092/query |

Latency and token usage depend on the question, retrieved context, model response, and API conditions. The cost figure is an estimate based on recorded usage and the applicable pricing assumptions; actual cost may vary with model pricing and token mix.

These figures are initial benchmark measurements, not a comprehensive load or scalability test.

### Current Failure Analysis

The initial evaluation identified areas for further investigation:

- **Numerical extraction:** One benchmark question exposed difficulty extracting numerical values from a retrieved table represented as flattened text.
- **Answer completeness:** One benchmark question received an incomplete explanation of the underlying mechanism.
- **Retrieval precision:** Some retrieved chunks were not judged relevant, despite high recall.
- **Citation quality:** The current benchmark includes a citation-correctness shortfall that warrants further testing.

These observations help prioritize future work on document parsing, table handling, retrieval ranking, answer completeness, and citation validation.

## Evaluation Scripts

The evaluation directory contains the following scripts and data files, subject to the current repository version:

- `eval_dataset.json` — benchmark questions and expected answers
- `gold_relevance.json` — manually assigned relevant-chunk labels
- `answer_evaluation.json` — generation-evaluation judgments
- `run_evaluation.py` — runs benchmark questions against the local API and records results
- `analyze_v1.py` — calculates retrieval metrics and summarizes performance
- `analyze_generation_v1.py` — summarizes generation-evaluation scores

Raw benchmark result files are generated separately. They may contain retrieved document text, source information, or other research content and should be reviewed before publication.

### Running the Evaluation

Start the FastAPI application and ensure it is accessible at `http://localhost:8000`.

From the project root, run:

```bash
python evaluation/run_evaluation.py
python evaluation/analyze_v1.py
python evaluation/analyze_generation_v1.py
```

The benchmark runner sends questions to the local `/ask` endpoint and records returned answers, sources, and available metrics. The analysis scripts summarize retrieval and generation evaluation results.

**Note:** The evaluation scripts may call the OpenAI API and incur usage charges. Ensure the API key is configured before running the benchmark.

## Technology Stack

### AI and Retrieval

- OpenAI GPT-4.1-mini
- OpenAI `text-embedding-3-small`
- FAISS vector search
- Retrieval-Augmented Generation (RAG)

### Backend

- Python 3.11
- FastAPI
- REST API

### Frontend

- Streamlit

### Evaluation

- Custom benchmark dataset
- Retrieval-ranking metrics
- Manual generation-quality judgments
- Latency measurement
- Token-usage tracking
- Estimated API-cost tracking

### DevOps and Deployment

- Docker
- Docker Compose
- GitHub Actions CI/CD
- AWS EC2 — previous deployment
- Nginx reverse proxy — previous deployment
- Certbot HTTPS — previous deployment

## Local Setup

### Prerequisites

- Python 3.11
- Git
- An OpenAI API key with the required API access
- Docker Desktop, if using Docker

### 1. Clone the Repository

```bash
git clone https://github.com/rahulshrivastava080-create/Research-Assistant-using-docker-and-CI-CD.git

cd Research-Assistant-using-docker-and-CI-CD
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your own API key.

**Security:** Never commit your `.env` file or expose API keys in source code, screenshots, logs, or public issues. Keep `.env` excluded through `.gitignore`.

### 3. Install Dependencies

Optionally create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Start the Backend

From the project root:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The backend should be accessible at `http://localhost:8000`.

### 5. Start the Streamlit Interface

Open a separate terminal in the project root and run:

```bash
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`.

Use the document input expected by the application and ensure that the configured OpenAI API key is available.

### 6. Open the V1 Evaluation Dashboard

If the dashboard file is present in the repository under `pages/2_V1_Evaluation_Dashboard.py`, Streamlit should display it as an additional page in the sidebar when the application is running.

The dashboard reads the evaluation data files expected by its implementation. Ensure the required files exist locally before opening the dashboard.

## Docker Setup

Ensure Docker Desktop is running and that the `.env` file is configured.

From the project root, build and start the services using Docker Compose:

```bash
docker compose up --build -d
```

Check service status:

```bash
docker compose ps
```

View application logs:

```bash
docker compose logs --tail=100
```

With the current local port configuration, the typical addresses are:

- FastAPI backend: `http://localhost:8000`
- Streamlit interface and dashboard: `http://localhost:8501`

The exact ports and service names depend on the repository's Docker Compose configuration.

To stop the services:

```bash
docker compose down
```

The command stops and removes the Compose containers and network but does not normally remove named volumes. Review the Compose configuration before running commands that remove volumes or stored data.

## Engineering Highlights

- Built an end-to-end RAG application for research-oriented question answering.
- Integrated semantic retrieval, LLM generation, API serving, and an interactive interface.
- Implemented retrieval-ranking evaluation using manually assigned relevance labels.
- Assessed answer correctness, completeness, faithfulness, and citation quality.
- Recorded latency, token usage, and estimated API cost.
- Developed a Streamlit dashboard for inspecting V1 evaluation results.
- Containerized the application using Docker and Docker Compose.
- Implemented a GitHub Actions CI/CD workflow.
- Established a measurable baseline to guide subsequent retrieval and evaluation improvements.

## Limitations and Future Improvements

V1 is an initial implementation and evaluation baseline. The current benchmark is small, generation judgments are manually assigned, and broader reliability, robustness, and scalability remain to be established.


### V2 — Planned Development

V2 will be developed incrementally in the following areas:

- **Hybrid retrieval:** Combine dense vector retrieval with BM25 keyword search.
- **Reranking:** Rerank retrieved candidates to improve top-k relevance.
- **Multi-document research:** Support literature comparison across multiple papers.
- **Literature-review workflows:** Explore evidence-backed summaries, research-gap analysis, and comparison of reported findings.
- **Evaluation:** Expand the benchmark and investigate automated evaluation frameworks.
- **Metadata filtering:** Improve retrieval using document, section, page, and other available metadata.
- **Storage:** Explore persistent storage and alternatives such as PostgreSQL with `pgvector`.
- **Scalability testing:** Measure concurrent-request behaviour, throughput, latency, CPU, and memory usage.
- **Production safeguards:** Explore authentication, rate limiting, logging, and improved observability.

These are planned improvements, not claims of completed V2 functionality.

## Security and Research Data

- Keep API keys in environment variables and never commit them to GitHub.
- Do not publish raw benchmark outputs until retrieved text and research data have been reviewed.
- Keep generated indexes, uploaded PDFs, and other local data out of version control unless there is a clear reason to share them.
- Verify that the `.gitignore` and `.dockerignore` files exclude sensitive files as intended.
- Use appropriate access controls if deploying the application with private research documents.

## Author

**Rahul Shrivastava**

AI/ML Engineer | RAG Applications | Backend AI Systems

- **Portfolio:** [rahulshrivastava.dev](https://rahulshrivastava.dev)
- **GitHub:** [rahulshrivastava080-create](https://github.com/rahulshrivastava080-create)
- **LinkedIn:** [Rahul Shrivastava](https://www.linkedin.com/in/rahul-shrivastava-7732a5392/)

---

*ResearchGPT is an evolving engineering project. V1 establishes the current RAG implementation and initial evaluation baseline, while V2 will build on the measured limitations through iterative development and testing.*