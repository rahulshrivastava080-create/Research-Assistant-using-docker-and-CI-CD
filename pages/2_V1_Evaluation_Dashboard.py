
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_DIR / "evaluation"

RESULTS_FILE = EVAL_DIR / "results_v1.json"
GOLD_FILE = EVAL_DIR / "gold_relevance.json"
GENERATION_FILE = EVAL_DIR / "answer_evaluation.json"

K = 5


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ResearchGPT V1 Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ResearchGPT — V1 Evaluation Dashboard")
st.caption(
    "Benchmark evaluation for retrieval quality, answer quality, "
    "latency, token usage, and estimated API cost."
)


# --------------------------------------------------
# Load JSON safely
# --------------------------------------------------
@st.cache_data
def load_json(path_string):
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


try:
    results = load_json(str(RESULTS_FILE))
    gold_data = load_json(str(GOLD_FILE))
    generation_data = load_json(str(GENERATION_FILE))
except (FileNotFoundError, json.JSONDecodeError) as exc:
    st.error(f"Unable to load evaluation data: {exc}")
    st.info(
        "Check that results_v1.json, gold_relevance.json, "
        "and answer_evaluation.json exist in the evaluation folder."
    )
    st.stop()


# --------------------------------------------------
# Retrieval evaluation helpers
# --------------------------------------------------
def get_gold_chunks(question_id):
    entry = gold_data.get(question_id, [])

    if isinstance(entry, dict):
        entry = entry.get("relevant_chunks", [])

    if not isinstance(entry, list):
        return set()

    return set(entry)


def calculate_retrieval_metrics(retrieved_chunks, relevant_chunks):
    retrieved = retrieved_chunks[:K]
    relevant = set(relevant_chunks)

    if not relevant:
        return {
            "Precision@5": None,
            "Recall@5": None,
            "MRR@5": None,
            "Hit Rate@5": None,
            "NDCG@5": None,
            "First Relevant Rank": None,
        }

    hits = [chunk for chunk in retrieved if chunk in relevant]
    first_rank = next(
        (
            rank
            for rank, chunk in enumerate(retrieved, start=1)
            if chunk in relevant
        ),
        None,
    )

    precision = len(hits) / K
    recall = len(hits) / len(relevant)
    mrr = 1 / first_rank if first_rank is not None else 0.0
    hit_rate = 1.0 if first_rank is not None else 0.0

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk in enumerate(retrieved, start=1)
        if chunk in relevant
    )

    ideal_hits = min(len(relevant), K)
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return {
        "Precision@5": precision,
        "Recall@5": recall,
        "MRR@5": mrr,
        "Hit Rate@5": hit_rate,
        "NDCG@5": ndcg,
        "First Relevant Rank": first_rank,
    }


def get_metric(metrics, possible_keys, default=0.0):
    for key in possible_keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return default


# --------------------------------------------------
# Build per-question retrieval results
# --------------------------------------------------
retrieval_rows = []

for result in results:
    question_id = result.get("id")

    if not question_id:
        continue

    sources = result.get("sources", [])
    retrieved_chunks = [
        source.get("chunk_id")
        for source in sources
        if source.get("chunk_id") is not None
    ]

    relevant_chunks = get_gold_chunks(question_id)
    retrieval_metrics = calculate_retrieval_metrics(
        retrieved_chunks,
        relevant_chunks,
    )

    runtime = result.get("metrics", {})

    retrieval_rows.append(
        {
            "Question": question_id,
            "Retrieval Metrics": retrieval_metrics,
            "Latency (s)": get_metric(
                runtime,
                ["total_latency_seconds", "latency_seconds", "latency"],
            ),
            "Tokens": get_metric(
                runtime,
                ["total_tokens", "tokens"],
            ),
            "Estimated Cost ($)": get_metric(
                runtime,
                ["estimated_cost_usd", "cost_usd", "cost"],
            ),
            "Retrieved Chunks": len(sources),
        }
    )

retrieval_df = pd.DataFrame(retrieval_rows)

if retrieval_df.empty:
    st.error("No valid question records were found in results_v1.json.")
    st.stop()

metric_names = [
    "Precision@5",
    "Recall@5",
    "MRR@5",
    "Hit Rate@5",
    "NDCG@5",
]

for metric in metric_names:
    retrieval_df[metric] = retrieval_df["Retrieval Metrics"].apply(
        lambda values: values[metric]
    )


# --------------------------------------------------
# Build generation evaluation results
# --------------------------------------------------
generation_metrics = [
    "correctness",
    "completeness",
    "faithfulness",
    "citation_correctness",
    "citation_coverage",
]

generation_rows = []

for question_id, scores in generation_data.items():
    if not isinstance(scores, dict):
        continue

    row = {"Question": question_id}

    for metric in generation_metrics:
        value = scores.get(metric)
        row[metric.replace("_", " ").title()] = (
            float(value) if value is not None else None
        )

    generation_rows.append(row)

generation_df = pd.DataFrame(generation_rows)

if not generation_df.empty:
    generation_columns = [
        metric.replace("_", " ").title()
        for metric in generation_metrics
    ]

    generation_summary = (
        generation_df[generation_columns].mean(numeric_only=True) * 100
    )
else:
    generation_summary = pd.Series(dtype=float)


# --------------------------------------------------
# Overview
# --------------------------------------------------
st.subheader("Benchmark overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Retrieval Questions", len(retrieval_df))
col2.metric("Generation Assessments", len(generation_df))

avg_latency = retrieval_df["Latency (s)"].mean()
avg_tokens = retrieval_df["Tokens"].mean()
avg_cost = retrieval_df["Estimated Cost ($)"].mean()

col3.metric("Average Latency", f"{avg_latency:.2f} s")
col4.metric("Average Estimated Cost", f"${avg_cost:.6f}")

st.caption(
    f"Average token usage per recorded question: {avg_tokens:,.1f}. "
    "Latency and cost depend on the recorded benchmark run."
)


# --------------------------------------------------
# Tabs
# --------------------------------------------------
retrieval_tab, generation_tab, questions_tab, notes_tab = st.tabs(
    [
        "Retrieval Quality",
        "Generation Quality",
        "Per-Question Results",
        "Limitations",
    ]
)


# --------------------------------------------------
# Retrieval tab
# --------------------------------------------------
with retrieval_tab:
    st.subheader("Retrieval quality metrics")

    metric_cols = st.columns(5)

    for col, metric in zip(metric_cols, metric_names):
        value = retrieval_df[metric].mean()
        col.metric(metric, f"{value:.1%}" if pd.notna(value) else "N/A")

    st.caption(
        "Precision@5 uses a fixed denominator of five. "
        "Recall@5 measures the fraction of gold-relevant chunks retrieved. "
        "MRR@5 rewards finding the first relevant chunk early; "
        "NDCG@5 measures the ranking of relevant chunks."
    )

    st.bar_chart(
        retrieval_df.set_index("Question")[metric_names] * 100,
        y_label="Score (%)",
    )

    st.markdown("**Runtime and resource measurements**")

    runtime_df = retrieval_df[
        [
            "Question",
            "Latency (s)",
            "Tokens",
            "Estimated Cost ($)",
            "Retrieved Chunks",
        ]
    ].copy()

    st.dataframe(runtime_df, use_container_width=True, hide_index=True)


# --------------------------------------------------
# Generation tab
# --------------------------------------------------
with generation_tab:
    st.subheader("Answer quality and citation metrics")

    if generation_summary.empty:
        st.warning("No generation assessment scores were found.")
    else:
        gen_cols = st.columns(5)

        for col, metric in zip(
            gen_cols,
            generation_summary.index,
        ):
            col.metric(metric, f"{generation_summary[metric]:.1f}%")

        st.bar_chart(
            generation_summary,
            y_label="Pass rate (%)",
        )

        st.markdown("**Manual assessment by question**")

        display_generation = generation_df.copy()

        for column in display_generation.columns:
            if column != "Question":
                display_generation[column] = (
                    display_generation[column] * 100
                )

        st.dataframe(
            display_generation,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Scores are binary manual assessments (0 = fail, 1 = pass). "
            "Percentages are pass rates over the assessed questions."
        )


# --------------------------------------------------
# Per-question tab
# --------------------------------------------------
with questions_tab:
    st.subheader("Combined benchmark results")

    combined = retrieval_df[
        [
            "Question",
            *metric_names,
            "Latency (s)",
            "Tokens",
            "Estimated Cost ($)",
        ]
    ].copy()

    if not generation_df.empty:
        combined = combined.merge(
            generation_df,
            on="Question",
            how="left",
        )

    st.dataframe(
        combined,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "This table shows aggregate metrics and assessment scores only; "
        "it does not expose retrieved paper text."
    )


# --------------------------------------------------
# Limitations tab
# --------------------------------------------------
with notes_tab:
    st.subheader("Known V1 limitations")

    st.markdown(
        """
        - **Small benchmark:** the current evaluation uses eight questions.
        - **Manual generation assessment:** scores can depend on the evaluator
          and the selected benchmark questions.
        - **Retrieval relevance labels:** results depend on the quality and
          completeness of the gold-relevance annotations.
        - **Table extraction:** Q05 previously showed difficulty extracting
          numerical values from a flattened table chunk.
        - **Answer completeness:** Q06 previously received an incomplete
          mechanism explanation.
        - **No universal guarantee:** high faithfulness or citation coverage
          on this benchmark does not establish zero hallucinations.
        - **Variable runtime:** latency and estimated cost can change between
          runs because of network, model, and workload differences.
        """
    )

    st.info(
        "V2 can target table-aware extraction, hybrid retrieval, reranking, "
        "and a broader regression benchmark."
    )


st.divider()
st.caption(
    "ResearchGPT V1 | Evaluation dashboard | "
    "Results reflect the benchmark files currently stored in evaluation/."
)
