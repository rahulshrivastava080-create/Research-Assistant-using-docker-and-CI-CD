import json
from pathlib import Path
from statistics import mean


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = BASE_DIR / "results_v1.json"
GOLD_FILE = BASE_DIR / "gold_relevance.json"


# ============================================================
# LOAD DATA
# ============================================================

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

with open(GOLD_FILE, "r", encoding="utf-8") as f:
    gold_data = json.load(f)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_gold_chunks(question_id):
    """
    Supports:

    {
        "Q01": {
            "relevant_chunks": ["chunk_11", "chunk_13"]
        }
    }

    or:

    {
        "Q01": ["chunk_11", "chunk_13"]
    }
    """

    item = gold_data.get(question_id, {})

    if isinstance(item, dict):
        return set(item.get("relevant_chunks", []))

    if isinstance(item, list):
        return set(item)

    return set()


def precision_at_k(retrieved, relevant, k=5):
    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant
    )

    return relevant_retrieved / len(top_k)


def recall_at_k(retrieved, relevant, k=5):
    if not relevant:
        return 0.0

    top_k = retrieved[:k]

    relevant_retrieved = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant
    )

    return relevant_retrieved / len(relevant)


def reciprocal_rank(retrieved, relevant, k=5):
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def hit_rate_at_k(retrieved, relevant, k=5):
    return 1.0 if any(
        chunk_id in relevant
        for chunk_id in retrieved[:k]
    ) else 0.0


def ndcg_at_k(retrieved, relevant, k=5):
    """
    Binary relevance NDCG.

    Relevant chunk = 1
    Irrelevant chunk = 0
    """

    top_k = retrieved[:k]

    dcg = 0.0

    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in relevant:
            dcg += 1 / __import__("math").log2(rank + 1)

    ideal_relevant = min(len(relevant), k)

    idcg = 0.0

    for rank in range(1, ideal_relevant + 1):
        idcg += 1 / __import__("math").log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# EVALUATION
# ============================================================

evaluation_rows = []

for result in results:

    question_id = result["id"]

    retrieved_chunks = [
        source["chunk_id"]
        for source in result.get("sources", [])
    ]

    relevant_chunks = get_gold_chunks(question_id)

    p5 = precision_at_k(
        retrieved_chunks,
        relevant_chunks,
        k=5
    )

    r5 = recall_at_k(
        retrieved_chunks,
        relevant_chunks,
        k=5
    )

    mrr5 = reciprocal_rank(
        retrieved_chunks,
        relevant_chunks,
        k=5
    )

    hit5 = hit_rate_at_k(
        retrieved_chunks,
        relevant_chunks,
        k=5
    )

    ndcg5 = ndcg_at_k(
        retrieved_chunks,
        relevant_chunks,
        k=5
    )

    first_rank = None

    for rank, chunk_id in enumerate(retrieved_chunks, start=1):
        if chunk_id in relevant_chunks:
            first_rank = rank
            break

    metrics = result.get("metrics", {})

    evaluation_rows.append({
        "id": question_id,
        "precision@5": p5,
        "recall@5": r5,
        "mrr@5": mrr5,
        "hit_rate@5": hit5,
        "ndcg@5": ndcg5,
        "first_relevant_rank": first_rank,
        "latency": metrics.get("total_latency_seconds", 0),
        "tokens": metrics.get("total_tokens", 0),
        "cost": metrics.get("estimated_cost_usd", 0)
    })


# ============================================================
# DISPLAY PER-QUESTION RESULTS
# ============================================================

print("\n" + "=" * 80)
print("V1 RETRIEVAL EVALUATION")
print("=" * 80)

for row in evaluation_rows:

    print(f"\n{row['id']}")
    print("-" * 40)

    print(f"Precision@5        : {row['precision@5']:.3f}")
    print(f"Recall@5           : {row['recall@5']:.3f}")
    print(f"MRR@5              : {row['mrr@5']:.3f}")
    print(f"Hit Rate@5         : {row['hit_rate@5']:.3f}")
    print(f"NDCG@5             : {row['ndcg@5']:.3f}")
    print(f"First relevant rank: {row['first_relevant_rank']}")
    print(f"Latency            : {row['latency']:.4f} sec")
    print(f"Tokens             : {row['tokens']}")
    print(f"Cost               : ${row['cost']:.6f}")


# ============================================================
# OVERALL RESULTS
# ============================================================

avg_precision = mean(
    row["precision@5"]
    for row in evaluation_rows
)

avg_recall = mean(
    row["recall@5"]
    for row in evaluation_rows
)

avg_mrr = mean(
    row["mrr@5"]
    for row in evaluation_rows
)

avg_hit_rate = mean(
    row["hit_rate@5"]
    for row in evaluation_rows
)

avg_ndcg = mean(
    row["ndcg@5"]
    for row in evaluation_rows
)

avg_latency = mean(
    row["latency"]
    for row in evaluation_rows
)

avg_tokens = mean(
    row["tokens"]
    for row in evaluation_rows
)

avg_cost = mean(
    row["cost"]
    for row in evaluation_rows
)


# ============================================================
# PRINT OVERALL BASELINE
# ============================================================

print("\n" + "=" * 80)
print("V1 OVERALL BASELINE")
print("=" * 80)

print(f"\nPrecision@5       : {avg_precision:.3f}")
print(f"Recall@5          : {avg_recall:.3f}")
print(f"MRR@5             : {avg_mrr:.3f}")
print(f"Hit Rate@5        : {avg_hit_rate:.3f}")
print(f"NDCG@5            : {avg_ndcg:.3f}")

print("\nPerformance")
print("-" * 40)

print(f"Average latency   : {avg_latency:.4f} sec")
print(f"Average tokens    : {avg_tokens:.1f}")
print(f"Average cost      : ${avg_cost:.6f}")

print("\n" + "=" * 80)
print("V1 EVALUATION COMPLETE")
print("=" * 80)