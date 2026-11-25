
import json
import requests
from pathlib import Path
from datetime import datetime


# -----------------------------
# Configuration
# -----------------------------

API_URL = "http://localhost:8000/ask"

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "eval_dataset.json"
OUTPUT_FILE = BASE_DIR / "results_v1.json"


# -----------------------------
# Load evaluation dataset
# -----------------------------

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)


print(f"Loaded {len(dataset)} evaluation questions.")
print("-" * 60)


# -----------------------------
# Run evaluation
# -----------------------------

results = []

for i, item in enumerate(dataset, start=1):

    question_id = item["id"]
    question = item["question"]

    print(f"[{i}/{len(dataset)}] Running {question_id}")
    print(f"Question: {question}")

    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        # -----------------------------
        # Store complete evaluation result
        # -----------------------------

        result = {
            "id": question_id,
            "question": question,
            "category": item.get("category"),

            # Ground-truth information
            "expected_answer": item.get("expected_answer"),
            "expected_pages": item.get("expected_pages", []),

            # Generated RAG result
            "answer": data.get("answer"),
            "sources": data.get("sources", []),

            # Performance metrics
            "metrics": data.get("metrics", {}),

            # Version information
            "evaluation_version": "V1",
            "timestamp": datetime.now().isoformat()
        }

        results.append(result)

        # -----------------------------
        # Print metrics
        # -----------------------------

        metrics = data.get("metrics", {})

        print(
            f"Latency: "
            f"{metrics.get('total_latency_seconds', 'N/A')} sec"
        )

        print(
            f"Cost: "
            f"${metrics.get('estimated_cost_usd', 'N/A')}"
        )

        print("✓ Completed")
        print("-" * 60)

    except requests.exceptions.RequestException as e:

        print(f"✗ API request failed: {e}")

        # Still save the ground-truth information
        # even if the API request fails.

        results.append({
            "id": question_id,
            "question": question,
            "category": item.get("category"),

            "expected_answer": item.get("expected_answer"),
            "expected_pages": item.get("expected_pages", []),

            "error": str(e),

            "evaluation_version": "V1",
            "timestamp": datetime.now().isoformat()
        })


# -----------------------------
# Save results
# -----------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# -----------------------------
# Completion message
# -----------------------------

print("=" * 60)
print("V1 evaluation completed.")
print("Results saved to:")
print(OUTPUT_FILE)
print("=" * 60)
