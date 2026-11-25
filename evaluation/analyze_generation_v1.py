import json
from pathlib import Path
from statistics import mean


# --------------------------------------------------
# Path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ANSWER_EVAL_FILE = BASE_DIR / "answer_evaluation.json"


# --------------------------------------------------
# Load evaluation file
# --------------------------------------------------

with open(ANSWER_EVAL_FILE, "r", encoding="utf-8") as f:
    evaluation = json.load(f)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

metrics = [
    "correctness",
    "completeness",
    "faithfulness",
    "citation_correctness",
    "citation_coverage",
]


# --------------------------------------------------
# Per-question results
# --------------------------------------------------

print("=" * 70)
print("V1 GENERATION EVALUATION")
print("=" * 70)

for question_id in sorted(evaluation.keys()):

    scores = evaluation[question_id]

    print(f"\n{question_id}")
    print("-" * 30)

    for metric in metrics:
        print(f"{metric:25s}: {scores[metric]}")


# --------------------------------------------------
# Overall averages
# --------------------------------------------------

print("\n")
print("=" * 70)
print("OVERALL V1 GENERATION METRICS")
print("=" * 70)

for metric in metrics:

    scores = [
        evaluation[q][metric]
        for q in evaluation
    ]

    average = mean(scores)

    print(
        f"{metric:25s}: "
        f"{average:.3f} "
        f"({average * 100:.1f}%)"
    )


# --------------------------------------------------
# Pass counts
# --------------------------------------------------

print("\n")
print("=" * 70)
print("PASS COUNTS")
print("=" * 70)

total_questions = len(evaluation)

for metric in metrics:

    passed = sum(
        evaluation[q][metric] == 1
        for q in evaluation
    )

    print(
        f"{metric:25s}: "
        f"{passed}/{total_questions}"
    )


print("\nAnalysis completed successfully.")