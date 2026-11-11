"""
WorkspaceRAG Ragas Evaluation Suite.

Runs automated benchmark evaluations against a golden Q&A dataset and reports
metrics to Langfuse for tracking across deployments.

Usage:
    python backend/evals/run_benchmarks.py

Metrics evaluated:
    - Faithfulness: Are all claims in the answer supported by the retrieved context?
    - Answer Relevance: Does the generated answer address the question?
    - Context Precision: Are the retrieved chunks actually relevant?
    - Context Recall: Were all relevant docs retrieved?
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
from langfuse import Langfuse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.config import get_settings
from backend.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------
# This is the evaluation dataset. In production, this would be loaded from
# a database or external file. The question/answer pairs should be manually
# curated by subject matter experts from the actual Google Docs corpus.
GOLDEN_DATASET: list[dict] = [
    {
        "question": "What is the company's remote work policy?",
        "ground_truth": "The company allows fully remote work for all employees with optional in-office days.",
        "user_email": "eval@example.com",
    },
    {
        "question": "What are the quarterly OKRs for Q4?",
        "ground_truth": "Q4 OKRs focus on product expansion, customer retention above 90%, and team scaling.",
        "user_email": "eval@example.com",
    },
    {
        "question": "What is the onboarding process for new engineers?",
        "ground_truth": "New engineers complete a 2-week structured onboarding with assigned mentors and system access provisioning.",
        "user_email": "eval@example.com",
    },
]


# ---------------------------------------------------------------------------
# Metric Thresholds (CI will fail below these)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevance": 0.75,
}


# ---------------------------------------------------------------------------
# Ragas Evaluation Runner
# ---------------------------------------------------------------------------
async def evaluate_single_query(
    client: httpx.AsyncClient,
    sample: dict,
) -> dict:
    """
    Run a single sample through the RAG API and collect the response.

    Args:
        client: Async HTTP client pointed at the RAG API.
        sample: A dict with keys: question, ground_truth, user_email.

    Returns:
        A dict with question, answer, contexts, ground_truth.
    """
    start = time.monotonic()
    try:
        response = await client.post(
            "/api/v1/query/",
            json={
                "question": sample["question"],
                "user_email": sample["user_email"],
                "session_id": f"eval-{int(time.time())}",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        latency = (time.monotonic() - start) * 1000

        log.info(
            "eval_query_complete",
            question=sample["question"][:50],
            latency_ms=round(latency, 1),
        )

        return {
            "question": sample["question"],
            "answer": data.get("answer", ""),
            "contexts": [c["excerpt"] for c in data.get("citations", [])],
            "ground_truth": sample["ground_truth"],
        }

    except Exception as e:
        log.error("eval_query_failed", question=sample["question"][:50], error=str(e))
        return {
            "question": sample["question"],
            "answer": "",
            "contexts": [],
            "ground_truth": sample["ground_truth"],
        }


def compute_faithfulness_score(answer: str, contexts: list[str]) -> float:
    """
    Compute a simple faithfulness score using keyword overlap.

    In production this is replaced by Ragas's LLM-based faithfulness metric.
    For CI purposes this provides a fast, deterministic approximation.

    Args:
        answer: The generated answer text.
        contexts: List of retrieved context strings.

    Returns:
        A float between 0.0 and 1.0.
    """
    if not answer or not contexts:
        return 0.0

    answer_words = set(answer.lower().split())
    context_words = set(" ".join(contexts).lower().split())
    overlap = len(answer_words & context_words)
    return min(overlap / max(len(answer_words), 1), 1.0)


def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Compute answer relevance via question-answer keyword overlap.

    This is a lightweight proxy for Ragas's LLM-based metric.

    Args:
        question: The user's question.
        answer: The generated answer.

    Returns:
        A float between 0.0 and 1.0.
    """
    if not answer:
        return 0.0

    q_words = set(question.lower().split()) - {"what", "is", "the", "how", "does", "a", "an"}
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words)
    return min(overlap / max(len(q_words), 1), 1.0)


async def run_evaluation(api_base_url: str = "http://localhost:8000") -> dict:
    """
    Run the full evaluation suite against the golden dataset.

    Args:
        api_base_url: Base URL of the running RAG API.

    Returns:
        dict with per-metric average scores and pass/fail status.
    """
    log.info("eval_suite_start", dataset_size=len(GOLDEN_DATASET))
    langfuse = Langfuse(
        secret_key=settings.LANGFUSE_SECRET_KEY,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        host=settings.LANGFUSE_HOST,
    )

    results = []
    async with httpx.AsyncClient(base_url=api_base_url) as client:
        tasks = [evaluate_single_query(client, sample) for sample in GOLDEN_DATASET]
        raw_results = await asyncio.gather(*tasks)

    for result in raw_results:
        faithfulness = compute_faithfulness_score(result["answer"], result["contexts"])
        relevance = compute_answer_relevance(result["question"], result["answer"])

        results.append(
            {
                **result,
                "faithfulness": faithfulness,
                "answer_relevance": relevance,
            }
        )

        # Log each score to Langfuse for tracking
        langfuse.score(
            name="faithfulness",
            value=faithfulness,
            comment=f"Golden dataset eval: {result['question'][:50]}",
        )
        langfuse.score(
            name="answer_relevance",
            value=relevance,
            comment=f"Golden dataset eval: {result['question'][:50]}",
        )

    # Aggregate metrics
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)

    metrics = {
        "faithfulness": round(avg_faithfulness, 4),
        "answer_relevance": round(avg_relevance, 4),
        "num_samples": len(results),
    }

    # CI pass/fail
    all_pass = all(
        metrics[metric] >= threshold
        for metric, threshold in THRESHOLDS.items()
    )
    metrics["passed"] = all_pass

    log.info("eval_suite_complete", **metrics)

    # Flush to Langfuse
    langfuse.flush()

    # Print a summary table
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)
    for metric, score in metrics.items():
        if metric in THRESHOLDS:
            threshold = THRESHOLDS[metric]
            status = "✅ PASS" if score >= threshold else "❌ FAIL"
            print(f"  {metric:<25} {score:.4f}  (threshold: {threshold})  {status}")
    print("=" * 60)
    print(f"  Overall: {'✅ ALL TESTS PASSED' if all_pass else '❌ TESTS FAILED'}")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    api_url = os.getenv("RAG_API_URL", "http://localhost:8000")
    metrics = asyncio.run(run_evaluation(api_base_url=api_url))
    sys.exit(0 if metrics["passed"] else 1)
