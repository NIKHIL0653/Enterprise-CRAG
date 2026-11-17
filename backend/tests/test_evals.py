"""Unit tests for offline evaluation suite utilities and dataset contracts."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.evals.run_benchmarks import (
    GOLDEN_DATASET,
    compute_answer_relevance,
    compute_faithfulness_score,
    evaluate_single_query,
)


def test_golden_dataset_structure() -> None:
    """Validate that every sample in the golden dataset adheres to the schema."""
    assert len(GOLDEN_DATASET) >= 5
    required_keys = {"question", "ground_truth", "user_email", "category"}

    for idx, sample in enumerate(GOLDEN_DATASET):
        missing = required_keys - set(sample.keys())
        assert not missing, f"Sample #{idx} missing required keys: {missing}"
        assert len(sample["question"].strip()) > 5
        assert len(sample["ground_truth"].strip()) > 5
        assert "@" in sample["user_email"]


def test_compute_faithfulness_score() -> None:
    """Test keyword-overlap faithfulness score calculation."""
    # Complete match
    contexts = ["The company remote work policy allows full-time remote."]
    answer = "The company remote work policy allows full-time remote."
    score = compute_faithfulness_score(answer, contexts)
    assert score == 1.0

    # Partial match
    answer_partial = "The company allows full-time remote in Antarctica."
    score_partial = compute_faithfulness_score(answer_partial, contexts)
    assert 0.0 < score_partial < 1.0

    # Empty inputs
    assert compute_faithfulness_score("", contexts) == 0.0
    assert compute_faithfulness_score(answer, []) == 0.0


def test_compute_answer_relevance() -> None:
    """Test keyword-overlap answer relevance calculation."""
    question = "What is the annual equipment stipend?"
    answer = "The annual equipment stipend is $1500."
    score = compute_answer_relevance(question, answer)
    assert score > 0.5

    # Completely irrelevant answer
    answer_irrelevant = "Bananas are yellow fruit grown in tropical climates."
    assert compute_answer_relevance(question, answer_irrelevant) == 0.0

    # Empty answer
    assert compute_answer_relevance(question, "") == 0.0


@pytest.mark.asyncio
async def test_evaluate_single_query_sse_parser() -> None:
    """Test that evaluate_single_query correctly parses an SSE stream."""
    sample = {
        "question": "What is the remote work policy?",
        "ground_truth": "Full time remote allowed.",
        "user_email": "test@acme.com",
    }

    sse_lines = [
        "data: Full\n",
        "data: time\n",
        "data: remote\n",
        "data: allowed.\n",
        "data: [DONE]\n",
        'data: {"citations": [{"excerpt": "Remote policy excerpt"}], "total_tokens": 120, "cost_usd": 0.000015, "latency_ms": 150}\n',
    ]

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = MagicMock()

    class MockStreamContext:
        async def __aenter__(self):
            return mock_response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    mock_client.stream = MagicMock(return_value=MockStreamContext())

    result = await evaluate_single_query(mock_client, sample)
    assert result["answer"] == "Full time remote allowed."
    assert result["contexts"] == ["Remote policy excerpt"]
    assert result["total_tokens"] == 120
    assert result["cost_usd"] == 0.000015
