"""Unit tests for offline evaluation suite utilities and dataset contracts."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.evals.generate_dataset import (
    extract_json,
    generate_dataset,
    get_mock_chunks,
)
from backend.evals.run_benchmarks import (
    GOLDEN_DATASET,
    compute_answer_relevance,
    compute_faithfulness_score,
    evaluate_single_query,
    load_evaluation_dataset,
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


def test_load_evaluation_dataset_from_disk(tmp_path) -> None:
    """Test loading a custom JSONL evaluation dataset."""
    test_jsonl = tmp_path / "custom_eval.jsonl"
    test_jsonl.write_text(
        '{"category": "test", "question": "What is Q?", "ground_truth": "Ans A", "user_email": "u@a.com"}\n'
        '{"category": "test2", "question": "What is Q2?", "ground_truth": "Ans B", "user_email": "u@a.com"}\n',
        encoding="utf-8",
    )
    dataset = load_evaluation_dataset(test_jsonl)
    assert len(dataset) == 2
    assert dataset[0]["question"] == "What is Q?"
    assert dataset[1]["ground_truth"] == "Ans B"


def test_load_evaluation_dataset_fallback_missing_file() -> None:
    """Test fallback when specified dataset path does not exist."""
    dataset = load_evaluation_dataset("non_existent_file_path_12345.jsonl")
    assert len(dataset) >= 5
    assert dataset[0]["category"] in {"policy", "stipend", "strategy", "engineering", "security", "negative_fallback"}


def test_generate_dataset_extract_json() -> None:
    """Test JSON extraction from model response prose."""
    raw = 'Here is the response:\n```json\n{"question": "How to deploy?", "ground_truth": "Run script", "category": "engineering"}\n```'
    parsed = extract_json(raw)
    assert parsed["question"] == "How to deploy?"
    assert parsed["category"] == "engineering"

    with pytest.raises(ValueError, match="No JSON object"):
        extract_json("No json here")


@pytest.mark.asyncio
async def test_generate_dataset_dry_run(tmp_path) -> None:
    """Test offline synthetic dataset generation."""
    out_file = tmp_path / "synthetic.jsonl"
    mock_chunks = get_mock_chunks(3)
    assert len(mock_chunks) == 3

    items = await generate_dataset(limit=2, output_path=str(out_file), dry_run=True)
    assert len(items) == 2
    assert out_file.exists()

    # Verify file content
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first_item = lines[0]
    assert "question" in first_item
    assert "ground_truth" in first_item

