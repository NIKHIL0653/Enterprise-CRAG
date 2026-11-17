"""Unit tests for cost calculation and token usage extraction."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.cost import (
    calculate_cache_savings,
    calculate_llm_cost,
    calculate_total_query_cost,
)


def test_calculate_llm_cost() -> None:
    """Test LLM cost calculation with known prompt and completion tokens."""
    # 1M prompt = $0.075, 1M completion = $0.30 => $0.375
    cost = calculate_llm_cost(1_000_000, 1_000_000)
    assert cost == 0.375

    # Small numbers
    cost_small = calculate_llm_cost(1000, 500)
    assert cost_small > 0.0
    assert isinstance(cost_small, float)


def test_calculate_cache_savings() -> None:
    """Test cache savings calculation."""
    savings = calculate_cache_savings(100_000)
    # 100,000 / 1,000,000 * 0.075 = 0.0075
    assert savings == 0.0075


def test_calculate_total_query_cost() -> None:
    """Test itemized total query cost calculation."""
    breakdown = calculate_total_query_cost(
        prompt_tokens=500,
        completion_tokens=200,
        query_embedding_tokens=15,
        include_rerank=True,
    )
    assert "llm_cost_usd" in breakdown
    assert "embedding_cost_usd" in breakdown
    assert "rerank_cost_usd" in breakdown
    assert "total_cost_usd" in breakdown
    assert breakdown["rerank_cost_usd"] == 0.002
    assert breakdown["total_cost_usd"] >= breakdown["rerank_cost_usd"]


@pytest.mark.asyncio
async def test_gemini_generate_with_usage_metadata() -> None:
    """Test that usage_metadata from Gemini response is correctly harvested."""
    from backend.agent.nodes import _gemini_generate_with_usage

    mock_response = MagicMock()
    mock_response.text = "Hello world response"
    mock_metadata = MagicMock()
    mock_metadata.prompt_token_count = 120
    mock_metadata.candidates_token_count = 45
    mock_metadata.total_token_count = 165
    mock_response.usage_metadata = mock_metadata

    mock_client = MagicMock()
    mock_client.generate_content_async = AsyncMock(return_value=mock_response)

    text, usage = await _gemini_generate_with_usage(mock_client, "Sample prompt text")
    assert text == "Hello world response"
    assert usage["prompt_tokens"] == 120
    assert usage["completion_tokens"] == 45
    assert usage["total_tokens"] == 165


@pytest.mark.asyncio
async def test_gemini_generate_fallback_when_metadata_absent() -> None:
    """Test fallback token estimation when usage_metadata is missing."""
    from backend.agent.nodes import _gemini_generate_with_usage

    mock_response = MagicMock()
    mock_response.text = "Hello world fallback"
    mock_response.usage_metadata = None

    mock_client = MagicMock()
    mock_client.generate_content_async = AsyncMock(return_value=mock_response)

    text, usage = await _gemini_generate_with_usage(mock_client, "Three word prompt")
    assert text == "Hello world fallback"
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 3
    assert usage["total_tokens"] == 6
