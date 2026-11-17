"""Cost and token calculation utilities for WorkspaceRAG.

Computes USD monetary cost based on configured Gemini LLM token rates,
Gemini embedding rates, and Cohere reranker search unit pricing.
"""
from __future__ import annotations

from typing import Any

from backend.core.config import get_settings


def calculate_llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate Gemini LLM cost in USD from token counts.

    Args:
        prompt_tokens: Number of input prompt tokens.
        completion_tokens: Number of output generated tokens.

    Returns:
        Estimated cost in USD rounded to 6 decimal places.
    """
    settings = get_settings()
    input_cost = (prompt_tokens / 1_000_000) * settings.GEMINI_INPUT_COST_PER_M
    output_cost = (completion_tokens / 1_000_000) * settings.GEMINI_OUTPUT_COST_PER_M
    return round(input_cost + output_cost, 6)


def calculate_cache_savings(tokens_saved: int) -> float:
    """Calculate monetary savings from a cache hit in USD.

    Assumes saved tokens would have been billed at the Gemini input rate.

    Args:
        tokens_saved: Total tokens spared by avoiding generation.

    Returns:
        Estimated USD savings rounded to 6 decimal places.
    """
    settings = get_settings()
    return round((tokens_saved / 1_000_000) * settings.GEMINI_INPUT_COST_PER_M, 6)


def calculate_total_query_cost(
    prompt_tokens: int,
    completion_tokens: int,
    query_embedding_tokens: int = 0,
    include_rerank: bool = True,
) -> dict[str, Any]:
    """Calculate itemized and total cost in USD for a full RAG query.

    Args:
        prompt_tokens: Total prompt tokens across all Gemini nodes.
        completion_tokens: Total output tokens across all Gemini nodes.
        query_embedding_tokens: Estimated tokens used to embed the search query.
        include_rerank: Whether Cohere reranking was performed.

    Returns:
        Dict with itemized costs (llm_cost, embedding_cost, rerank_cost, total_cost).
    """
    settings = get_settings()
    llm_cost = calculate_llm_cost(prompt_tokens, completion_tokens)
    embedding_cost = round(
        (query_embedding_tokens / 1_000_000) * settings.GEMINI_EMBEDDING_COST_PER_M,
        6,
    )
    rerank_cost = (
        round(settings.COHERE_RERANK_COST_PER_K / 1_000, 6) if include_rerank else 0.0
    )
    total_cost = round(llm_cost + embedding_cost + rerank_cost, 6)

    return {
        "llm_cost_usd": llm_cost,
        "embedding_cost_usd": embedding_cost,
        "rerank_cost_usd": rerank_cost,
        "total_cost_usd": total_cost,
    }
