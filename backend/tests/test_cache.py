"""Unit tests for QueryCacheManager and normalization utilities."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.services.cache import (
    QueryCacheManager,
    _canonical_normalize,
    _sha256_hex,
)


def test_canonical_normalize_prefixes_and_articles() -> None:
    """Test prefix removal, article removal, and whitespace collapsing."""
    # Prefix and article stripping
    assert _canonical_normalize("What is a transformer?") == "transformer"
    assert _canonical_normalize("what is the transformer?") == "transformer"
    assert _canonical_normalize("Explain the remote work policy?") == "remote work policy"
    assert _canonical_normalize("Tell me about an equipment stipend!") == "equipment stipend"

    # Preserves core words when no prefix
    assert _canonical_normalize("Quarterly OKRs 2026") == "quarterly okrs 2026"

    # Handles edge cases without crashing
    assert _canonical_normalize("   ") == ""
    assert _canonical_normalize("What is?") == "what is?"


def test_sha256_hex_isolation() -> None:
    """Ensure user email isolation in SHA-256 hash generation."""
    hash_user1 = _sha256_hex("alice@acme.com", "transformer")
    hash_user2 = _sha256_hex("bob@acme.com", "transformer")
    hash_user1_diff_q = _sha256_hex("alice@acme.com", "remote work")

    assert hash_user1 != hash_user2
    assert hash_user1 != hash_user1_diff_q
    assert len(hash_user1) == 64


@pytest.mark.asyncio
async def test_cache_lookup_tier1_exact_hit() -> None:
    """Test Tier-1 exact match returns immediately without invoking embedder."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock()
    mock_pool.acquire.return_value = mock_acquire

    # Return cached record on exact lookup
    mock_conn.fetchrow = AsyncMock(
        return_value={
            "answer": "A transformer is a deep learning architecture.",
            "citations": json.dumps([{"excerpt": "Transformer paper"}]),
            "tokens_used": 150,
        }
    )

    mock_embedder = MagicMock()
    mock_embedder.embed_query = AsyncMock()

    manager = QueryCacheManager(pool=mock_pool, embedder=mock_embedder)
    result = await manager.lookup(question="What is a transformer?", user_email="alice@acme.com")

    assert result is not None
    assert result["cache_type"] == "exact"
    assert "deep learning architecture" in result["answer"]
    assert result["tokens_saved"] == 150

    # Embedder must NOT be called on Tier-1 exact match
    mock_embedder.embed_query.assert_not_called()


@pytest.mark.asyncio
async def test_cache_lookup_tier2_semantic_hit() -> None:
    """Test Tier-2 semantic match when Tier-1 misses."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock()
    mock_pool.acquire.return_value = mock_acquire

    # First call (exact) returns None, second call (semantic) returns hit
    mock_conn.fetchrow = AsyncMock(
        side_effect=[
            None,
            {
                "answer": "Home office stipend is $1500.",
                "citations": json.dumps([{"excerpt": "Remote stipend doc"}]),
                "tokens_used": 180,
                "distance": 0.09,
            },
        ]
    )

    mock_embedder = MagicMock()
    mock_embedder.embed_query = AsyncMock(return_value=[0.05] * 768)

    manager = QueryCacheManager(pool=mock_pool, embedder=mock_embedder)
    result = await manager.lookup(question="How much do we get for remote office setup?", user_email="bob@acme.com")

    assert result is not None
    assert result["cache_type"] == "semantic"
    assert "$1500" in result["answer"]
    assert result["distance"] == 0.09
    mock_embedder.embed_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_lookup_miss() -> None:
    """Test cache miss when neither exact nor semantic matches exist."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock()
    mock_pool.acquire.return_value = mock_acquire

    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_embedder = MagicMock()
    mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)

    manager = QueryCacheManager(pool=mock_pool, embedder=mock_embedder)
    result = await manager.lookup(question="Completely unknown query", user_email="bob@acme.com")

    assert result is None


@pytest.mark.asyncio
async def test_cache_store() -> None:
    """Test storing an answer in the two-tier cache."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock()
    mock_pool.acquire.return_value = mock_acquire

    mock_conn.execute = AsyncMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)

    manager = QueryCacheManager(pool=mock_pool, embedder=mock_embedder)
    await manager.store(
        question="What is our security policy?",
        user_email="alice@acme.com",
        answer="Security policy requires MFA.",
        citations=[{"doc_id": "doc_sec", "excerpt": "MFA is mandatory."}],
        tokens_used=120,
    )

    mock_conn.execute.assert_awaited_once()
    mock_embedder.embed_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_with_embedded_store_two_tier(tmp_path) -> None:
    """Test two-tier caching over EmbeddedVectorStore (SQLite + NumPy)."""
    from backend.services.embedded_store import EmbeddedVectorStore

    embedded_store = EmbeddedVectorStore(tmp_path / "cache_test.db")

    mock_embedder = MagicMock()
    # Vector for remote work
    vec_remote = [1.0, 0.0, 0.0, 0.0]
    # Slightly perturbed vector for semantic match (cosine distance ~0.05)
    vec_perturbed = [0.98, 0.20, 0.0, 0.0]
    # Orthogonal vector for miss
    vec_unrelated = [0.0, 0.0, 1.0, 0.0]

    mock_embedder.embed_query = AsyncMock(side_effect=lambda q: (
        vec_remote if "remote" in q.lower()
        else vec_perturbed if "work" in q.lower() or "guidelines" in q.lower()
        else vec_unrelated
    ))

    cache = QueryCacheManager(pool=embedded_store, embedder=mock_embedder)

    # 1. Store
    await cache.store(
        question="What is the remote work policy?",
        user_email="alice@acme.com",
        answer="Remote work is allowed full time.",
        citations=[{"doc_id": "doc1", "excerpt": "Remote work is allowed."}],
        tokens_used=150,
    )

    # 2. Exact hit (normalized)
    hit_exact = await cache.lookup("what is the remote work policy?", "alice@acme.com")
    assert hit_exact is not None
    assert hit_exact["cache_type"] == "exact"
    assert hit_exact["answer"] == "Remote work is allowed full time."
    assert hit_exact["tokens_saved"] == 150

    # 3. Semantic hit (different phrasing)
    hit_semantic = await cache.lookup("Explain remote guidelines", "alice@acme.com")
    assert hit_semantic is not None
    assert hit_semantic["cache_type"] == "semantic"
    assert hit_semantic["distance"] < 0.16

    # 4. Cache miss
    hit_miss = await cache.lookup("How to cook pasta?", "alice@acme.com")
    assert hit_miss is None

    # 5. Invalidation
    deleted = await cache.invalidate_for_docs(embedded_store)
    assert deleted == 1

    hit_after_invalidation = await cache.lookup("What is the remote work policy?", "alice@acme.com")
    assert hit_after_invalidation is None
