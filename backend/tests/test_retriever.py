"""
Unit tests for the HybridRetriever and RRF fusion logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.document import DocumentChunk


def make_chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc001",
        doc_title="Test Doc",
        source_url="https://docs.google.com/document/d/doc001",
        section_header="Section A",
        content=content,
        token_count=10,
        chunk_index=0,
        allowed_emails=["user@acme.com"],
        allowed_domain="acme.com",
        embedding=None,
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )


@pytest.mark.asyncio
async def test_rrf_merges_dense_and_sparse_results():
    """
    RRF fusion should produce a combined ranked list that includes chunks
    from both the dense and sparse retrieval lists.
    """
    from backend.retrieval.hybrid_retriever import HybridRetriever

    mock_pool = MagicMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    retriever = HybridRetriever(db_pool=mock_pool, embedder=mock_embedder)

    dense_results = [make_chunk(f"dense-{i}", f"Dense result {i}") for i in range(5)]
    sparse_results = [make_chunk(f"sparse-{i}", f"Sparse result {i}") for i in range(5)]

    # Test the internal RRF fusion method directly
    fused = retriever._rrf_fusion([dense_results, sparse_results], top_k=10, k=60)

    all_ids = {c.chunk_id for c in fused}
    # Should contain chunks from both lists
    assert any("dense" in cid for cid in all_ids)
    assert any("sparse" in cid for cid in all_ids)


@pytest.mark.asyncio
async def test_rrf_boosts_chunks_in_both_lists():
    """
    A chunk appearing in both dense and sparse results should rank higher
    via RRF than a chunk appearing in only one list.
    """
    from backend.retrieval.hybrid_retriever import HybridRetriever

    mock_pool = MagicMock()
    mock_embedder = AsyncMock()

    retriever = HybridRetriever(db_pool=mock_pool, embedder=mock_embedder)

    shared_chunk = make_chunk("shared-001", "Important shared document content.")
    dense_only = make_chunk("dense-only-001", "Only from dense search.")

    # shared_chunk appears last in dense list (rank=4) but first in sparse (rank=0)
    dense_results = [dense_only, make_chunk("d2", "d2"), make_chunk("d3", "d3"), make_chunk("d4", "d4"), shared_chunk]
    sparse_results = [shared_chunk, make_chunk("s2", "s2")]

    fused = retriever._rrf_fusion([dense_results, sparse_results], top_k=10, k=60)

    # shared_chunk should beat dense_only because it accumulates score from both lists
    fused_ids = [c.chunk_id for c in fused]
    assert fused_ids.index("shared-001") < fused_ids.index("dense-only-001")


@pytest.mark.asyncio
async def test_retrieve_forwards_doc_ids_to_subsearches() -> None:
    """Ensure doc_ids parameter is forwarded to both dense and sparse subsearches."""
    from backend.retrieval.hybrid_retriever import HybridRetriever

    mock_pool = MagicMock(spec=["acquire"])
    mock_conn = MagicMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_acquire

    mock_embedder = MagicMock()
    mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)

    retriever = HybridRetriever(db_pool=mock_pool, embedder=mock_embedder)
    retriever._dense_search = AsyncMock(return_value=[])
    retriever._sparse_search = AsyncMock(return_value=[])

    await retriever.retrieve(
        query="remote work policy",
        user_email="alice@acme.com",
        top_k=5,
        doc_ids=["doc-remote-1", "doc-remote-2"],
    )

    retriever._dense_search.assert_awaited_once_with(
        mock_conn, [0.1] * 768, "alice@acme.com", 5, doc_ids=["doc-remote-1", "doc-remote-2"]
    )
    retriever._sparse_search.assert_awaited_once_with(
        mock_conn, "remote work policy", "alice@acme.com", 5, doc_ids=["doc-remote-1", "doc-remote-2"]
    )


@pytest.mark.asyncio
async def test_dense_search_includes_doc_ids_filter() -> None:
    """Ensure _dense_search appends doc_id = ANY($4::text[]) when doc_ids are passed."""
    from backend.retrieval.hybrid_retriever import HybridRetriever

    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])

    mock_embedder = MagicMock()
    retriever = HybridRetriever(db_pool=mock_pool, embedder=mock_embedder)

    await retriever._dense_search(
        conn=mock_conn,
        vector=[0.1] * 768,
        user_email="alice@acme.com",
        top_k=10,
        doc_ids=["doc123"],
    )

    mock_conn.fetch.assert_awaited_once()
    called_sql = mock_conn.fetch.call_args[0][0]
    called_args = mock_conn.fetch.call_args[0][1:]
    assert "doc_id = ANY($4::text[])" in called_sql
    assert called_args == ("alice@acme.com", [0.1] * 768, 10, ["doc123"])

