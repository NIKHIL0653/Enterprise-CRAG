"""Unit tests for the EmbeddedVectorStore (SQLite + NumPy) and its integration with HybridRetriever."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.models.document import DocumentChunk
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.services.embedded_store import EmbeddedVectorStore


@pytest.fixture
def temp_store(tmp_path) -> EmbeddedVectorStore:
    db_file = tmp_path / "test_rag.db"
    return EmbeddedVectorStore(db_file)


@pytest.mark.asyncio
async def test_embedded_store_upsert_and_dense_search(temp_store: EmbeddedVectorStore) -> None:
    """Validate that dense search computes cosine similarity and ranks correctly."""
    await temp_store.upsert_document(doc_id="doc1", title="Remote Work Policy")

    # Three 4-dimensional unit vectors
    c1 = DocumentChunk(
        chunk_id="c1",
        doc_id="doc1",
        doc_title="Remote Work Policy",
        source_url="http://drive/doc1",
        section_header="Stipends",
        content="Remote employees receive $1500 annual equipment stipend.",
        token_count=10,
        chunk_index=0,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        doc_id="doc1",
        doc_title="Remote Work Policy",
        source_url="http://drive/doc1",
        section_header="Hours",
        content="Core working hours are 10am to 4pm EST.",
        token_count=10,
        chunk_index=1,
        embedding=[0.0, 1.0, 0.0, 0.0],
    )

    await temp_store.upsert_chunks([c1, c2])

    # Query pointing towards c1
    results = await temp_store.dense_search(query_vector=[0.9, 0.1, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[1].chunk_id == "c2"


@pytest.mark.asyncio
async def test_embedded_store_sparse_search_fts5(temp_store: EmbeddedVectorStore) -> None:
    """Validate SQLite FTS5 BM25 sparse keyword search."""
    await temp_store.upsert_document(doc_id="doc1", title="Engineering Standards")

    c1 = DocumentChunk(
        chunk_id="c_eng",
        doc_id="doc1",
        doc_title="Engineering Standards",
        source_url="http://drive/eng",
        section_header="Python",
        content="All Python microservices must use Ruff linter and strict type annotations.",
        token_count=12,
        chunk_index=0,
    )
    c2 = DocumentChunk(
        chunk_id="c_sec",
        doc_id="doc1",
        doc_title="Engineering Standards",
        source_url="http://drive/eng",
        section_header="Security",
        content="Rotate all OAuth credentials and API keys every 90 days.",
        token_count=12,
        chunk_index=1,
    )

    await temp_store.upsert_chunks([c1, c2])

    results = await temp_store.sparse_search(query="Ruff type annotations", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c_eng"


@pytest.mark.asyncio
async def test_embedded_store_document_scoping(temp_store: EmbeddedVectorStore) -> None:
    """Validate that search results can be strictly scoped to specific doc_ids."""
    await temp_store.upsert_document(doc_id="doc_A", title="Doc A", mime_type="application/pdf")
    await temp_store.upsert_document(doc_id="doc_B", title="Doc B", mime_type="text/csv")

    c_a = DocumentChunk(
        chunk_id="ca",
        doc_id="doc_A",
        doc_title="Doc A",
        source_url="",
        content="Important policy guidelines for project Alpha.",
        token_count=8,
        chunk_index=0,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    c_b = DocumentChunk(
        chunk_id="cb",
        doc_id="doc_B",
        doc_title="Doc B",
        source_url="",
        content="Important policy guidelines for project Beta.",
        token_count=8,
        chunk_index=0,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )

    await temp_store.upsert_chunks([c_a, c_b])

    # Scope to Doc B only
    dense_scoped = await temp_store.dense_search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=5, doc_ids=["doc_B"])
    assert len(dense_scoped) == 1
    assert dense_scoped[0].doc_id == "doc_B"

    sparse_scoped = await temp_store.sparse_search(query="policy guidelines", top_k=5, doc_ids=["doc_A"])
    assert len(sparse_scoped) == 1
    assert sparse_scoped[0].doc_id == "doc_A"

    # Filter by mime_type using predicate pushdown
    dense_filtered = await temp_store.dense_search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"mime_type": "text/csv"},
    )
    assert len(dense_filtered) == 1
    assert dense_filtered[0].doc_id == "doc_B"

    sparse_filtered = await temp_store.sparse_search(
        query="policy guidelines",
        top_k=5,
        filters={"mime_type": "application/pdf"},
    )
    assert len(sparse_filtered) == 1
    assert sparse_filtered[0].doc_id == "doc_A"


@pytest.mark.asyncio
async def test_embedded_store_hybrid_retriever_fusion(temp_store: EmbeddedVectorStore) -> None:
    """Validate that HybridRetriever seamlessly operates on EmbeddedVectorStore."""
    await temp_store.upsert_document(doc_id="doc_fusion", title="Fusion Doc")

    c1 = DocumentChunk(
        chunk_id="c_fused_1",
        doc_id="doc_fusion",
        doc_title="Fusion Doc",
        source_url="",
        content="Kubernetes cluster auto-scaling instructions.",
        token_count=8,
        chunk_index=0,
        embedding=[0.8, 0.2, 0.0, 0.0],
    )
    await temp_store.upsert_chunks([c1])

    mock_embedder = AsyncMock()
    mock_embedder.embed_query = AsyncMock(return_value=[0.8, 0.2, 0.0, 0.0])

    retriever = HybridRetriever(db_pool=temp_store, embedder=mock_embedder, rrf_k=60)
    results = await retriever.retrieve(query="Kubernetes cluster", user_email="test@acme.com", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "c_fused_1"


@pytest.mark.asyncio
async def test_embedded_store_delete_document(temp_store: EmbeddedVectorStore) -> None:
    """Validate cascade deletion of document and its chunks."""
    await temp_store.upsert_document(doc_id="doc_del", title="Delete Me")
    c = DocumentChunk(
        chunk_id="c_del",
        doc_id="doc_del",
        doc_title="Delete Me",
        source_url="",
        content="Temporary scratch content to be deleted.",
        token_count=8,
        chunk_index=0,
        embedding=[0.5, 0.5, 0.0, 0.0],
    )
    await temp_store.upsert_chunks([c])

    docs_before = await temp_store.list_documents()
    assert len(docs_before) == 1

    await temp_store.delete_document("doc_del")

    docs_after = await temp_store.list_documents()
    assert len(docs_after) == 0

    sparse_after = await temp_store.sparse_search("scratch content", top_k=5)
    assert len(sparse_after) == 0


@pytest.mark.asyncio
async def test_embedded_store_acl_user_permission_filtering(temp_store: EmbeddedVectorStore) -> None:
    """Validate that chunks are filtered by user_email ACLs."""
    await temp_store.upsert_document(doc_id="doc_hr", title="HR Confidential")
    await temp_store.upsert_chunks([
        DocumentChunk(
            chunk_id="c_hr_confidential",
            doc_id="doc_hr",
            doc_title="HR Confidential",
            source_url="",
            content="Executive compensation and executive bonuses list.",
            token_count=10,
            chunk_index=0,
            allowed_emails=["hr_director@example.com"],
            allowed_domain="executive.example.com",
            embedding=[0.9, 0.1, 0.0, 0.0],
        ),
        DocumentChunk(
            chunk_id="c_general_policy",
            doc_id="doc_hr",
            doc_title="HR Confidential",
            source_url="",
            content="General holiday calendar for all employees.",
            token_count=8,
            chunk_index=1,
            allowed_emails=[],
            allowed_domain=None,
            embedding=[0.9, 0.1, 0.0, 0.0],
        ),
    ])

    # 1. Unauthorized user should only see general chunk
    unauth_results = await temp_store.dense_search(
        query_vector=[0.9, 0.1, 0.0, 0.0],
        top_k=5,
        user_email="intern@example.com",
    )
    assert len(unauth_results) == 1
    assert unauth_results[0].chunk_id == "c_general_policy"

    # 2. Authorized email should see both chunks
    auth_results = await temp_store.dense_search(
        query_vector=[0.9, 0.1, 0.0, 0.0],
        top_k=5,
        user_email="hr_director@example.com",
    )
    assert len(auth_results) == 2

    # 3. Authorized domain should see both chunks
    domain_results = await temp_store.dense_search(
        query_vector=[0.9, 0.1, 0.0, 0.0],
        top_k=5,
        user_email="cfo@executive.example.com",
    )
    assert len(domain_results) == 2
