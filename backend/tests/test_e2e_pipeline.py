"""End-to-End integration tests for FastAPI query and sync endpoints."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.v1.query import _stream_rag_response, router as query_router
from backend.api.v1.sync import router as sync_router
from backend.models.query import QueryRequest


def create_test_app() -> FastAPI:
    """Instantiate a lightweight FastAPI app for endpoint integration testing."""
    app = FastAPI()
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(sync_router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_stream_rag_response_cold_path() -> None:
    """Test full SSE streaming lifecycle on a cache miss."""
    mock_graph = MagicMock()
    mock_graph.run = AsyncMock(
        return_value={
            "answer": "The remote work policy allows full-time remote.",
            "citations": [
                {
                    "doc_title": "Remote Policy",
                    "source_url": "http://drive/doc1",
                    "section_header": "Overview",
                    "excerpt": "Full-time remote is supported.",
                }
            ],
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 25,
                "total_tokens": 145,
                "total_cost_usd": 0.000018,
            },
            "is_relevant": True,
            "is_faithful": True,
        }
    )

    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock(return_value=None)
    mock_cache.store = AsyncMock()

    mock_request = MagicMock()
    mock_request.app.state.cache_manager = mock_cache

    body = QueryRequest(
        question="What is our remote work policy?",
        user_email="user@acme.com",
    )

    frames = []
    async for frame in _stream_rag_response(mock_graph, body, mock_request):
        frames.append(frame)

    assert len(frames) > 3
    # Check [DONE] delimiter
    assert any("[DONE]" in f for f in frames)

    # Check terminal metadata JSON frame
    meta_frame = frames[-1]
    assert meta_frame.startswith("data: {")
    payload = json.loads(meta_frame.replace("data: ", "").strip())
    assert payload["total_tokens"] == 145
    assert payload["cost_usd"] == 0.000018
    assert payload["is_faithful"] is True
    assert len(payload["citations"]) == 1


@pytest.mark.asyncio
async def test_stream_rag_response_cache_hit() -> None:
    """Test SSE streaming lifecycle on a Tier-1 exact cache hit."""
    mock_graph = MagicMock()
    mock_graph.run = AsyncMock()

    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock(
        return_value={
            "answer": "Cached answer for transformer query.",
            "citations": [],
            "cache_type": "exact",
            "latency_ms": 2.5,
            "tokens_saved": 210,
        }
    )

    mock_request = MagicMock()
    mock_request.app.state.cache_manager = mock_cache

    body = QueryRequest(
        question="What is a transformer?",
        user_email="alice@acme.com",
    )

    frames = []
    async for frame in _stream_rag_response(mock_graph, body, mock_request):
        frames.append(frame)

    # Graph must NOT be invoked on cache hit
    mock_graph.run.assert_not_called()

    # Terminal metadata
    meta_frame = frames[-1]
    payload = json.loads(meta_frame.replace("data: ", "").strip())
    assert payload["cache_hit"] == "exact"
    assert payload["tokens_saved"] == 210
    assert payload["cost_saved_usd"] > 0
    assert payload["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_api_upload_endpoint() -> None:
    """Test POST /api/v1/sync/upload endpoint with multipart file upload."""
    app = create_test_app()
    mock_sync_manager = MagicMock()
    mock_sync_manager.ingest_file = AsyncMock(
        return_value={
            "status": "success",
            "doc_id": "upload_abc123",
            "title": "handbook.md",
            "chunks_created": 3,
            "duration_seconds": 0.45,
        }
    )
    app.state.sync_manager = mock_sync_manager

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {
            "file": ("handbook.md", b"# Company Handbook\n\nWelcome to Acme Inc.", "text/markdown"),
        }
        data = {"owner_email": "admin@acme.com"}
        response = await client.post("/api/v1/sync/upload", files=files, data=data)

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["doc_id"] == "upload_abc123"
    assert res_data["chunks_created"] == 3
    mock_sync_manager.ingest_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_list_documents_endpoint() -> None:
    """Test GET /api/v1/sync/documents endpoint."""
    app = create_test_app()
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock()
    mock_pool.acquire.return_value = mock_acquire

    mock_conn.fetch = AsyncMock(
        return_value=[
            {
                "doc_id": "doc_1",
                "title": "Engineering Handbook",
                "source_url": "local://doc_1",
                "mime_type": "text/markdown",
                "synced_at": None,
                "chunk_count": 5,
            }
        ]
    )
    app.state.db_pool = mock_pool

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/sync/documents")

    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc_1"
    assert docs[0]["chunk_count"] == 5
