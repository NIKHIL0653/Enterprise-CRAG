"""
Pytest fixtures and integration test helpers for the WorkspaceRAG backend.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def mock_settings(monkeypatch):
    """Patch settings with safe test defaults."""
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "/tmp/fake.json")
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "test_folder_id")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test_lf_secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test_lf_public")


@pytest.fixture
def sample_chunk():
    """Return a minimal valid DocumentChunk for use in tests."""
    from datetime import datetime, timezone

    from backend.models.document import DocumentChunk

    return DocumentChunk(
        chunk_id="abc123",
        doc_id="doc001",
        doc_title="Test Policy Doc",
        source_url="https://docs.google.com/document/d/doc001/edit",
        section_header="Introduction",
        content="This document describes the remote work policy at Acme Corp.",
        token_count=12,
        chunk_index=0,
        allowed_emails=["user@acme.com"],
        allowed_domain="acme.com",
        embedding=None,
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )
