"""Unit tests for direct file ingestion and parsers."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.ingestion.parsers import parse_bytes_to_markdown, parse_docx_bytes
from backend.ingestion.sync_manager import SyncError, SyncManager
from backend.models.document import DocumentChunk


def test_parse_bytes_markdown_and_text() -> None:
    """Test text and markdown byte parsing."""
    raw_md = b"# Policy Document\n\nThis is a test policy."
    parsed = parse_bytes_to_markdown(raw_md, filename="policy.md")
    assert "# Policy Document" in parsed
    assert "This is a test policy." in parsed


def test_parse_docx_bytes_valid() -> None:
    """Test DOCX XML byte extraction via parse_docx_bytes and parse_bytes_to_markdown."""
    fh = io.BytesIO()
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        "  <w:body>\n"
        "    <w:p><w:r><w:t>Quarterly Financial Results</w:t></w:r></w:p>\n"
        "    <w:p><w:r><w:t>Net profit increased by 15%.</w:t></w:r></w:p>\n"
        "  </w:body>\n"
        "</w:document>"
    )
    with zipfile.ZipFile(fh, mode="w") as z:
        z.writestr("word/document.xml", doc_xml)

    docx_bytes = fh.getvalue()
    extracted = parse_docx_bytes(docx_bytes)
    assert "Quarterly Financial Results" in extracted
    assert "Net profit increased by 15%." in extracted

    # Test via parse_bytes_to_markdown with filename
    result_by_name = parse_bytes_to_markdown(docx_bytes, filename="q3.docx")
    assert "Quarterly Financial Results" in result_by_name


def test_parse_docx_empty_zip() -> None:
    """Test empty zip returns empty string without crashing."""
    fh = io.BytesIO()
    with zipfile.ZipFile(fh, mode="w") as z:
        z.writestr("other.txt", "hello")
    assert parse_docx_bytes(fh.getvalue()) == ""
    assert parse_docx_bytes(b"") == ""


@pytest.mark.asyncio
async def test_sync_manager_ingest_file_success() -> None:
    """Test SyncManager.ingest_file creates chunks, embeds them, and persists to db."""
    class MockTx:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    mock_conn = MagicMock()
    mock_conn.transaction.return_value = MockTx()
    mock_conn.execute = AsyncMock()
    mock_conn.executemany = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = MockAcquire()

    now = datetime.now(timezone.utc)
    mock_drive = MagicMock()
    mock_chunker = MagicMock()
    mock_chunker.chunk_document.return_value = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="upload_test",
            doc_title="test.md",
            source_url="local://test.md",
            section_header="Introduction",
            content="Sample text content for chunk.",
            token_count=12,
            chunk_index=0,
            allowed_emails=["user@test.com"],
            allowed_domain="",
            created_at=now,
            updated_at=now,
        )
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_chunks = AsyncMock(
        return_value=[
            DocumentChunk(
                chunk_id="chunk-1",
                doc_id="upload_test",
                doc_title="test.md",
                source_url="local://test.md",
                section_header="Introduction",
                content="Sample text content for chunk.",
                token_count=12,
                chunk_index=0,
                allowed_emails=["user@test.com"],
                allowed_domain="",
                embedding=[0.1] * 768,
                created_at=now,
                updated_at=now,
            )
        ]
    )

    manager = SyncManager(
        db_pool=mock_pool,
        drive_client=mock_drive,
        chunker=mock_chunker,
        embedder=mock_embedder,
    )

    result = await manager.ingest_file(
        filename="test.md",
        content_bytes=b"# Introduction\n\nSample text content for chunk.",
        owner_email="user@test.com",
    )

    assert result["status"] == "success"
    assert result["title"] == "test.md"
    assert result["chunks_created"] == 1
    assert "upload_" in result["doc_id"]
    mock_embedder.embed_chunks.assert_awaited_once()
    mock_conn.execute.assert_awaited()
    mock_conn.executemany.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_manager_ingest_empty_file_fails() -> None:
    """Test ingesting empty file raises SyncError."""
    mock_pool = MagicMock()
    mock_drive = MagicMock()
    mock_chunker = MagicMock()
    mock_embedder = MagicMock()

    manager = SyncManager(
        db_pool=mock_pool,
        drive_client=mock_drive,
        chunker=mock_chunker,
        embedder=mock_embedder,
    )

    with pytest.raises(SyncError, match="No extractable text found"):
        await manager.ingest_file(
            filename="empty.txt",
            content_bytes=b"   ",
            owner_email="user@test.com",
        )
