"""
Unit tests for the SemanticMarkdownChunker.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.models.document import Document


@pytest.fixture
def sample_document():
    return Document(
        doc_id="doc-test-001",
        title="Remote Work Policy",
        source_url="https://docs.google.com/document/d/doc-test-001/edit",
        mime_type="application/vnd.google-apps.document",
        owner_email="hr@acme.com",
        allowed_emails=["all@acme.com"],
        allowed_domain="acme.com",
        chunks=[],
        last_modified=datetime.now(tz=timezone.utc),
        synced_at=datetime.now(tz=timezone.utc),
    )


SAMPLE_TEXT = r"""
# Remote Work Policy

## Overview
Acme Corp allows all employees to work remotely on a full-time basis.
Managers may request in-office attendance for key planning meetings.

## Eligibility
All full-time employees hired after January 2023 are eligible for remote work.
Part-time contractors must obtain written approval from their hiring manager.

## Equipment
The company provides a \$1,500 home office stipend per year for all remote employees.
IT will provision a laptop and necessary software licenses within the first week.
"""


def test_chunker_produces_chunks(sample_document):
    """Chunker should produce at least one chunk from non-trivial text."""
    from backend.ingestion.chunker import SemanticMarkdownChunker

    chunker = SemanticMarkdownChunker(max_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk_document(sample_document, SAMPLE_TEXT)
    assert len(chunks) > 0


def test_chunks_have_correct_doc_id(sample_document):
    """All chunks must have the parent document's doc_id."""
    from backend.ingestion.chunker import SemanticMarkdownChunker

    chunker = SemanticMarkdownChunker()
    chunks = chunker.chunk_document(sample_document, SAMPLE_TEXT)
    for chunk in chunks:
        assert chunk.doc_id == sample_document.doc_id


def test_chunks_have_section_headers(sample_document):
    """At least some chunks should capture heading context."""
    from backend.ingestion.chunker import SemanticMarkdownChunker

    chunker = SemanticMarkdownChunker()
    chunks = chunker.chunk_document(sample_document, SAMPLE_TEXT)
    headers = [c.section_header for c in chunks if c.section_header]
    assert len(headers) > 0


def test_chunk_ids_are_unique(sample_document):
    """All chunk_ids within a document must be unique."""
    from backend.ingestion.chunker import SemanticMarkdownChunker

    chunker = SemanticMarkdownChunker()
    chunks = chunker.chunk_document(sample_document, SAMPLE_TEXT)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"


def test_chunk_content_is_non_empty(sample_document):
    """No chunk should have empty content."""
    from backend.ingestion.chunker import SemanticMarkdownChunker

    chunker = SemanticMarkdownChunker()
    chunks = chunker.chunk_document(sample_document, SAMPLE_TEXT)
    for chunk in chunks:
        assert chunk.content.strip(), f"Empty chunk at index {chunk.chunk_index}"


def test_token_count_matches_content(sample_document):
    """Token count should be non-zero and plausible for the content."""
    from backend.ingestion.chunker import SemanticMarkdownChunker

    chunker = SemanticMarkdownChunker()
    chunks = chunker.chunk_document(sample_document, SAMPLE_TEXT)
    for chunk in chunks:
        assert chunk.token_count > 0
        assert chunk.token_count <= 600  # well within our 512 max + tolerance
