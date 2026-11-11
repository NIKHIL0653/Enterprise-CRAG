"""Pydantic v2 domain models for documents and their chunks.

These models represent the core data structures that flow through the ingestion
pipeline, vector store, and retrieval layer.

Example::

    from backend.models.document import Document, DocumentChunk

    chunk = DocumentChunk(
        chunk_id="c_001",
        doc_id="doc_abc",
        doc_title="Q3 Strategy Deck",
        source_url="https://drive.google.com/file/d/abc/view",
        content="Our primary objective for Q3 is to …",
        token_count=128,
        chunk_index=0,
        allowed_emails=["alice@acme.com"],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single text chunk derived from a source document.

    Chunks are the atomic units stored in the vector database.  Each chunk
    carries its own access-control lists so that retrieval can enforce
    per-user permissions without loading the parent :class:`Document`.

    Attributes:
        chunk_id: Globally unique identifier for this chunk (UUID or hash).
        doc_id: Identifier of the parent :class:`Document`.
        doc_title: Human-readable title of the parent document.
        source_url: Canonical URL to the original Google Drive file.
        section_header: Optional heading that immediately precedes this chunk
            in the source document, used for citation rendering.
        content: The raw text content of this chunk.
        token_count: Number of tokens in ``content`` as counted by the
            tokeniser used during ingestion.
        chunk_index: Zero-based position of this chunk within the parent
            document, used for ordering during display.
        allowed_emails: Explicit list of email addresses that may access this
            chunk.  An empty list means *no individual overrides* (fall back
            to ``allowed_domain``).
        allowed_domain: If set, any authenticated user whose email domain
            matches this value may access the chunk.
        embedding: The floating-point embedding vector for this chunk.
            ``None`` before the embedding step has run.
        created_at: UTC timestamp when this chunk record was first created.
        updated_at: UTC timestamp of the most recent update to this chunk.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    source_url: str
    section_header: str | None = None
    content: str
    token_count: int
    chunk_index: int
    allowed_emails: list[str] = Field(default_factory=list)
    allowed_domain: str | None = None
    embedding: list[float] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"frozen": False}


class Document(BaseModel):
    """A source document ingested from Google Drive.

    A :class:`Document` is the top-level unit of ingestion.  After processing,
    it is split into one or more :class:`DocumentChunk` instances that are
    embedded and stored in the vector database.

    Attributes:
        doc_id: Globally unique identifier for this document (typically the
            Google Drive file ID).
        title: Human-readable title of the document.
        source_url: Canonical URL to the Google Drive file.
        mime_type: MIME type of the original file, e.g.
            ``"application/vnd.google-apps.document"``.
        owner_email: Email address of the Google Drive file owner.
        allowed_emails: Explicit list of email addresses granted read access.
        allowed_domain: If set, any user whose email domain matches this value
            is implicitly granted read access.
        chunks: Ordered list of :class:`DocumentChunk` objects produced from
            this document.
        last_modified: UTC timestamp of the most recent modification to the
            source Google Drive file.
        synced_at: UTC timestamp when this document was last successfully
            ingested by the sync pipeline.
    """

    doc_id: str
    title: str
    source_url: str
    mime_type: str
    owner_email: str
    allowed_emails: list[str] = Field(default_factory=list)
    allowed_domain: str | None = None
    chunks: list[DocumentChunk] = Field(default_factory=list)
    last_modified: datetime
    synced_at: datetime

    model_config = {"frozen": False}
