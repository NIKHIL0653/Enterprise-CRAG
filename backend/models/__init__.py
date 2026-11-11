"""Public API for the ``backend.models`` package.

Re-exports all domain model classes so that consumers can use a single,
stable import path::

    from backend.models import Document, DocumentChunk, QueryRequest, QueryResponse

Symbols exported:
    - :class:`~backend.models.document.DocumentChunk`
    - :class:`~backend.models.document.Document`
    - :class:`~backend.models.query.QueryRequest`
    - :class:`~backend.models.query.Citation`
    - :class:`~backend.models.query.QueryResponse`
    - :class:`~backend.models.query.RAGState`
"""

from __future__ import annotations

from backend.models.document import Document, DocumentChunk
from backend.models.query import Citation, QueryRequest, QueryResponse, RAGState

__all__ = [
    # Document models
    "DocumentChunk",
    "Document",
    # Query / response models
    "QueryRequest",
    "Citation",
    "QueryResponse",
    # LangGraph state
    "RAGState",
]
