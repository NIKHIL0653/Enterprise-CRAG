"""Public API surface for the ``backend.ingestion`` package.

Re-exports the four main classes so callers can do::

    from backend.ingestion import (
        GoogleDriveClient,
        SemanticMarkdownChunker,
        GeminiEmbedder,
        SyncManager,
    )
"""

from __future__ import annotations

from backend.ingestion.chunker import SemanticMarkdownChunker
from backend.ingestion.embedder import GeminiEmbedder
from backend.ingestion.google_drive_client import GoogleDriveClient
from backend.ingestion.sync_manager import SyncManager

__all__: list[str] = [
    "GoogleDriveClient",
    "SemanticMarkdownChunker",
    "GeminiEmbedder",
    "SyncManager",
]
