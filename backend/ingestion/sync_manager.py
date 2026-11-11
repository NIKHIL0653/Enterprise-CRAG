"""Sync manager — orchestrates full and incremental Google Drive sync.

Coordinates the Drive client, chunker, and embedder to ingest documents
into PostgreSQL, supporting both full re-sync and incremental change-based
update workflows.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import structlog

from backend.core.config import get_settings
from backend.ingestion.chunker import SemanticMarkdownChunker
from backend.ingestion.embedder import EmbeddingError, GeminiEmbedder
from backend.ingestion.google_drive_client import GoogleDriveClient, GoogleDriveError
from backend.models.document import Document, DocumentChunk

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class SyncError(Exception):
    """Raised for unrecoverable errors during the sync pipeline."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialise SyncError.

        Args:
            message: Human-readable description of the failure.
            cause: The underlying exception, if any.
        """
        super().__init__(message)
        self.cause = cause


# ---------------------------------------------------------------------------
# SyncManager
# ---------------------------------------------------------------------------


class SyncManager:
    """Orchestrates full and incremental Google Drive document sync.

    Pulls documents from Drive, chunks them, embeds the chunks, and upserts
    everything to a PostgreSQL database via an ``asyncpg`` connection pool.

    Example::

        manager = SyncManager(
            db_pool=pool,
            drive_client=GoogleDriveClient(),
            chunker=SemanticMarkdownChunker(),
            embedder=GeminiEmbedder(),
        )
        stats = await manager.run_full_sync()
    """

    def __init__(
        self,
        db_pool: Any,
        drive_client: GoogleDriveClient,
        chunker: SemanticMarkdownChunker,
        embedder: GeminiEmbedder,
    ) -> None:
        """Initialise the sync manager with its dependencies.

        Args:
            db_pool: An ``asyncpg`` (or compatible) connection pool.
            drive_client: Configured :class:`GoogleDriveClient` instance.
            chunker: Configured :class:`SemanticMarkdownChunker` instance.
            embedder: Configured :class:`GeminiEmbedder` instance.
        """
        self._pool = db_pool
        self._drive = drive_client
        self._chunker = chunker
        self._embedder = embedder
        self._settings = get_settings()
        logger.info("SyncManager initialised")

    # ------------------------------------------------------------------
    # Public sync entry points
    # ------------------------------------------------------------------

    async def run_full_sync(self) -> dict[str, Any]:
        """Perform a full re-sync of all Google Docs in the configured folder.

        Fetches every file from ``GOOGLE_DRIVE_FOLDER_ID``, processes each
        through the chunker and embedder, then upserts to the database.

        Returns:
            A stats dict with keys ``processed``, ``failed``, and
            ``duration_seconds``.
        """
        log = logger.bind(method="run_full_sync")
        log.info("Full sync started")
        start_time = time.monotonic()

        folder_id: str = self._settings.GOOGLE_DRIVE_FOLDER_ID
        try:
            files = await self._drive.get_all_files_in_folder(folder_id)
        except GoogleDriveError as exc:
            raise SyncError(
                f"Failed to list files in folder {folder_id}: {exc}", cause=exc
            ) from exc

        log.info("Files retrieved", count=len(files))

        processed = 0
        failed = 0

        for file_meta in files:
            doc = await self._process_file(file_meta)
            if doc is None:
                failed += 1
                continue
            try:
                await self._upsert_to_db(doc)
                processed += 1
                log.info(
                    "Document upserted",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    chunks=len(doc.chunks),
                )
            except Exception as exc:
                log.error(
                    "DB upsert failed",
                    doc_id=doc.doc_id,
                    error=str(exc),
                    exc_info=True,
                )
                failed += 1

        if processed > 0:
            try:
                from backend.services.cache import QueryCacheManager
                await QueryCacheManager.invalidate_for_docs(None, self._db_pool)
            except Exception as exc:
                log.warning("cache_invalidation_failed", error=str(exc))

        duration = time.monotonic() - start_time
        stats: dict[str, Any] = {
            "processed": processed,
            "failed": failed,
            "duration_seconds": round(duration, 2),
        }
        log.info("Full sync complete", **stats)
        return stats

    async def run_incremental_sync(
        self, page_token: str | None
    ) -> tuple[dict[str, Any], str]:
        """Sync only files that have changed since the last run.

        Uses the Drive changes API with *page_token* for incremental updates.

        Args:
            page_token: The Drive changes page token from the previous sync, or
                ``None`` to begin from the current state.

        Returns:
            A two-tuple ``(stats_dict, new_page_token)`` where *stats_dict*
            has keys ``processed``, ``failed``, ``duration_seconds``, and
            ``new_page_token`` is the token to persist for the next run.

        Raises:
            SyncError: If the Drive changes list cannot be fetched.
        """
        log = logger.bind(method="run_incremental_sync", page_token=page_token)
        log.info("Incremental sync started")
        start_time = time.monotonic()

        try:
            files, new_page_token = await self._drive.list_changed_files(page_token)
        except GoogleDriveError as exc:
            raise SyncError(
                f"Failed to list changed files: {exc}", cause=exc
            ) from exc

        # Filter to Google Docs only
        doc_files = [f for f in files if f.get("mimeType") == _GOOGLE_DOC_MIME]
        log.info(
            "Changed files fetched",
            total=len(files),
            google_docs=len(doc_files),
            new_page_token=new_page_token,
        )

        processed = 0
        failed = 0

        for file_meta in doc_files:
            doc = await self._process_file(file_meta)
            if doc is None:
                failed += 1
                continue
            try:
                await self._upsert_to_db(doc)
                processed += 1
                log.info(
                    "Document upserted",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    chunks=len(doc.chunks),
                )
            except Exception as exc:
                log.error(
                    "DB upsert failed",
                    doc_id=doc.doc_id,
                    error=str(exc),
                    exc_info=True,
                )
                failed += 1

        if processed > 0:
            try:
                from backend.services.cache import QueryCacheManager
                await QueryCacheManager.invalidate_for_docs(None, self._db_pool)
            except Exception as exc:
                log.warning("cache_invalidation_failed", error=str(exc))

        duration = time.monotonic() - start_time
        stats: dict[str, Any] = {
            "processed": processed,
            "failed": failed,
            "duration_seconds": round(duration, 2),
        }
        log.info("Incremental sync complete", **stats)
        return stats, new_page_token

    # ------------------------------------------------------------------
    # Private pipeline helpers
    # ------------------------------------------------------------------

    async def _process_file(self, file_meta: dict[str, Any]) -> Document | None:
        """Fetch, chunk, and embed a single Drive file.

        Args:
            file_meta: Drive API file metadata dict containing at minimum
                ``id``, ``name``, ``mimeType``, ``modifiedTime``, and
                ``webViewLink``.

        Returns:
            A fully populated :class:`~backend.models.document.Document` with
            embedded chunks, or ``None`` if any step in the pipeline fails.
        """
        file_id: str = file_meta.get("id", "")
        title: str = file_meta.get("name", "")
        log = logger.bind(method="_process_file", file_id=file_id, title=title)
        log.info("Processing file")

        # 1. Fetch raw text
        mime_type = file_meta.get("mimeType", "")
        try:
            raw_text = await self._drive.get_document_as_markdown(file_id, mime_type=mime_type)
        except GoogleDriveError as exc:
            log.error("Failed to export document text", error=str(exc))
            return None

        # 2. Fetch permissions
        try:
            allowed_emails, allowed_domain = await self._drive.get_file_permissions(file_id)
        except GoogleDriveError as exc:
            log.error("Failed to fetch permissions", error=str(exc))
            return None

        # 3. Build the owner email from metadata
        owners: list[dict[str, Any]] = file_meta.get("owners", [])
        owner_email: str = owners[0].get("emailAddress", "") if owners else ""

        # 4. Parse modified time
        modified_str: str = file_meta.get("modifiedTime", "")
        try:
            last_modified = datetime.fromisoformat(
                modified_str.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            last_modified = datetime.now(timezone.utc)

        now = datetime.now(timezone.utc)

        # 5. Construct a minimal Document (no chunks yet) for the chunker
        doc = Document(
            doc_id=file_id,
            title=title,
            source_url=file_meta.get("webViewLink", ""),
            mime_type=file_meta.get("mimeType", _GOOGLE_DOC_MIME),
            owner_email=owner_email,
            allowed_emails=allowed_emails,
            allowed_domain=allowed_domain,
            chunks=[],
            last_modified=last_modified,
            synced_at=now,
        )

        # 6. Chunk the raw text
        try:
            chunks: list[DocumentChunk] = self._chunker.chunk_document(doc, raw_text)
        except Exception as exc:
            log.error("Chunking failed", error=str(exc), exc_info=True)
            return None

        if not chunks:
            log.warning("No chunks produced — skipping document")
            return None

        # 7. Embed chunks
        try:
            chunks = await self._embedder.embed_chunks(chunks)
        except EmbeddingError as exc:
            log.error("Embedding failed", error=str(exc), exc_info=True)
            return None

        # 8. Attach chunks to document and return
        doc.chunks = chunks
        log.info(
            "File processed successfully",
            chunk_count=len(chunks),
            owner_email=owner_email,
        )
        return doc

    async def _upsert_to_db(self, doc: Document) -> None:
        """Upsert a document and its chunks to PostgreSQL.

        Strategy:
        1. Upsert document-level metadata into the ``documents`` table.
        2. Delete all existing ``document_chunks`` rows for ``doc_id``.
        3. Bulk-insert new chunks (including their embedding vectors).

        Args:
            doc: The :class:`~backend.models.document.Document` to persist.

        Raises:
            Exception: Propagates any ``asyncpg`` database error upward; the
                caller is responsible for catching and logging.
        """
        log = logger.bind(
            method="_upsert_to_db",
            doc_id=doc.doc_id,
            chunk_count=len(doc.chunks),
        )
        log.debug("Upserting document to database")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # --- Upsert document metadata ---
                await conn.execute(
                    """
                    INSERT INTO documents (
                        doc_id, title, source_url, mime_type, owner_email,
                        allowed_emails, allowed_domain, last_modified, synced_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title          = EXCLUDED.title,
                        source_url     = EXCLUDED.source_url,
                        owner_email    = EXCLUDED.owner_email,
                        allowed_emails = EXCLUDED.allowed_emails,
                        allowed_domain = EXCLUDED.allowed_domain,
                        last_modified  = EXCLUDED.last_modified,
                        synced_at      = EXCLUDED.synced_at
                    """,
                    doc.doc_id,
                    doc.title,
                    doc.source_url,
                    doc.mime_type,
                    doc.owner_email,
                    doc.allowed_emails,
                    doc.allowed_domain,
                    doc.last_modified,
                    doc.synced_at,
                )

                # --- Remove stale chunks ---
                deleted = await conn.execute(
                    "DELETE FROM document_chunks WHERE doc_id = $1", doc.doc_id
                )
                log.debug("Stale chunks removed", result=deleted)

                # --- Bulk-insert new chunks ---
                chunk_rows = [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.doc_title,
                        chunk.source_url,
                        chunk.section_header,
                        chunk.content,
                        chunk.token_count,
                        chunk.chunk_index,
                        chunk.allowed_emails,
                        chunk.allowed_domain,
                        chunk.embedding,
                        chunk.created_at,
                        chunk.updated_at,
                    )
                    for chunk in doc.chunks
                ]

                await conn.executemany(
                    """
                    INSERT INTO document_chunks (
                        chunk_id, doc_id, doc_title, source_url,
                        section_header, content, token_count, chunk_index,
                        allowed_emails, allowed_domain, embedding,
                        created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8,
                        $9, $10, $11, $12, $13
                    )
                    """,
                    chunk_rows,
                )

        log.info("Document upserted successfully", doc_id=doc.doc_id)

    # Aliases for compatibility
    full_sync = run_full_sync
    incremental_sync = run_incremental_sync
