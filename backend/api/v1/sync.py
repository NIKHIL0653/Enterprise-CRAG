"""FastAPI router for Google Drive sync endpoints.

Provides endpoints to trigger full or incremental Drive syncs and to query
the current sync status from the database.
"""

from __future__ import annotations

from typing import Any

import asyncpg
from fastapi import APIRouter, Query, Request

from backend.core.logging import get_logger
from backend.ingestion import SyncManager

logger = get_logger(__name__)

router = APIRouter(prefix="/sync", tags=["Sync"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/full",
    summary="Trigger a full Google Drive sync",
    description=(
        "Crawls all accessible Google Drive files and upserts them into the "
        "vector store.  Returns sync statistics when the operation completes."
    ),
    responses={
        200: {"description": "Sync completed; returns stats dict"},
        500: {"description": "Sync failed"},
    },
)
async def full_sync(request: Request) -> dict[str, Any]:
    """Execute a full synchronisation of all Google Drive documents.

    Args:
        request: The raw FastAPI ``Request`` used to access ``app.state``.

    Returns:
        A dictionary of sync statistics (e.g. ``docs_indexed``,
        ``chunks_created``, ``duration_seconds``).
    """
    sync_manager: SyncManager = request.app.state.sync_manager
    log = logger.bind(sync_type="full")
    log.info("full_sync_started")
    stats: dict[str, Any] = await sync_manager.run_full_sync()
    log.info("full_sync_complete", stats=stats)
    return stats


@router.post(
    "/incremental",
    summary="Trigger an incremental Google Drive sync",
    description=(
        "Fetches only the changes since the last sync using the Drive Changes "
        "API.  An optional ``page_token`` can be supplied to resume from a "
        "specific position in the change log."
    ),
    responses={
        200: {"description": "Incremental sync completed; returns stats dict"},
        500: {"description": "Sync failed"},
    },
)
async def incremental_sync(
    request: Request,
    page_token: str | None = Query(
        default=None,
        description=(
            "Optional Drive Changes API page token to resume from.  "
            "If omitted the manager will use the last saved token."
        ),
    ),
) -> dict[str, Any]:
    """Execute an incremental synchronisation of recently changed documents.

    Args:
        request: The raw FastAPI ``Request`` used to access ``app.state``.
        page_token: Optional Drive Changes API page token.  When provided,
            syncing resumes from this position rather than the last
            persisted token.

    Returns:
        A dictionary of sync statistics describing only the incremental
        changes processed.
    """
    sync_manager: SyncManager = request.app.state.sync_manager
    log = logger.bind(sync_type="incremental", page_token=page_token)
    log.info("incremental_sync_started")
    stats_result = await sync_manager.run_incremental_sync(page_token=page_token)
    if isinstance(stats_result, tuple):
        stats, new_token = stats_result
        stats["next_page_token"] = new_token
    else:
        stats = stats_result
    log.info("incremental_sync_complete", stats=stats)
    return stats


@router.get(
    "/status",
    summary="Retrieve current sync status",
    description=(
        "Queries the database to return the last sync timestamp, total number "
        "of indexed documents, and total number of stored chunks."
    ),
    responses={
        200: {"description": "Sync status retrieved successfully"},
    },
)
async def sync_status(request: Request) -> dict[str, Any]:
    """Return the current sync status derived from the database.

    Executes lightweight aggregate queries against the ``documents`` and
    ``document_chunks`` tables to produce a status snapshot.

    Args:
        request: The raw FastAPI ``Request`` used to access ``app.state``.

    Returns:
        A dictionary containing:

        - ``last_sync_at``: ISO-8601 timestamp of the most recent sync, or
          ``None`` if no sync has occurred yet.
        - ``total_documents``: Total count of indexed documents.
        - ``total_chunks``: Total count of stored document chunks.
    """
    pool: asyncpg.Pool = request.app.state.db_pool
    log = logger.bind(endpoint="sync_status")
    log.debug("fetching_sync_status")

    async with pool.acquire() as conn:
        doc_row = await conn.fetchrow(
            "SELECT COUNT(*) AS total_documents, MAX(synced_at) AS last_sync_at "
            "FROM documents;"
        )
        chunk_row = await conn.fetchrow(
            "SELECT COUNT(*) AS total_chunks FROM document_chunks;"
        )

    last_sync_at = doc_row["last_sync_at"]
    status: dict[str, Any] = {
        "last_sync_at": last_sync_at.isoformat() if last_sync_at else None,
        "total_documents": doc_row["total_documents"],
        "total_chunks": chunk_row["total_chunks"],
    }
    log.info("sync_status_fetched", **status)
    return status


@router.get(
    "/documents",
    summary="List all indexed documents",
    description="Returns metadata and chunk count for each indexed document in PostgreSQL.",
)
async def list_documents(request: Request) -> list[dict[str, Any]]:
    """Return all indexed documents with their chunk counts."""
    pool: asyncpg.Pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.doc_id, d.title, d.source_url, d.mime_type, d.synced_at, COUNT(c.chunk_id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c ON d.doc_id = c.doc_id
            GROUP BY d.doc_id, d.title, d.source_url, d.mime_type, d.synced_at
            ORDER BY d.title ASC;
            """
        )
    return [
        {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "source_url": r["source_url"],
            "mime_type": r["mime_type"],
            "synced_at": r["synced_at"].isoformat() if r["synced_at"] else None,
            "chunk_count": r["chunk_count"],
        }
        for r in rows
    ]
