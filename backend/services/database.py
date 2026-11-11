"""Database pool management using asyncpg.

This module provides utilities to create, manage, and migrate a PostgreSQL
connection pool for the WorkspaceRAG backend. It uses pgvector and pg_trgm
extensions for hybrid (semantic + full-text) search.
"""

from __future__ import annotations

import asyncpg

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQL migration script
# ---------------------------------------------------------------------------

_MIGRATION_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT        NOT NULL,
    source_url      TEXT        NOT NULL,
    mime_type       TEXT        NOT NULL,
    owner_email     TEXT        NOT NULL,
    allowed_emails  TEXT[]      NOT NULL DEFAULT '{}',
    allowed_domain  TEXT,
    last_modified   TIMESTAMPTZ NOT NULL,
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT        NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    doc_title       TEXT        NOT NULL,
    source_url      TEXT        NOT NULL,
    section_header  TEXT,
    content         TEXT        NOT NULL,
    token_count     INT         NOT NULL,
    chunk_index     INT         NOT NULL,
    allowed_emails  TEXT[]      NOT NULL DEFAULT '{}',
    allowed_domain  TEXT,
    embedding       vector(768),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON document_chunks USING gin(to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON document_chunks(doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_allowed_emails
    ON document_chunks USING gin(allowed_emails);

CREATE TABLE IF NOT EXISTS rag_query_cache (
    cache_id        TEXT PRIMARY KEY,
    query_text      TEXT NOT NULL,
    query_hash      TEXT NOT NULL,
    user_email      TEXT NOT NULL,
    embedding       vector(768) NOT NULL,
    answer          TEXT NOT NULL,
    citations       JSONB NOT NULL DEFAULT '[]',
    tokens_used     INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_cache_hash
    ON rag_query_cache(query_hash);

CREATE INDEX IF NOT EXISTS idx_query_cache_embedding
    ON rag_query_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);
"""


# ---------------------------------------------------------------------------
# Pool lifecycle helpers
# ---------------------------------------------------------------------------


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Initialize connection with pgvector codec."""
    try:
        import pgvector.asyncpg
        await pgvector.asyncpg.register_vector(conn)
    except Exception as exc:
        logger.warning("pgvector_codec_register_failed", error=str(exc))


async def create_pool() -> asyncpg.Pool:
    """Create and return an asyncpg connection pool.

    Reads ``DATABASE_URL`` from application settings.

    Returns:
        asyncpg.Pool: A fully initialised connection pool ready for use.

    Raises:
        asyncpg.PostgresError: If the pool cannot be established.
    """
    settings = get_settings()
    log = logger.bind(database_url=settings.DATABASE_URL)

    log.info("creating_asyncpg_pool", min_size=5, max_size=20)
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=5,
        max_size=20,
        init=_init_connection,
    )
    log.info("asyncpg_pool_created")
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    """Gracefully close an asyncpg connection pool.

    Args:
        pool: The asyncpg pool to close.

    Raises:
        asyncpg.PostgresError: If the pool cannot be closed cleanly.
    """
    logger.info("closing_asyncpg_pool")
    await pool.close()
    logger.info("asyncpg_pool_closed")


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Execute DDL migrations against the database.

    Creates all required extensions, tables, and indexes if they do not
    already exist. Uses an advisory lock to prevent race conditions during
    multi-process application startup.

    Args:
        pool: An open asyncpg connection pool.

    Raises:
        asyncpg.PostgresError: If any DDL statement fails.
    """
    logger.info("running_migrations")
    async with pool.acquire() as conn:
        # Advisory lock prevents parallel workers from colliding on extension creation
        await conn.execute("SELECT pg_advisory_lock(71423891);")
        try:
            await conn.execute(_MIGRATION_SQL)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(71423891);")
    logger.info("migrations_complete")
