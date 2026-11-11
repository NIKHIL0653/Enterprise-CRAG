"""Query cache management for the WorkspaceRAG backend.

This module provides :class:`QueryCacheManager`, which persists and retrieves
LLM answers keyed by both an exact SHA-256 hash **and** a cosine-similarity
embedding match against the ``rag_query_cache`` PostgreSQL table.

Lookup strategy
---------------
1. **Exact match** – a deterministic SHA-256 hash of ``user_email:normalized_question``
   is compared against the ``query_hash`` column.  Zero latency embedding cost.
2. **Semantic match** – when the exact hash misses, the question is embedded
   with :class:`~backend.ingestion.embedder.GeminiEmbedder` and the nearest
   cached embedding is retrieved via pgvector's cosine distance operator
   ``<=>`` with a threshold of ``0.08``.

Both ``lookup`` and ``store`` normalise the raw question (lowercase, strip,
collapse internal whitespace) before hashing/embedding so that trivially
equivalent phrasings share a single cache entry.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import asyncpg

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.ingestion.embedder import GeminiEmbedder

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default cosine-distance threshold (~84%+ semantic similarity).
_DEFAULT_SEMANTIC_THRESHOLD: float = 0.16

#: SQL used for the semantic-similarity lookup.
_SEMANTIC_SQL = """
    SELECT
        answer,
        citations,
        tokens_used,
        created_at,
        (embedding <=> $2) AS distance
    FROM rag_query_cache
    WHERE user_email = $1
      AND (embedding <=> $2) < $3
    ORDER BY embedding <=> $2
    LIMIT 1
"""

#: SQL used for the exact-hash lookup.
_EXACT_SQL = """
    SELECT
        answer,
        citations,
        tokens_used,
        created_at
    FROM rag_query_cache
    WHERE query_hash = $1
      AND user_email  = $2
    LIMIT 1
"""

#: SQL used to persist a new cache entry.
_INSERT_SQL = """
    INSERT INTO rag_query_cache
        (cache_id, query_text, query_hash, user_email, embedding,
         answer, citations, tokens_used)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    ON CONFLICT (cache_id) DO UPDATE
    SET answer = EXCLUDED.answer,
        citations = EXCLUDED.citations,
        embedding = EXCLUDED.embedding,
        tokens_used = EXCLUDED.tokens_used,
        created_at = NOW()
"""

#: SQL used to bulk-invalidate all cached answers (e.g. after a document sync).
_DELETE_ALL_SQL = "DELETE FROM rag_query_cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_normalize(question: str) -> str:
    """Return a canonicalized representation of *question* for exact matching.

    Applies in order:
    1. Fold to lowercase.
    2. Strip punctuation and special characters.
    3. Remove common conversational question prefixes ('what is', 'explain', etc.).
    4. Remove standalone articles ('a', 'an', 'the').
    5. Collapse whitespace and trim.

    Examples:
        - "What is a transformer?" -> "transformer"
        - "what is transformer?"   -> "transformer"
        - "Explain the transformer" -> "transformer"
    """
    q = question.strip().lower()
    # Strip punctuation and symbols
    q = re.sub(r"[?!.,\"';:\-_/\\()\[\]{}]+", " ", q)
    # Strip common conversational question prefixes
    prefix_pattern = (
        r"^(can you please |please |can you |tell me about |what is |what are |"
        r"what was |what were |explain |describe |how does |what does |give me |define )\s*"
    )
    q = re.sub(prefix_pattern, "", q)
    # Strip articles
    q = re.sub(r"\b(a|an|the)\b", " ", q)
    # Collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    return q or question.strip().lower()


def _sha256_hex(user_email: str, normalized_question: str) -> str:
    """Compute the SHA-256 hex digest of ``user_email:normalized_question``."""
    raw = f"{user_email}:{normalized_question}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class QueryCacheManager:
    """Semantic + exact-match cache for RAG query results.

    Uses the ``rag_query_cache`` PostgreSQL table (created by the migration in
    :mod:`backend.services.database`) to store and retrieve LLM answers.

    Args:
        pool: An open :class:`asyncpg.Pool` shared with the rest of the app.
        embedder: An initialised :class:`~backend.ingestion.embedder.GeminiEmbedder`
            instance used to embed questions for semantic lookup and storage.

    Example::

        cache = QueryCacheManager(pool=app.state.pool, embedder=app.state.embedder)

        hit = await cache.lookup(question="What is PEP 8?", user_email="alice@example.com")
        if hit:
            return hit["answer"]

        answer, citations, tokens = await run_rag_pipeline(question)
        await cache.store(
            question=question,
            user_email="alice@example.com",
            answer=answer,
            citations=citations,
            tokens_used=tokens,
        )
    """

    def __init__(self, pool: asyncpg.Pool, embedder: GeminiEmbedder) -> None:
        self._pool = pool
        self._embedder = embedder
        self._log = logger.bind(component="QueryCacheManager")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def lookup(self, question: str, user_email: str) -> dict[str, Any] | None:
        """Return a cached answer for *question* / *user_email*, or ``None``.

        The method first attempts an **exact match** using the SHA-256 hash of
        the normalised question.  If that misses it falls back to a **semantic
        match** via pgvector cosine similarity.

        Args:
            question: The raw user question (may contain extra whitespace,
                mixed casing, etc.).
            user_email: E-mail address of the authenticated user.  Cache entries
                are user-scoped so that answers honouring per-user ACLs are not
                leaked across accounts.

        Returns:
            A :class:`dict` on a cache hit with the following keys:

            * ``answer`` (:class:`str`) – the cached LLM response.
            * ``citations`` (:class:`list`) – list of source-chunk citation dicts.
            * ``cache_type`` (:class:`str`) – ``"exact"`` or ``"semantic"``.
            * ``latency_ms`` (:class:`float`) – time spent inside this method (ms).
            * ``tokens_saved`` (:class:`int`) – ``tokens_used`` value that was saved.

            Returns ``None`` when no suitable cached answer exists.
        """
        t0 = time.monotonic()
        canonical_norm = _canonical_normalize(question)
        canonical_hash = _sha256_hex(user_email, canonical_norm)
        log = self._log.bind(user_email=user_email, canonical_key=canonical_norm[:20])

        # ------------------------------------------------------------------ #
        # 1. Canonical exact-hash lookup – zero LLM cost, instant (< 5ms)     #
        # ------------------------------------------------------------------ #
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_EXACT_SQL, canonical_hash, user_email)

        if row is not None:
            latency_ms = (time.monotonic() - t0) * 1_000
            log.info(
                "cache_hit_exact",
                canonical_key=canonical_norm,
                latency_ms=round(latency_ms, 2),
                tokens_saved=row["tokens_used"],
            )
            return {
                "answer": row["answer"],
                "citations": json.loads(row["citations"]) if isinstance(row["citations"], str) else row["citations"],
                "cache_type": "exact",
                "latency_ms": latency_ms,
                "tokens_saved": row["tokens_used"],
            }

        # ------------------------------------------------------------------ #
        # 2. Semantic lookup via pgvector cosine distance                     #
        # ------------------------------------------------------------------ #
        log.debug("cache_miss_exact_attempting_semantic", canonical_key=canonical_norm)
        try:
            embedding: list[float] = await self._embedder.embed_query(question)
        except Exception as exc:
            log.warning("cache_embed_failed_during_lookup", error=str(exc))
            return None

        threshold = getattr(settings, "CACHE_SIMILARITY_THRESHOLD", _DEFAULT_SEMANTIC_THRESHOLD)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                _SEMANTIC_SQL,
                user_email,
                embedding,
                threshold,
            )

        if row is not None:
            latency_ms = (time.monotonic() - t0) * 1_000
            log.info(
                "cache_hit_semantic",
                distance=round(float(row["distance"]), 4),
                threshold=threshold,
                latency_ms=round(latency_ms, 2),
                tokens_saved=row["tokens_used"],
            )
            return {
                "answer": row["answer"],
                "citations": json.loads(row["citations"]) if isinstance(row["citations"], str) else row["citations"],
                "cache_type": "semantic",
                "latency_ms": latency_ms,
                "tokens_saved": row["tokens_used"],
            }

        latency_ms = (time.monotonic() - t0) * 1_000
        log.debug("cache_miss_both", latency_ms=round(latency_ms, 2))
        return None

    async def store(
        self,
        question: str,
        user_email: str,
        answer: str,
        citations: list[dict[str, Any]],
        tokens_used: int,
    ) -> None:
        """Persist an LLM answer in ``rag_query_cache`` under its canonical hash.

        Args:
            question: The raw user question.
            user_email: E-mail address of the authenticated user.
            answer: The full LLM-generated answer string.
            citations: List of source-chunk citation dicts.
            tokens_used: Total tokens consumed to produce the answer.
        """
        log = self._log.bind(user_email=user_email)

        canonical_norm = _canonical_normalize(question)
        canonical_hash = _sha256_hex(user_email, canonical_norm)

        cache_id = canonical_hash

        log.debug("cache_embedding_for_store", canonical_key=canonical_norm[:20])
        try:
            embedding: list[float] = await self._embedder.embed_query(question)
        except Exception as exc:
            log.warning("cache_embed_failed_during_store", error=str(exc))
            return

        citations_json = json.dumps(citations)

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _INSERT_SQL,
                    cache_id,
                    canonical_norm,
                    canonical_hash,
                    user_email,
                    embedding,
                    answer,
                    citations_json,
                    tokens_used,
                )
            log.info(
                "cache_stored",
                canonical_key=canonical_norm,
                tokens_used=tokens_used,
                citations_count=len(citations),
            )
        except asyncpg.PostgresError as exc:
            log.error("cache_store_db_error", error=str(exc))

    async def invalidate_for_docs(self, pool: asyncpg.Pool) -> int:
        """Delete **all** rows from ``rag_query_cache`` and return the count.

        Call this after a document ingestion / sync cycle to ensure stale
        answers derived from superseded document content are evicted.

        Args:
            pool: An open :class:`asyncpg.Pool`.  Accepts an explicit pool
                argument so the caller can pass a different pool if needed
                (e.g. a privileged connection during maintenance mode).

        Returns:
            The number of rows deleted.

        Raises:
            asyncpg.PostgresError: If the DELETE statement fails.
        """
        log = self._log
        log.info("cache_invalidating_all")
        try:
            async with pool.acquire() as conn:
                result: str = await conn.execute(_DELETE_ALL_SQL)
            # asyncpg returns a command-status string like "DELETE 42"
            deleted_count = int(result.split()[-1]) if result else 0
            log.info("cache_invalidated", deleted_count=deleted_count)
            return deleted_count
        except asyncpg.PostgresError as exc:
            log.error("cache_invalidation_db_error", error=str(exc))
            raise
