"""Hybrid retriever combining dense vector search and sparse BM25/full-text search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both retrieval strategies,
providing robust, complementary document ranking.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import asyncpg
import structlog

from backend.models import DocumentChunk

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


class RetrievalError(Exception):
    """Raised when a retrieval operation fails (DB error, embedding error, etc.)."""


class HybridRetriever:
    """Combines dense vector retrieval and sparse full-text search via RRF fusion.

    Dense retrieval leverages pgvector cosine similarity to find semantically
    related chunks. Sparse retrieval uses PostgreSQL full-text search (tsvector /
    plainto_tsquery) for exact keyword coverage. Results from both are merged
    with Reciprocal Rank Fusion (RRF) so that chunks appearing highly in either
    list are promoted.

    Attributes:
        db_pool: An ``asyncpg.Pool`` connected to the target PostgreSQL database.
        embedder: Any object exposing ``async embed_query(query: str) -> list[float]``.
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        embedder: Any,
        rrf_k: int = 60,
    ) -> None:
        """Initialise the retriever.

        Args:
            db_pool: Async connection pool to the PostgreSQL instance with pgvector.
            embedder: Embedder instance with an ``embed_query`` coroutine.
            rrf_k: Reciprocal Rank Fusion smoothing constant (default 60).
        """
        self._pool = db_pool
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._log = log.bind(component="HybridRetriever")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: asyncpg.Record) -> DocumentChunk:
        """Deserialise an asyncpg ``Record`` into a ``DocumentChunk``.

        Args:
            row: A raw database record from document_chunks.

        Returns:
            Fully populated ``DocumentChunk`` instance.
        """
        raw_emb = row["embedding"]
        parsed_emb: list[float] | None = None
        if raw_emb is not None:
            if isinstance(raw_emb, str):
                parsed_emb = [float(x.strip()) for x in raw_emb.strip("[]").split(",") if x.strip()]
            else:
                parsed_emb = [float(x) for x in raw_emb]

        return DocumentChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            doc_title=row["doc_title"],
            source_url=row["source_url"],
            section_header=row["section_header"],
            content=row["content"],
            token_count=row["token_count"],
            chunk_index=row["chunk_index"],
            allowed_emails=list(row["allowed_emails"] or []),
            allowed_domain=row["allowed_domain"],
            embedding=parsed_emb,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def _dense_search(
        self,
        conn: asyncpg.Connection,
        vector: list[float],
        user_email: str,
        top_k: int,
    ) -> list[DocumentChunk]:
        """Execute pgvector cosine-distance nearest-neighbour search.

        Args:
            conn: Active asyncpg connection.
            vector: Query embedding vector.
            user_email: Caller's email address used for access-control filtering.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of ``DocumentChunk`` objects (closest first).

        Raises:
            RetrievalError: On any database-level error.
        """
        sql = """
            SELECT
                chunk_id, doc_id, doc_title, source_url, section_header,
                content, token_count, chunk_index,
                allowed_emails, allowed_domain, embedding,
                created_at, updated_at
            FROM document_chunks
            WHERE
                (allowed_emails @> ARRAY[$1]::text[]
                 OR allowed_domain IS NOT NULL)
            ORDER BY embedding <=> $2
            LIMIT $3
        """
        try:
            rows = await conn.fetch(sql, user_email, vector, top_k)
        except Exception as exc:
            self._log.error("dense_search_failed", error=str(exc))
            raise RetrievalError(f"Dense search failed: {exc}") from exc

        return [self._row_to_chunk(r) for r in rows]

    async def _sparse_search(
        self,
        conn: asyncpg.Connection,
        query: str,
        user_email: str,
        top_k: int,
    ) -> list[DocumentChunk]:
        """Execute PostgreSQL full-text search with ts_rank ordering.

        Args:
            conn: Active asyncpg connection.
            query: Raw text query that will be passed to ``plainto_tsquery``.
            user_email: Caller's email address used for access-control filtering.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of ``DocumentChunk`` objects (highest ts_rank first).

        Raises:
            RetrievalError: On any database-level error.
        """
        sql = """
            SELECT
                chunk_id, doc_id, doc_title, source_url, section_header,
                content, token_count, chunk_index,
                allowed_emails, allowed_domain, embedding,
                created_at, updated_at
            FROM document_chunks
            WHERE
                (allowed_emails @> ARRAY[$1]::text[]
                 OR allowed_domain IS NOT NULL)
                AND to_tsvector('english', content) @@ plainto_tsquery('english', $2)
            ORDER BY
                ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) DESC
            LIMIT $3
        """
        try:
            rows = await conn.fetch(sql, user_email, query, top_k)
        except Exception as exc:
            self._log.error("sparse_search_failed", error=str(exc))
            raise RetrievalError(f"Sparse search failed: {exc}") from exc

        return [self._row_to_chunk(r) for r in rows]

    @staticmethod
    def _rrf_fusion(
        ranked_lists: list[list[DocumentChunk]],
        top_k: int,
        k: int = 60,
    ) -> list[DocumentChunk]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

        Score for each chunk: ``sum(1 / (k + rank))`` across all lists where
        rank is 1-indexed. Higher combined score = better rank.

        Args:
            ranked_lists: One or more ordered lists of ``DocumentChunk`` objects.
            top_k: Number of top results to return after fusion.
            k: RRF smoothing constant (default 60, as in the original paper).

        Returns:
            Fused and re-ranked list of unique ``DocumentChunk`` objects.
        """
        scores: dict[str, float] = defaultdict(float)
        chunks_by_id: dict[str, DocumentChunk] = {}

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                scores[chunk.chunk_id] += 1.0 / (k + rank)
                # Keep the first occurrence so we don't lose metadata.
                chunks_by_id.setdefault(chunk.chunk_id, chunk)

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [chunks_by_id[cid] for cid in sorted_ids[:top_k]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        user_email: str,
        top_k: int,
    ) -> list[DocumentChunk]:
        """Retrieve and fuse relevant document chunks for *query*.

        1. Embeds *query* asynchronously.
        2. Launches dense and sparse DB searches **in parallel**.
        3. Merges results via RRF and returns the top *top_k* chunks.

        Args:
            query: Natural-language query string.
            user_email: Email of the requesting user (access control).
            top_k: Number of fused results to return.

        Returns:
            Ranked list of ``DocumentChunk`` objects, most relevant first.

        Raises:
            RetrievalError: If embedding or any database query fails.
        """
        self._log.info("retrieve_start", query=query, user_email=user_email, top_k=top_k)

        # 1. Embed the query.
        try:
            vector: list[float] = await self._embedder.embed_query(query)
        except Exception as exc:
            self._log.error("embedding_failed", error=str(exc))
            raise RetrievalError(f"Query embedding failed: {exc}") from exc

        # 2. Run both searches in parallel using the same connection per search.
        async with self._pool.acquire() as conn_dense, self._pool.acquire() as conn_sparse:
            dense_task = self._dense_search(conn_dense, vector, user_email, top_k)
            sparse_task = self._sparse_search(conn_sparse, query, user_email, top_k)

            dense_results, sparse_results = await asyncio.gather(
                dense_task, sparse_task, return_exceptions=False
            )

        self._log.info(
            "retrieve_raw_counts",
            dense=len(dense_results),
            sparse=len(sparse_results),
        )

        # 3. Fuse with RRF.
        fused = self._rrf_fusion(
            [dense_results, sparse_results],
            top_k=top_k,
            k=self._rrf_k,
        )

        self._log.info("retrieve_done", fused_count=len(fused))
        return fused
