"""Cohere-based cross-encoder reranker for retrieved document chunks.

Calls the Cohere Rerank v3.5 API to score each candidate chunk against the
query and returns the top-N highest-scoring chunks, preserving all original
``DocumentChunk`` metadata.
"""
from __future__ import annotations

import cohere
import structlog

from backend.core.config import get_settings
from backend.models import DocumentChunk
from backend.retrieval.hybrid_retriever import RetrievalError

log = structlog.get_logger(__name__)

_COHERE_RERANK_MODEL = "rerank-v3.5"


class CohereReranker:
    """Cross-encoder reranker powered by the Cohere Rerank API.

    Submits candidate document chunks to the Cohere Rerank v3.5 endpoint,
    which scores each document relative to the query using a cross-encoder
    model. Results are sorted descending by relevance score, and the top-N
    ``DocumentChunk`` objects (with all metadata intact) are returned.

    Attributes:
        _client: Async Cohere client instance.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the reranker with an async Cohere client.

        Args:
            api_key: Cohere API key. If not provided, sourced lazily from
                application settings.
        """
        cohere_key = api_key or get_settings().COHERE_API_KEY
        self._client: cohere.AsyncClient = cohere.AsyncClient(
            api_key=cohere_key
        )
        self._log = log.bind(component="CohereReranker", model=_COHERE_RERANK_MODEL)

    async def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_n: int,
    ) -> list[DocumentChunk]:
        """Rerank *chunks* against *query* using Cohere Rerank v3.5.

        The function sends the raw ``content`` text of each chunk to Cohere
        and uses the returned relevance scores to re-order the original
        ``DocumentChunk`` objects (which retain all their metadata fields).

        Args:
            query: The natural-language query to rank documents against.
            chunks: Candidate ``DocumentChunk`` objects to be reranked. Must
                be non-empty.
            top_n: Maximum number of reranked results to return.

        Returns:
            Up to *top_n* ``DocumentChunk`` objects sorted by Cohere relevance
            score (highest first).

        Raises:
            RetrievalError: If the Cohere API call fails for any reason.
        """
        if not chunks:
            self._log.warning("rerank_called_with_empty_chunks")
            return []

        self._log.info(
            "rerank_start",
            query=query,
            num_chunks=len(chunks),
            top_n=top_n,
        )

        documents = [chunk.content for chunk in chunks]

        try:
            response = await self._client.rerank(
                model=_COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        except cohere.CohereAPIError as exc:
            self._log.error("cohere_api_error", status_code=exc.status_code, error=str(exc))
            raise RetrievalError(f"Cohere rerank API error: {exc}") from exc
        except Exception as exc:
            self._log.error("cohere_unexpected_error", error=str(exc))
            raise RetrievalError(f"Cohere rerank unexpected failure: {exc}") from exc

        # Sort by relevance_score descending and map back to original chunks.
        reranked_results = sorted(
            response.results,
            key=lambda r: r.relevance_score,
            reverse=True,
        )

        reranked_chunks: list[DocumentChunk] = [
            chunks[result.index] for result in reranked_results
        ]

        self._log.info(
            "rerank_done",
            returned=len(reranked_chunks),
            top_score=reranked_results[0].relevance_score if reranked_results else None,
        )
        return reranked_chunks
