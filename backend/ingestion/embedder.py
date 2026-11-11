"""Gemini embedding client for the RAG ingestion pipeline.

Wraps the ``google.generativeai`` SDK to provide async batch embedding of
:class:`~backend.models.document.DocumentChunk` objects and single-query
embedding for retrieval.
"""

from __future__ import annotations

import asyncio
from typing import Any

import google.generativeai as genai
import structlog

from backend.core.config import get_settings
from backend.models.document import DocumentChunk

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_BATCH_SIZE = 100  # Gemini embed_content batch limit


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Raised when an embedding API call fails or returns unexpected data."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialise EmbeddingError.

        Args:
            message: Human-readable description of the failure.
            cause: The underlying exception that triggered this error, if any.
        """
        super().__init__(message)
        self.cause = cause


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class GeminiEmbedder:
    """Async Gemini embedding client for documents and queries.

    Uses the ``google.generativeai`` SDK with the model and API key
    configured in application settings.  Document chunks are embedded with
    ``task_type="RETRIEVAL_DOCUMENT"`` and query strings with
    ``task_type="RETRIEVAL_QUERY"`` for asymmetric retrieval quality.

    Example::

        embedder = GeminiEmbedder()
        chunks = await embedder.embed_chunks(chunks)
        query_vec = await embedder.embed_query("What is the refund policy?")
    """

    def __init__(self) -> None:
        """Configure the Gemini client with the application API key.

        Reads ``GEMINI_API_KEY`` and ``GEMINI_EMBEDDING_MODEL`` from
        :func:`~backend.core.config.get_settings`.

        Raises:
            EmbeddingError: If the API key is missing or empty.
        """
        settings = get_settings()
        api_key: str = settings.GEMINI_API_KEY
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY setting is empty.")

        genai.configure(api_key=api_key)
        self._model: str = settings.GEMINI_EMBEDDING_MODEL
        self._expected_dim: int = settings.EMBEDDING_DIMENSION

        logger.info(
            "GeminiEmbedder initialised",
            model=self._model,
            expected_dim=self._expected_dim,
        )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def embed_chunks(
        self, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Embed a list of document chunks in batches of 100.

        Calls ``genai.embed_content`` with ``task_type="RETRIEVAL_DOCUMENT"``
        for each batch.  The ``embedding`` field of each
        :class:`~backend.models.document.DocumentChunk` is populated in-place
        and the updated list is returned.

        Args:
            chunks: Document chunks to embed.  Must be non-empty.

        Returns:
            The same list of chunks with ``embedding`` populated.

        Raises:
            EmbeddingError: If the API call fails or the returned embeddings
                do not match the expected batch size or dimension.
        """
        log = logger.bind(method="embed_chunks", total=len(chunks), model=self._model)
        log.info("Starting batch embedding")

        if not chunks:
            log.warning("embed_chunks called with empty list — returning immediately")
            return chunks

        updated: list[DocumentChunk] = []

        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            batch_log = log.bind(batch_start=batch_start, batch_size=len(batch))
            batch_log.debug("Embedding batch")

            contents = [c.content for c in batch]

            try:
                response: Any = await asyncio.to_thread(
                    genai.embed_content,
                    model=self._model,
                    content=contents,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._expected_dim,
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"genai.embed_content failed for batch starting at {batch_start}: {exc}",
                    cause=exc,
                ) from exc

            embeddings: list[list[float]] = self._parse_response(response, expected_count=len(batch))

            for chunk, embedding in zip(batch, embeddings, strict=False):
                # Pydantic v2 — use model_copy for immutability or mutate directly
                chunk.embedding = embedding
                updated.append(chunk)

            batch_log.debug("Batch embedded successfully")

        log.info("All chunks embedded", total_embedded=len(updated))
        return updated

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval.

        Uses ``task_type="RETRIEVAL_QUERY"`` which is paired asymmetrically
        with ``RETRIEVAL_DOCUMENT`` embeddings for better retrieval quality.

        Args:
            query: The natural-language query string to embed.

        Returns:
            A list of floats representing the query embedding vector.

        Raises:
            EmbeddingError: If the API call fails or returns no embedding.
        """
        log = logger.bind(method="embed_query", model=self._model, query_len=len(query))
        log.debug("Embedding query")

        if not query.strip():
            raise EmbeddingError("Query string is empty — cannot embed.")

        try:
            response: Any = await asyncio.to_thread(
                genai.embed_content,
                model=self._model,
                content=query,
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._expected_dim,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"genai.embed_content failed for query embedding: {exc}", cause=exc
            ) from exc

        embedding = self._extract_single_embedding(response)
        log.debug("Query embedded", dim=len(embedding))
        return embedding

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(
        self, response: Any, expected_count: int
    ) -> list[list[float]]:
        """Extract and validate a batch embedding response.

        Args:
            response: The raw response object from ``genai.embed_content``.
            expected_count: Expected number of embedding vectors.

        Returns:
            A list of float vectors, one per input content item.

        Raises:
            EmbeddingError: If the response is malformed or the count/dimension
                does not match expectations.
        """
        # Handle response shapes returned by google.generativeai SDK
        embeddings: list[list[float]] = []
        if isinstance(response, dict):
            raw = response.get("embeddings") or response.get("embedding")
            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], list):
                    # Batch of vectors: list of lists
                    embeddings = raw
                elif isinstance(raw[0], (int, float)):
                    # Single vector: list of numbers
                    embeddings = [raw]
                elif hasattr(raw[0], "values"):
                    embeddings = [list(e.values) for e in raw]
            elif raw is not None:
                embeddings = [raw]
        elif hasattr(response, "embeddings") and isinstance(response.embeddings, list):
            embeddings = [
                list(e.values) if hasattr(e, "values") else list(e)
                for e in response.embeddings
            ]
        elif hasattr(response, "embedding"):
            raw = response.embedding
            if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
                embeddings = raw
            else:
                embeddings = [list(raw.values) if hasattr(raw, "values") else list(raw)]
        else:
            raise EmbeddingError(
                f"Unexpected embed_content response shape: {type(response)}"
            )

        if len(embeddings) != expected_count:
            raise EmbeddingError(
                f"Expected {expected_count} embeddings, got {len(embeddings)}."
            )

        for i, vec in enumerate(embeddings):
            if len(vec) != self._expected_dim:
                raise EmbeddingError(
                    f"Embedding {i} has dimension {len(vec)}, "
                    f"expected {self._expected_dim}."
                )

        return embeddings

    def _extract_single_embedding(self, response: Any) -> list[float]:
        """Extract a single embedding vector from a query embed response.

        Args:
            response: The raw response object from ``genai.embed_content``.

        Returns:
            A single float vector.

        Raises:
            EmbeddingError: If no embedding can be found in the response.
        """
        if hasattr(response, "embedding"):
            raw = response.embedding
            vec = raw.values if hasattr(raw, "values") else raw
            return list(vec[0]) if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list) else list(vec)
        if isinstance(response, dict) and "embedding" in response:
            vec = response["embedding"]
            return list(vec[0]) if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list) else list(vec)
        # Fallback: batch response with one item
        embeddings = self._parse_response(response, expected_count=1)
        return embeddings[0]
