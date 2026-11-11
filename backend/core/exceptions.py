"""Custom exception hierarchy for WorkspaceRAG.

All application-specific exceptions inherit from :class:`WorkspaceRAGError` so
that callers can catch either the specific sub-type or the base class for
broad error handling.

Example::

    from backend.core.exceptions import RetrievalError

    raise RetrievalError(
        message="Vector store query timed out.",
        details={"table": "document_chunks", "timeout_s": 30},
    )
"""

from __future__ import annotations


class WorkspaceRAGError(Exception):
    """Base exception for all WorkspaceRAG application errors.

    Args:
        message: A human-readable description of what went wrong.
        details: An optional mapping of arbitrary key/value pairs that provide
            additional structured context (e.g. identifiers, counts, upstream
            error codes).

    Attributes:
        message: The human-readable error description.
        details: Supplemental structured context.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details: dict = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Return a compact string representation of the exception.

        Returns:
            A string in the format ``"<ClassName>: <message>"``.  If
            ``details`` is non-empty, they are appended after a pipe separator.
        """
        base = f"{self.__class__.__name__}: {self.message}"
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{base} | {detail_str}"
        return base

    def to_dict(self) -> dict:
        """Serialise the exception to a plain dictionary.

        Useful for returning structured error payloads in HTTP responses or
        for structured logging.

        Returns:
            A dictionary with keys ``"error_type"``, ``"message"``, and
            ``"details"``.

        Example:
            >>> err = WorkspaceRAGError("Something broke", {"code": 42})
            >>> err.to_dict()
            {'error_type': 'WorkspaceRAGError', 'message': 'Something broke', 'details': {'code': 42}}
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class GoogleDriveError(WorkspaceRAGError):
    """Raised when an operation against the Google Drive API fails.

    This covers authentication failures, missing files/folders, quota
    exhaustion, and other Drive-specific problems encountered during the
    document ingestion pipeline.

    Args:
        message: Description of the Drive API failure.
        details: Optional structured context (e.g. ``file_id``, HTTP status).

    Example::

        raise GoogleDriveError(
            "Failed to list files in folder.",
            details={"folder_id": "1AbC...", "http_status": 403},
        )
    """


class EmbeddingError(WorkspaceRAGError):
    """Raised when an embedding API call fails or returns unexpected output.

    Args:
        message: Description of the embedding failure.
        details: Optional structured context (e.g. ``model``, ``chunk_id``).

    Example::

        raise EmbeddingError(
            "Gemini embedding API returned empty vector.",
            details={"model": "models/text-embedding-004", "chunk_id": "c99"},
        )
    """


class RetrievalError(WorkspaceRAGError):
    """Raised when the retrieval layer cannot fetch or rank candidate chunks.

    This includes vector-store query failures, RRF computation errors, and
    Cohere reranking API errors.

    Args:
        message: Description of the retrieval failure.
        details: Optional structured context (e.g. ``query``, ``top_k``).

    Example::

        raise RetrievalError(
            "pgvector similarity search timed out.",
            details={"query_hash": "deadbeef", "timeout_s": 10},
        )
    """


class LLMError(WorkspaceRAGError):
    """Raised when a call to the generative LLM fails or is refused.

    Covers network errors, safety filter blocks, context-length violations,
    and other generation-time failures.

    Args:
        message: Description of the LLM failure.
        details: Optional structured context (e.g. ``model``, ``prompt_tokens``).

    Example::

        raise LLMError(
            "Gemini refused the request due to safety filters.",
            details={"model": "gemini-2.0-flash", "finish_reason": "SAFETY"},
        )
    """


class AuthorizationError(WorkspaceRAGError):
    """Raised when a user is not permitted to access a document or resource.

    This is distinct from authentication errors; the user is known but lacks
    the necessary ACL permissions for the requested operation.

    Args:
        message: Description of the authorisation failure.
        details: Optional structured context (e.g. ``user_email``, ``doc_id``).

    Example::

        raise AuthorizationError(
            "User is not in the allowed list for this document.",
            details={"user_email": "bob@example.com", "doc_id": "doc_42"},
        )
    """
