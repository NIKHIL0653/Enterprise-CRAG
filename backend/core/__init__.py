"""Public API for the ``backend.core`` package.

Re-exports the most commonly used symbols so that call-sites can use a single
clean import path::

    from backend.core import get_settings, get_logger, RetrievalError

Symbols exported:
    - :func:`~backend.core.config.get_settings`
    - :func:`~backend.core.logging.get_logger`
    - :func:`~backend.core.logging.configure_logging`
    - :class:`~backend.core.exceptions.WorkspaceRAGError`
    - :class:`~backend.core.exceptions.GoogleDriveError`
    - :class:`~backend.core.exceptions.EmbeddingError`
    - :class:`~backend.core.exceptions.RetrievalError`
    - :class:`~backend.core.exceptions.LLMError`
    - :class:`~backend.core.exceptions.AuthorizationError`
"""

from __future__ import annotations

from backend.core.config import get_settings
from backend.core.exceptions import (
    AuthorizationError,
    EmbeddingError,
    GoogleDriveError,
    LLMError,
    RetrievalError,
    WorkspaceRAGError,
)
from backend.core.logging import configure_logging, get_logger

__all__ = [
    # Config
    "get_settings",
    # Logging
    "configure_logging",
    "get_logger",
    # Exceptions
    "WorkspaceRAGError",
    "GoogleDriveError",
    "EmbeddingError",
    "RetrievalError",
    "LLMError",
    "AuthorizationError",
]
