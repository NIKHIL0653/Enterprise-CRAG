"""Application configuration module for WorkspaceRAG.

This module provides a centralized, validated settings object built on top of
Pydantic v2 ``BaseSettings``.  Values are loaded from environment variables
and/or a ``.env`` file at the project root.

Example:
    >>> from backend.core.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.APP_NAME)
    WorkspaceRAG
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised application settings loaded from environment / .env file.

    All fields without a default value **must** be provided via the environment
    or the ``.env`` file, otherwise the application will fail to start with a
    clear validation error.

    Attributes:
        APP_NAME: Human-readable application name.
        APP_VERSION: Semantic version string of the running service.
        DEBUG: Enable debug mode (verbose tracebacks, reload, etc.).
        LOG_LEVEL: Python logging level name (e.g. ``"INFO"``, ``"DEBUG"``).

        GOOGLE_SERVICE_ACCOUNT_JSON: Absolute (or project-relative) path to
            the Google service-account JSON key file used for Drive access.
        GOOGLE_DRIVE_FOLDER_ID: ID of the top-level Google Drive folder that
            the ingestion pipeline watches.

        GEMINI_API_KEY: API key for Gemini (Google AI Studio / Vertex AI).
        GEMINI_MODEL: Gemini model identifier used for chat/generation.
        GEMINI_EMBEDDING_MODEL: Gemini model identifier used for embeddings.
        EMBEDDING_DIMENSION: Dimensionality of the embedding vectors produced
            by ``GEMINI_EMBEDDING_MODEL``.

        DATABASE_URL: SQLAlchemy-compatible Postgres connection string that
            must point to a database with the ``pgvector`` extension enabled.

        TOP_K_RETRIEVAL: Number of candidate chunks fetched from the vector
            store before reranking.
        TOP_K_RERANK: Number of chunks kept after the Cohere reranking step.
        RRF_K: Constant *k* used in Reciprocal Rank Fusion (RRF) scoring.
        RELEVANCE_THRESHOLD: Minimum Cohere relevance score for a chunk to be
            included in the final context window.

        COHERE_API_KEY: API key for the Cohere reranking endpoint.

        LANGFUSE_SECRET_KEY: Langfuse project secret key for observability.
        LANGFUSE_PUBLIC_KEY: Langfuse project public key.
        LANGFUSE_HOST: Base URL of the Langfuse server.

        SYNC_POLL_INTERVAL_SECONDS: How often (in seconds) the background
            sync worker polls Google Drive for changes.
    """

    # ------------------------------------------------------------------ #
    # Application                                                           #
    # ------------------------------------------------------------------ #
    APP_NAME: str = "WorkspaceRAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Google                                                                #
    # ------------------------------------------------------------------ #
    GOOGLE_SERVICE_ACCOUNT_JSON: str = Field(
        ...,
        description="Path to the Google service-account JSON key file.",
    )
    GOOGLE_DRIVE_FOLDER_ID: str = Field(
        ...,
        description="Google Drive folder ID to watch for documents.",
    )

    # ------------------------------------------------------------------ #
    # LLM / Embeddings                                                      #
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API key.")
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    EMBEDDING_DIMENSION: int = 768

    # ------------------------------------------------------------------ #
    # Database                                                              #
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = Field(
        ...,
        description=(
            "Async Postgres connection string, e.g. "
            "postgresql+asyncpg://user:pass@host/dbname"
        ),
    )

    # ------------------------------------------------------------------ #
    # Retrieval                                                             #
    # ------------------------------------------------------------------ #
    TOP_K_RETRIEVAL: int = 15
    TOP_K_RERANK: int = 5
    RRF_K: int = 60
    RELEVANCE_THRESHOLD: float = 0.5

    # ------------------------------------------------------------------ #
    # Cohere                                                                #
    # ------------------------------------------------------------------ #
    COHERE_API_KEY: str = Field(..., description="Cohere API key for reranking.")

    # ------------------------------------------------------------------ #
    # Langfuse                                                              #
    # ------------------------------------------------------------------ #
    LANGFUSE_SECRET_KEY: str = Field(..., description="Langfuse project secret key.")
    LANGFUSE_PUBLIC_KEY: str = Field(..., description="Langfuse project public key.")
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ------------------------------------------------------------------ #
    # Query Cache                                                           #
    # ------------------------------------------------------------------ #
    CACHE_ENABLED: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = 0.16

    # ------------------------------------------------------------------ #
    # Background sync                                                       #
    # ------------------------------------------------------------------ #
    SYNC_POLL_INTERVAL_SECONDS: int = 60

    model_config = SettingsConfigDict(env_file=(".env", ".env.test"), extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` singleton.

    The instance is constructed once and cached for the lifetime of the process
    using :func:`functools.lru_cache`.  This avoids repeatedly parsing the
    ``.env`` file and re-validating values on every call-site.

    Returns:
        A fully-validated :class:`Settings` instance populated from environment
        variables and/or the ``.env`` file.

    Raises:
        pydantic.ValidationError: If one or more required environment variables
            are missing or fail validation.

    Example:
        >>> settings = get_settings()
        >>> assert settings.APP_NAME == "WorkspaceRAG"
    """
    return Settings()
