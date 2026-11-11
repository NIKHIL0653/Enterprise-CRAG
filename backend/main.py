"""FastAPI application entry point for WorkspaceRAG.

This module creates and configures the FastAPI application, wires up the
lifespan context manager (startup/shutdown), registers routers, and applies
all middleware layers.

Usage::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.agent import WorkspaceRAGGraph
from backend.api.middleware import RequestLoggingMiddleware, add_error_handlers
from backend.api.v1 import query_router, sync_router
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.ingestion import SyncManager
from backend.services.database import close_pool, create_pool, run_migrations

logger = get_logger(__name__)

settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application-wide resources for startup and shutdown.

    **Startup sequence:**

    1. Create the asyncpg connection pool.
    2. Execute DDL migrations (idempotent).
    3. Instantiate infrastructure clients:
       - ``GoogleDriveClient``
       - ``SemanticMarkdownChunker``
       - ``GeminiEmbedder``
       - ``HybridRetriever``
       - ``CohereReranker``
    4. Instantiate the high-level services:
       - ``SyncManager``
       - ``WorkspaceRAGGraph``
    5. Attach all instances to ``app.state`` for injection in route handlers.

    **Shutdown sequence:**

    * Closes the asyncpg connection pool cleanly.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the ASGI server while the application is running.
    """
    # ----- Startup -----
    logger.info("application_startup_begin", version=settings.APP_VERSION)

    # 1. Database pool
    db_pool = await create_pool()
    app.state.db_pool = db_pool

    # 2. Migrations
    await run_migrations(db_pool)

    # 3. Infrastructure clients
    import google.generativeai as genai
    from langfuse import Langfuse

    from backend.ingestion.chunker import SemanticMarkdownChunker
    from backend.ingestion.embedder import GeminiEmbedder
    from backend.ingestion.google_drive_client import GoogleDriveClient
    from backend.retrieval.hybrid_retriever import HybridRetriever
    from backend.retrieval.reranker import CohereReranker

    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm_client = genai.GenerativeModel(settings.GEMINI_MODEL)
    langfuse_client = Langfuse(
        secret_key=settings.LANGFUSE_SECRET_KEY,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        host=settings.LANGFUSE_HOST,
    )

    drive_client = GoogleDriveClient()
    chunker = SemanticMarkdownChunker()
    embedder = GeminiEmbedder()
    retriever = HybridRetriever(db_pool=db_pool, embedder=embedder, rrf_k=settings.RRF_K)
    reranker = CohereReranker()

    # 3b. Query cache (two-tier: SHA-256 exact + pgvector semantic)
    from backend.services.cache import QueryCacheManager
    cache_manager = QueryCacheManager(pool=db_pool, embedder=embedder)

    app.state.drive_client = drive_client
    app.state.chunker = chunker
    app.state.embedder = embedder
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.cache_manager = cache_manager

    # 4. High-level services
    sync_manager = SyncManager(
        db_pool=db_pool,
        drive_client=drive_client,
        chunker=chunker,
        embedder=embedder,
    )
    rag_graph = WorkspaceRAGGraph(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
        langfuse_client=langfuse_client,
    )
    rag_graph.build_graph()

    app.state.sync_manager = sync_manager
    app.state.rag_graph = rag_graph

    logger.info("application_startup_complete")

    yield  # ← application is running

    # ----- Shutdown -----
    logger.info("application_shutdown_begin")
    await close_pool(db_pool)
    logger.info("application_shutdown_complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        A fully configured :class:`FastAPI` instance ready to be served.
    """
    app = FastAPI(
        title="WorkspaceRAG API",
        version=settings.APP_VERSION,
        description=(
            "Retrieval-Augmented Generation over your Google Workspace documents. "
            "Supports hybrid semantic + full-text search with Cohere reranking."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- CORS ---
    # Allow all origins in development; restrict via ALLOWED_ORIGINS in prod.
    allowed_origins: list[str] = getattr(settings, "ALLOWED_ORIGINS", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Request logging (must be added after CORS) ---
    app.add_middleware(RequestLoggingMiddleware)

    # --- Global error handlers ---
    add_error_handlers(app)

    # --- Routers ---
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(sync_router, prefix="/api/v1")

    # --- Web UI & Root endpoints ---
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"

    @app.get("/chat", tags=["UI"], summary="Web Chat Interface")
    async def chat_ui():
        """Serve the interactive web chat interface."""
        if frontend_path.exists():
            return FileResponse(frontend_path)
        return {"message": "Chat frontend not found"}

    @app.get("/", tags=["Meta"], summary="API root")
    async def root():
        """Return the Web UI or API information."""
        if frontend_path.exists():
            return FileResponse(frontend_path)
        return {
            "title": "WorkspaceRAG API",
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "redoc": "/redoc",
            "chat": "/chat",
        }

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn / gunicorn)
# ---------------------------------------------------------------------------

app: FastAPI = create_app()
