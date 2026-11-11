"""FastAPI router for RAG query endpoints.

Exposes a streaming SSE endpoint that runs the ``WorkspaceRAGGraph`` agent
pipeline and a lightweight health-check endpoint. Requests are served from an
in-process semantic/exact cache when available, falling back to the full RAG
pipeline otherwise. All pipeline runs are timed and latency is surfaced in the
final metadata SSE frame.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.agent import WorkspaceRAGGraph
from backend.core.config import get_settings
from backend.core.exceptions import AuthorizationError, WorkspaceRAGError
from backend.core.logging import get_logger
from backend.models import QueryRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_event(data: str) -> str:
    """Format a payload as a single Server-Sent Events frame.

    Args:
        data: The raw string payload to embed in the ``data:`` field.

    Returns:
        A properly formatted SSE frame string (ending with double newline).
    """
    return f"data: {data}\n\n"


async def _stream_rag_response(
    graph: WorkspaceRAGGraph,
    body: QueryRequest,
    request: Request,
) -> AsyncIterator[str]:
    """Drive the RAG graph and yield SSE frames, with cache integration.

    Checks the application-level ``cache_manager`` (if present) before
    invoking the full RAG pipeline.  On a cache hit the cached answer is
    streamed immediately and the function returns early.  On a cache miss the
    pipeline is executed, the result is stored in the cache as a fire-and-
    forget background task, and timing metadata is included in the final frame.

    Streams token-by-token text chunks from the agent.  After all tokens have
    been yielded the function emits a final ``[DONE]`` frame followed by a
    JSON frame containing citations and metadata.

    Args:
        graph: An initialised ``WorkspaceRAGGraph`` instance.
        body: The validated query request from the HTTP layer.
        request: The raw FastAPI ``Request`` used to access ``app.state``.

    Yields:
        SSE-formatted string frames.

    Raises:
        AuthorizationError: Re-raised if the agent denies access.
        WorkspaceRAGError: Re-raised on agent-level failures.
    """
    # ------------------------------------------------------------------
    # Cache lookup — short-circuit the pipeline on a hit.
    # ------------------------------------------------------------------
    cache_manager = getattr(request.app.state, "cache_manager", None)

    if cache_manager:
        cached = await cache_manager.lookup(body.question, str(body.user_email))
        if cached:
            cached_answer: str = cached["answer"]
            words = cached_answer.split(" ")
            for word in words:
                yield _sse_event(word)
                await asyncio.sleep(0.005)

            yield _sse_event("[DONE]")

            meta = {
                "citations": cached.get("citations", []),
                "session_id": body.session_id or str(uuid.uuid4()),
                "query_id": str(uuid.uuid4()),
                "is_relevant": True,
                "is_faithful": True,
                "latency_ms": cached.get("latency_ms", 0),
                "cache_hit": cached.get("cache_type", "exact"),
                "tokens_saved": cached.get("tokens_saved", 0),
            }
            yield _sse_event(json.dumps(meta))
            return  # early exit — no pipeline invocation needed

    # ------------------------------------------------------------------
    # Full RAG pipeline — timed execution.
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    final_state = await graph.run(
        question=body.question,
        user_email=str(body.user_email),
        session_id=body.session_id or str(uuid.uuid4()),
    )
    elapsed_ms = round((time.monotonic() - t0) * 1000)

    answer: str = final_state.get("answer") or ""

    # Stream tokens smoothly for a responsive UI.
    words = answer.split(" ")
    for word in words:
        yield _sse_event(word)
        await asyncio.sleep(0.01)

    # Always emit [DONE] so clients know streaming has ended.
    yield _sse_event("[DONE]")

    raw_citations = final_state.get("citations") or []
    citations = []
    for c in raw_citations:
        if hasattr(c, "model_dump"):
            citations.append(c.model_dump())
        elif isinstance(c, dict):
            citations.append(c)

    # ------------------------------------------------------------------
    # Store result in cache — fire-and-forget, non-blocking.
    # ------------------------------------------------------------------
    if cache_manager and answer:
        asyncio.create_task(
            cache_manager.store(
                question=body.question,
                user_email=str(body.user_email),
                answer=answer,
                citations=citations,
                tokens_used=len(answer.split()) * 2,  # rough estimate
            )
        )

    meta = {
        "citations": citations,
        "session_id": body.session_id or str(uuid.uuid4()),
        "query_id": str(uuid.uuid4()),
        "is_relevant": final_state.get("is_relevant", True),
        "is_faithful": final_state.get("is_faithful", True),
        "latency_ms": elapsed_ms,
        "cache_hit": None,
        "tokens_saved": 0,
    }
    yield _sse_event(json.dumps(meta))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/",
    summary="Submit a RAG query",
    description=(
        "Accepts a natural-language question and streams an answer via "
        "Server-Sent Events (SSE).  The final SSE frame contains a JSON "
        "object with citations and metadata."
    ),
    responses={
        200: {"description": "Streaming SSE response with answer tokens"},
        403: {"description": "User is not authorised to access the requested documents"},
        500: {"description": "Internal agent or retrieval error"},
    },
)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
) -> StreamingResponse:
    """Stream a RAG-generated answer for the submitted question.

    Args:
        body: Validated ``QueryRequest`` containing the question, user e-mail,
            and optional session identifier.
        request: The raw FastAPI ``Request`` used to access ``app.state``
            (specifically ``app.state.rag_graph`` and the optional
            ``app.state.cache_manager``).

    Returns:
        A ``StreamingResponse`` using ``text/event-stream`` content type.

    Raises:
        JSONResponse(403): When the user lacks permission to view relevant docs.
        JSONResponse(500): On any unhandled ``WorkspaceRAGError``.
    """
    log = logger.bind(
        user_email=body.user_email,
        session_id=body.session_id,
        question_len=len(body.question),
    )
    log.info("query_request_received")

    graph: WorkspaceRAGGraph = request.app.state.rag_graph

    async def _safe_stream() -> AsyncIterator[str]:
        try:
            async for frame in _stream_rag_response(graph, body, request):
                yield frame
        except AuthorizationError as exc:
            log.warning("authorization_error", detail=str(exc))
            error_payload = json.dumps({"error": "Forbidden", "details": str(exc)})
            yield _sse_event(f"ERROR:{error_payload}")
        except WorkspaceRAGError as exc:
            log.error("rag_error", detail=str(exc))
            error_payload = json.dumps({"error": exc.message, "details": exc.details})
            yield _sse_event(f"ERROR:{error_payload}")

    return StreamingResponse(
        _safe_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/health",
    summary="Query service health check",
    description="Returns the operational status of the query service.",
    responses={200: {"description": "Service is healthy"}},
)
async def health_check() -> dict[str, str]:
    """Return the health status and application version.

    Returns:
        A dictionary with keys ``status`` and ``version``.
    """
    settings = get_settings()
    return {"status": "ok", "version": settings.APP_VERSION}
