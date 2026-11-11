"""FastAPI/Starlette middleware for the WorkspaceRAG backend.

Provides two middleware components:

1. :class:`RequestLoggingMiddleware` — structured request/response logging
   with an injected ``X-Request-ID`` header.
2. :func:`add_error_handlers` — registers global exception handlers that
   translate :class:`~backend.core.exceptions.WorkspaceRAGError` and
   bare ``Exception`` instances into well-formed JSON error responses.
"""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.core.exceptions import WorkspaceRAGError
from backend.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Request Logging Middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request and response with structured metadata.

    For each inbound request the middleware:

    * Reads or generates a ``X-Request-ID`` UUID.
    * Binds the request-id to the structlog context for the duration of the
      request so that all log statements within the same call share it.
    * Times the request and logs ``method``, ``path``, ``status_code``, and
      ``duration_ms`` on completion.
    * Injects the ``X-Request-ID`` header into the outgoing response.

    Args:
        app: The inner ASGI application.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process a single request/response cycle.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that invokes the next middleware or route
                handler and returns the response.

        Returns:
            The HTTP response with the ``X-Request-ID`` header attached.
        """
        request_id: str = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        bound_logger = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        bound_logger.info("request_started")

        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            bound_logger.exception(
                "request_unhandled_exception", duration_ms=duration_ms
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        bound_logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# 2. Global Error Handlers
# ---------------------------------------------------------------------------


async def _workspace_rag_error_handler(
    request: Request, exc: WorkspaceRAGError
) -> JSONResponse:
    """Convert a :class:`WorkspaceRAGError` into a JSON response.

    Args:
        request: The request that triggered the exception.
        exc: The domain error raised during request processing.

    Returns:
        A :class:`JSONResponse` with the error message and details, using
        the status code carried by the exception (defaulting to 500).
    """
    status_code: int = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    logger.warning(
        "workspace_rag_error",
        message=exc.message,
        details=exc.details,
        status_code=status_code,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status_code,
        content={"error": exc.message, "details": exc.details},
    )


async def _unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Convert any unhandled exception into a generic HTTP 500 response.

    The full traceback is logged internally but is **never** exposed to
    the client.

    Args:
        request: The request that triggered the exception.
        exc: The unhandled exception.

    Returns:
        A :class:`JSONResponse` with a generic error message and HTTP 500
        status code.
    """
    tb = traceback.format_exc()
    logger.error(
        "unhandled_exception",
        exc_type=type(exc).__name__,
        exc_message=str(exc),
        path=request.url.path,
        traceback=tb,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected internal error occurred.",
            "details": "Please contact support if this persists.",
        },
    )


def add_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI application.

    Must be called **after** the :class:`FastAPI` instance is created but
    **before** the application starts serving traffic.

    Args:
        app: The FastAPI application instance to register handlers on.
    """
    app.add_exception_handler(WorkspaceRAGError, _workspace_rag_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)
