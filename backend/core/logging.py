"""Structured logging configuration for WorkspaceRAG.

This module configures ``structlog`` for the entire application.  It bridges
the standard-library ``logging`` module so that any third-party library that
uses ``logging.getLogger(...)`` will have its output routed through structlog's
processing chain.

Usage::

    from backend.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("document_indexed", doc_id="abc123", chunks=12)

The output format depends on the runtime environment:

* **Production** (``DEBUG=False``): JSON lines suitable for log aggregators
  such as Cloud Logging, Datadog, or Loki.
* **Development** (``DEBUG=True``): Human-readable coloured console output via
  :class:`structlog.dev.ConsoleRenderer`.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def _add_app_context(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Processor that injects a static ``app`` key into every log record.

    Args:
        logger: The wrapped logger instance (unused but required by the
            structlog processor signature).
        method_name: The logging method name, e.g. ``"info"`` (unused).
        event_dict: The mutable event dictionary being processed.

    Returns:
        The event dictionary augmented with an ``"app"`` key.
    """
    event_dict.setdefault("app", "workspace-rag")
    return event_dict


def configure_logging(debug: bool = False, log_level: str = "INFO") -> None:
    """Bootstrap structlog and stdlib logging for the application.

    This function is **idempotent** — calling it multiple times is safe.  It
    should be invoked once during application startup (e.g. inside the FastAPI
    ``lifespan`` context or at the top of a CLI entry-point).

    Args:
        debug: When ``True``, use the human-readable
            :class:`structlog.dev.ConsoleRenderer`.  When ``False`` (default),
            emit newline-delimited JSON records.
        log_level: Minimum log level to emit, e.g. ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``.  Case-insensitive.

    Example:
        >>> configure_logging(debug=True, log_level="DEBUG")
    """
    numeric_level: int = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_app_context,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if debug:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Avoid adding duplicate handlers on repeated calls.
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Silence overly chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "google.auth", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog ``BoundLogger`` bound to the given name.

    The returned logger is a drop-in replacement for ``logging.getLogger`` and
    supports both positional and keyword context arguments::

        logger = get_logger(__name__)
        logger.info("user_query_received", user="alice@example.com", latency_ms=42)

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`structlog.stdlib.BoundLogger` instance.

    Example:
        >>> logger = get_logger("backend.ingestion")
        >>> logger.debug("chunk_created", chunk_id="c1")
    """
    return structlog.get_logger(name)
