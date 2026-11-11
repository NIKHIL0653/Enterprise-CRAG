"""Public re-exports for the ``backend.api.v1`` package.

Imports and exposes both the query and sync routers so that the application
factory can register them with a single import.

Example::

    from backend.api.v1 import query_router, sync_router

    app.include_router(query_router, prefix="/api/v1")
    app.include_router(sync_router, prefix="/api/v1")
"""

from __future__ import annotations

from backend.api.v1.query import router as query_router
from backend.api.v1.sync import router as sync_router

__all__: list[str] = ["query_router", "sync_router"]
