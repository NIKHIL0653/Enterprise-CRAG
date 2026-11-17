"""Unit tests for Google Drive Changes API push notification webhook receiver."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.v1.sync import router


@pytest.fixture
def sync_app() -> tuple[TestClient, MagicMock]:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    mock_sync_manager = MagicMock()
    mock_sync_manager.run_incremental_sync = AsyncMock(
        return_value=({"processed": 2, "failed": 0, "duration_seconds": 0.45}, "new_token_123")
    )
    app.state.sync_manager = mock_sync_manager
    app.state.db_pool = MagicMock()

    client = TestClient(app)
    return client, mock_sync_manager


def test_drive_webhook_missing_headers(sync_app) -> None:
    """Ensure webhook ignores calls without X-Goog headers."""
    client, _ = sync_app
    resp = client.post("/api/v1/sync/webhook")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ignored"
    assert "Missing X-Goog headers" in data["detail"]


def test_drive_webhook_sync_handshake(sync_app) -> None:
    """Validate Google Drive channel subscription verification handshake ('sync' state)."""
    client, mock_manager = sync_app
    headers = {
        "X-Goog-Channel-ID": "channel-abc-123",
        "X-Goog-Resource-State": "sync",
        "X-Goog-Message-Number": "1",
    }
    resp = client.post("/api/v1/sync/webhook", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "subscribed"
    assert data["channel_id"] == "channel-abc-123"
    # Sync state is handshake only; should not trigger re-indexing
    mock_manager.run_incremental_sync.assert_not_called()


def test_drive_webhook_change_triggers_incremental_sync(sync_app) -> None:
    """Validate that 'change' or 'add' notification triggers incremental sync."""
    client, mock_manager = sync_app
    headers = {
        "X-Goog-Channel-ID": "channel-abc-123",
        "X-Goog-Resource-State": "change",
        "X-Goog-Message-Number": "42",
    }
    resp = client.post("/api/v1/sync/webhook", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processed"
    assert data["resource_state"] == "change"
    assert data["stats"]["processed"] == 2
    mock_manager.run_incremental_sync.assert_awaited_once()
