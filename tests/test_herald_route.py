"""Tests for Herald API routes."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from herald.herald_route import (
    herald_router,
    get_herald_prompt,
    get_herald_app,
    get_session_store,
    _get_or_create_session,
    SESSION_TTL_SECONDS,
)
from herald.usage_tracker import DAILY_MESSAGE_LIMIT


class TestDependencies:
    """Tests for FastAPI dependency functions."""

    def test_get_herald_prompt(self):
        mock_request = MagicMock()
        mock_request.app.state.herald_prompt = "mock_prompt"
        assert get_herald_prompt(mock_request) == "mock_prompt"

    def test_get_herald_app(self):
        mock_request = MagicMock()
        mock_request.app.state.herald_app = "mock_app"
        assert get_herald_app(mock_request) == "mock_app"

    def test_get_session_store(self):
        mock_request = MagicMock()
        mock_request.app.state.session_store = {"key": "value"}
        assert get_session_store(mock_request) == {"key": "value"}


class TestGetOrCreateSession:
    """Tests for session management helper."""

    @patch("herald.herald_route.SQLiteSession")
    def test_creates_new_session(self, mock_sqlite_session):
        mock_session = MagicMock()
        mock_sqlite_session.return_value = mock_session

        session_store = {}
        result = _get_or_create_session(session_store, "session_123")

        assert result == mock_session
        assert "session_123" in session_store

    @patch("herald.herald_route.time")
    def test_refreshes_existing_session_timestamp(self, mock_time):
        mock_time.monotonic.return_value = 2000.0
        mock_session = MagicMock()
        session_store = {"existing": (mock_session, 1000.0)}

        result = _get_or_create_session(session_store, "existing")

        assert result == mock_session
        assert session_store["existing"][1] == 2000.0

    @patch("herald.herald_route.SQLiteSession")
    @patch("herald.herald_route.time")
    def test_evicts_stale_sessions(self, mock_time, mock_sqlite_session):
        now = 10000.0
        mock_time.monotonic.return_value = now
        mock_sqlite_session.return_value = MagicMock()

        stale_session = MagicMock()
        session_store = {
            "stale": (stale_session, now - SESSION_TTL_SECONDS - 1),
        }

        _get_or_create_session(session_store, "new_session")

        assert "stale" not in session_store
        assert "new_session" in session_store


class TestRoutes:
    """Tests for API route endpoints."""

    def _make_app(self, herald_app=None, session_store=None, usage_tracker=None):
        app = FastAPI()
        app.include_router(herald_router)
        app.state.herald_prompt = MagicMock()
        app.state.herald_app = herald_app or MagicMock()
        app.state.session_store = session_store if session_store is not None else {}
        if usage_tracker is None:
            usage_tracker = MagicMock()
            usage_tracker.check_quota.return_value = (0, DAILY_MESSAGE_LIMIT)
            usage_tracker.increment.return_value = 1
        app.state.usage_tracker = usage_tracker
        return app

    def test_root_returns_version(self):
        client = TestClient(self._make_app())
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"version": "default"}

    @patch("herald.herald_route.SQLiteSession")
    def test_ask_api_returns_first_chunk(self, mock_sqlite_session):
        mock_sqlite_session.return_value = MagicMock()

        async def mock_run(message, session):
            yield "Test response"

        mock_herald_app = MagicMock()
        mock_herald_app.run = mock_run

        client = TestClient(self._make_app(herald_app=mock_herald_app))
        response = client.post(
            "/ai/ask",
            json={"message": "Hello", "session_id": "sess_1"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "response": "Test response",
            "usage": {"used": 1, "limit": DAILY_MESSAGE_LIMIT, "remaining": DAILY_MESSAGE_LIMIT - 1},
        }
