
"""Herald API routes."""

import logging
import time
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request, Depends, Header
from agents import SQLiteSession

from herald.app import HeraldApp
from herald.context_manager.icontext import ContextInterface
from herald.usage_tracker import UsageTracker, DAILY_MESSAGE_LIMIT

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes
HERALD_DB_PATH = "herald_traces.db"


class ChatRequest(BaseModel):
    """Chat request model for the API."""
    message: str = Field(description="Chat message")
    session_id: str = Field(description="Session Identifier")


def get_herald_prompt(request: Request) -> ContextInterface:
    """Dependency to get the Herald prompt from the application state."""
    return request.app.state.herald_prompt


def get_herald_app(request: Request) -> HeraldApp:
    """Dependency to get the Herald application instance from the application state."""
    return request.app.state.herald_app


def get_session_store(request: Request) -> dict:
    """Dependency to get the session store from application state."""
    return request.app.state.session_store


def get_usage_tracker(request: Request) -> UsageTracker:
    """Dependency to get the UsageTracker from application state."""
    return request.app.state.usage_tracker


herald_router = APIRouter()


def _get_or_create_session(session_store: dict, session_id: str) -> SQLiteSession:
    """Return an existing SQLiteSession for session_id or create a new one.

    Also evicts sessions that have been idle longer than SESSION_TTL_SECONDS.
    """
    now = time.monotonic()

    # Lazy TTL eviction
    stale = [sid for sid, (_, last_active) in session_store.items()
             if now - last_active > SESSION_TTL_SECONDS]
    for sid in stale:
        logger.info("Evicting idle session: %s", sid)
        del session_store[sid]

    if session_id not in session_store:
        logger.info("Creating new session: %s", session_id)
        session_store[session_id] = (
            SQLiteSession(session_id=session_id, db_path=HERALD_DB_PATH),
            now,
        )
    else:
        # Refresh last-active timestamp
        sql_session, _ = session_store[session_id]
        session_store[session_id] = (sql_session, now)

    return session_store[session_id][0]


@herald_router.get("/")
def app_root() -> dict:
    """Root endpoint for the API."""
    return {
        "version": "default"
    }


@herald_router.get("/ai/usage")
def get_usage(
    usage_tracker: UsageTracker = Depends(get_usage_tracker),
    x_user_id: str = Header(default="anonymous"),
) -> dict:
    """Return today's usage stats for the requesting user."""
    used = usage_tracker.get_count(x_user_id)
    return {
        "used": used,
        "limit": DAILY_MESSAGE_LIMIT,
        "remaining": max(0, DAILY_MESSAGE_LIMIT - used),
    }


@herald_router.post("/ai/ask")
async def ask_api(
    chat_request: ChatRequest,
    herald_app: HeraldApp = Depends(get_herald_app),
    session_store: dict = Depends(get_session_store),
    usage_tracker: UsageTracker = Depends(get_usage_tracker),
    x_user_id: str = Header(default="anonymous"),
) -> dict:
    """API endpoint to handle chat requests."""
    # Enforce daily quota before spending any tokens
    used, remaining = usage_tracker.check_quota(x_user_id)

    logger.info(
        "Processing chat request [user=%s, session=%s, usage=%d/%d]: %s",
        x_user_id, chat_request.session_id, used, DAILY_MESSAGE_LIMIT, chat_request.message,
    )

    session = _get_or_create_session(session_store, chat_request.session_id)

    async for chunk in herald_app.run(message=chat_request.message, session=session):
        new_count = usage_tracker.increment(x_user_id)
        return {
            "response": chunk,
            "usage": {
                "used": new_count,
                "limit": DAILY_MESSAGE_LIMIT,
                "remaining": max(0, DAILY_MESSAGE_LIMIT - new_count),
            },
        }
