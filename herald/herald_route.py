
"""Herald API routes."""

import logging
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request, Depends

from herald.app import HeraldApp
from herald.context_manager.icontext import ContextInterface

logger = logging.getLogger(__name__)


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

herald_router = APIRouter()


@herald_router.get("/")
def app_root() -> dict:
    """Root endpoint for the API."""
    return {
        "version": "default"
    }


@herald_router.post("/ai/ask")
async def ask_api(
    chat_request: ChatRequest,
    herald_app: HeraldApp = Depends(get_herald_app)
) -> dict:
    """API endpoint to handle chat requests."""
    logger.info("Processing chat request: %s", chat_request.message)

    async for chunk in herald_app.run(message=chat_request.message, history=[]):
        return {"response": chunk}
