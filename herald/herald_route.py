
"""Herald API routes."""

import logging
from pydantic import BaseModel, Field
from fastapi import APIRouter

from herald.app import HeraldApp
from herald.context_manager.rag_based import HeraldRAGContextManager

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Chat request model for the API."""
    message: str = Field(description="Chat message")
    session_id: str = Field(description="Session Identifier")


herald_router = APIRouter()


@herald_router.get("/")
def app_root() -> dict:
    """Root endpoint for the API."""
    return {
        "version": "default"
    }


@herald_router.post("/ai/ask")
async def ask_api(body: ChatRequest) -> dict:
    """API endpoint to handle chat requests."""
    logger.info("Processing chat request: %s", body.message)
    prompt = HeraldRAGContextManager()
    async for chunk in HeraldApp(prompt=prompt).run(message=body.message, history=[]):
        return {"response": chunk}
