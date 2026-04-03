"""Application entry point for the herald package."""

import logging
import os
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from agents import Agent, Runner, SQLiteSession
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from herald.context_manager.icontext import ContextInterface

_GROQ_MODEL = "openai/gpt-oss-120b"
_FALLBACK_MODEL = "gpt-4o-mini"

logger = logging.getLogger(__name__)


def _build_groq_model() -> OpenAIChatCompletionsModel:
    """Build a Groq-backed chat completions model, bypassing the agents SDK prefix router."""
    client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    return OpenAIChatCompletionsModel(model=_GROQ_MODEL, openai_client=client)


class HeraldApp:
    """Herald application."""

    def __init__(self, prompt: ContextInterface):
        """Initialize the Herald application.

        :param ContextInterface prompt: The context interface for the application,
        which provides the necessary context for the agent to operate.
        """
        self.prompt = prompt

    def _base_agent_options(self) -> dict:
        """Build shared agent options (name, instructions, tools)."""
        options = {
            "name": "heralder",
            "instructions": self.prompt.get_system_instructions(),
        }
        if self.prompt.type == "rag_based":
            options["tools"] = self.prompt.context_store.create_tools()
        return options

    def herald_agent(self):
        """Primary heralder agent backed by Groq."""
        return Agent(**self._base_agent_options(), model=_build_groq_model())

    def _fallback_agent(self):
        """Fallback heralder agent backed by OpenAI (gpt-5-nano)."""
        return Agent(**self._base_agent_options(), model=_FALLBACK_MODEL)

    async def run(self, message: str, session: SQLiteSession):
        """
        Run query on the CV provided, maintaining conversation history via the given session.
        Falls back to OpenAI gpt-5-nano if the Groq call fails.

        :param message: Message provided by the user
        :param session: Per-user SQLiteSession that stores conversation history
        """
        print(f"Session ID: {session.session_id}")
        try:
            result = await Runner.run(self.herald_agent(), message, session=session)
        except (APIConnectionError, RateLimitError, APIStatusError) as exc:
            logger.warning("Groq call failed (%s) — falling back to OpenAI", exc)
            result = await Runner.run(self._fallback_agent(), message, session=session)
        yield result.final_output
