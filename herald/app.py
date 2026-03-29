"""Application entry point for the herald package."""

import os
from openai import AsyncOpenAI
from agents import Agent, Runner, SQLiteSession
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from herald.context_manager.icontext import ContextInterface

_GROQ_MODEL = "openai/gpt-oss-120b"


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

    def herald_agent(self):
        """Heralder Agent for CV conversations"""

        agent_options = {
            "name": "heralder",
            "instructions": self.prompt.get_system_instructions(),
            "model": _build_groq_model(),
        }

        # Add vector store retriever to the agent options if the prompt type is RAG based
        if self.prompt.type == "rag_based":
            agent_options["tools"] = self.prompt.context_store.create_tools()

        return Agent(**agent_options)

    async def run(self, message: str, session: SQLiteSession):
        """
        Run query on the CV provided, maintaining conversation history via the given session.

        :param message: Message provided by the user
        :param session: Per-user SQLiteSession that stores conversation history
        """
        print(f"Session ID: {session.session_id}")
        result = await Runner.run(
            self.herald_agent(), message, session=session
        )
        yield result.final_output
