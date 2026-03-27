"""Application entry point for the herald package."""

from agents import Agent, Runner, trace, gen_trace_id, SQLiteSession

from herald.context_manager.icontext import ContextInterface


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
            "model": "gemini-2.0-flash",
        }

        # Add vector store retriever to the agent options if the prompt type is RAG based
        if self.prompt.type == "rag_based":
            agent_options["tools"] = [self.prompt.context_store.create_tool()]

        return Agent(**agent_options)

    async def run(self, message: str, session: SQLiteSession):
        """
        Run query on the CV provided, maintaining conversation history via the given session.

        :param message: Message provided by the user
        :param session: Per-user SQLiteSession that stores conversation history
        """
        print(f"Session ID: {session.session_id}")
        trace_id = gen_trace_id()
        with trace("Herald Trace", trace_id=trace_id):
            print("Asking Herald!")
            result = await Runner.run(
                self.herald_agent(), message, session=session
            )
            yield result.final_output


# import asyncio

# async def current_app():
#     h_app = HeraldApp()
#     async for resp in h_app.run(query="Did Varun work for BMW ?"):
#         print(resp)

# asyncio.run(current_app())
