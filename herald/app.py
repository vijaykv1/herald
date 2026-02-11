"""Application entry point for the herald package."""

import uuid
from agents import Agent, Runner, trace, gen_trace_id, SQLiteSession

from herald.context_manager.icontext import ContextInterface

# import herald.response_handles as resp_handles


class HeraldApp:
    """Herald application."""

    def __init__(self, prompt: ContextInterface):
        """Initialize the Herald application.

        :param ContextInterface prompt: The context interface for the application,
        which provides the necessary context for the agent to operate.
        """
        self.uuid = uuid.uuid4()
        # ":memory:" for in-memory database, "herald_traces.db" for file-based persistence
        self.session = SQLiteSession(session_id=str(self.uuid), db_path="herald_traces.db")
        self.prompt = prompt

    def herald_agent(self):
        """Heralder Agent for CV conversations"""
        return Agent(
            name="heralder",
            instructions=self.prompt.get_system_instructions(),
            model="gpt-5-nano",
            # output_type=resp_handles.CVResponse
        )

    async def run(self, message: str, history: list):
        """
        Run query on the CV provided

        .. note::
            The message is the user query for the CV and the history is the conversation history for the
            current session. The history is used to provide context to the agent for better responses.
            But then the context provided in the system prompt is already quite comprehensive, so the history might
            not be that useful in this case. We can experiment with it in future iterations.

        :param message: Message provided by the user
        :type message: str
        """
        print(f"Session ID: {self.session.session_id}")
        # create a trace path for current LLM run
        trace_id = gen_trace_id()
        with trace("Herald Trace", trace_id=trace_id):
            yield f"Traces @ https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print("Asking Herald!")
            result = await Runner.run(
                self.herald_agent(), message, session=self.session  # for conversation history and traceability
            )
            yield result.final_output


# import asyncio

# async def current_app():
#     h_app = HeraldApp()
#     async for resp in h_app.run(query="Did Varun work for BMW ?"):
#         print(resp)

# asyncio.run(current_app())
