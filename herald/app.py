"""Application entry point for the herald package."""

import os
import uuid
from agents import Agent, Runner, trace, gen_trace_id, SQLiteSession

from herald.context_manager.icontext import ContextInterface
from herald.context_manager.rag import CVVectorStore

from herald.cv_parser.linkedin import LinkedInCVParser


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

        # initialize the vector store if the prompt type is RAG based
        self.vector_store = None
        if self.prompt.type == "rag_based":
            cv_type = os.getenv("CV_TYPE", "linkedin")

            if cv_type == "linkedin":
                cv_parser = LinkedInCVParser(cv=self.prompt.cv_md_content)
            else:
                raise ValueError(f"Unsupported CV type: {cv_type}")

            # perform parse to get the chunked data ready for vector store creation
            cv_chunks = cv_parser.parse()
            self.vector_store = CVVectorStore(cv_chunks=cv_chunks)  # type: ignore

    def herald_agent(self):
        """Heralder Agent for CV conversations"""

        agent_options = {
            "name": "heralder",
            "instructions": self.prompt.get_system_instructions(),
            "model": "gpt-5-nano",
            "tools": [self.vector_store.retrieve_relevant_chunks],
        }

        # Add vector store retriever to the agent options if the prompt type is RAG based
        if self.prompt.type == "rag_based":
            agent_options["tools"] = [self.vector_store.retrieve_relevant_chunks]

        return Agent(**agent_options)

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
