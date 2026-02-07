"""Application entry point for the herald package."""

from agents import Agent, Runner, trace, gen_trace_id

from herald.context import HeraldPrompter
# import herald.response_handles as resp_handles


class HeraldApp:
    """Herald application."""

    def herald_agent(self):
        """Heralder Agent for CV conversations"""
        return Agent(
            name="heralder",
            instructions=HeraldPrompter.get_basic_system_instructions(),
            model="gpt-5-nano",
            # output_type=resp_handles.CVResponse
        )

    async def run(self, query: str):
        """
        Run query on the CV provided
        
        :param query: Query provided my the user
        :type query: str
        """
        # create a trace path for current LLM run
        trace_id = gen_trace_id()
        with trace("Herald Trace", trace_id=trace_id):
            yield f"Traces @ https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print("Starting Herald!")
            result = await Runner.run(
                self.herald_agent(),
                f"Query: {query}"
            )
            yield result.final_output

        
# import asyncio

# async def current_app():
#     h_app = HeraldApp()
#     async for resp in h_app.run(query="Did Varun work for BMW ?"):
#         print(resp)

# asyncio.run(current_app())
    
