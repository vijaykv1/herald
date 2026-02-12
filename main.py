"""Herald Main Application."""

import os
import asyncio
import dotenv
import gradio as gr

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from herald.app import HeraldApp
from herald.context_manager.prompt_based import HeraldBasicPrompter
from herald.context_manager.rag_based import HeraldRAGContextManager

dotenv.load_dotenv()


async def terminal_ui(prompt):
    """For terminal based console."""

    console = Console()

    console.print(Panel.fit("🎺 The Herald", style="bold cyan"))
    console.print("Ask questions about the CV (type 'exit' to quit)\n", style="dim")

    while True:
        query = Prompt.ask("[bold green]Query[/bold green]")

        if query.lower() in ["exit", "quit", "q"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if not query:
            continue

        console.print("\n[bold blue]Answer:[/bold blue]")

        async for chunk in HeraldApp(prompt=prompt).run(message=query, history=[]):
            console.print(chunk)

        console.print()


if __name__ == "__main__":

    try:
        # Get user selection for Browser based UI or terminal UI
        browser_based = os.getenv("WITH_BROWSER", "no")
        prompt_option = os.getenv("PROMPT_OPTION", "basic")

        if prompt_option == "basic":
            prompt_type = HeraldBasicPrompter()
        elif prompt_option == "rag":  # RAG based
            prompt_type = HeraldRAGContextManager()
        else:
            raise ValueError(f"Unsupported PROMPT_OPTION: {prompt_option}. Supported options are 'basic' and 'rag'.")

        if browser_based == "yes":
            # ui_debug()
            gr.ChatInterface(HeraldApp(prompt=prompt_type).run).launch()

        else:  # Run on terminal
            asyncio.run(terminal_ui(prompt=prompt_type))

    finally:  # clean up workspace by removing the traces db after the run
        print("Cleaning up traces database...")
        for fname in ["herald_traces.db", "herald_traces.db-shm", "herald_traces.db-wal"]:
            if os.path.exists(fname):
                print(f"Cleaning up {fname}...")
                os.remove(fname)
