"""Herald Main Application."""

import os
import dotenv
import gradio as gr

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from herald.app import HeraldApp
from herald.context_manager.prompt_based import HeraldBasicPrompter

dotenv.load_dotenv()


async def terminal_ui():
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

        async for chunk in HeraldApp().run(message=query, history=[]):
            console.print(chunk)

        console.print()


if __name__ == "__main__":

    try:
        # Get user selection for Browser based UI or terminal UI
        browser_based = os.getenv("WITH_BROWSER", "no")
        prompt_option = os.getenv("PROMPT_OPTION", "basic")

        if prompt_option == "basic":
            prompt = HeraldBasicPrompter()
        else:
            raise ValueError(f"Invalid PROMPT_OPTION: {prompt_option}")

        if browser_based == "yes":
            # ui_debug()
            gr.ChatInterface(HeraldApp(prompt=prompt.run)).launch()

        else:  # Run on terminal
            asyncio.run(terminal_ui())

    finally:  # clean up workspace by removing the traces db after the run
        print("Cleaning up traces database...")
        for fname in ["herald_traces.db", "herald_traces.db-shm", "herald_traces.db-wal"]:
            if os.path.exists(fname):
                print(f"Cleaning up {fname}...")
                os.remove(fname)
