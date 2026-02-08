"""Herald Main Application."""
import os
import dotenv
import gradio as gr

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from herald.app import HeraldApp

dotenv.load_dotenv()

async def run_herald(query: str):
    """
    Main Herald entry point
    
    :param query: User Query for the herald system
    :type query: str
    """
    async for chunk in HeraldApp().run(query=query):
        yield chunk


def ui_debug():
    """For activating Gradio based UI."""

    with gr.Blocks(theme=gr.themes.Glass(primary_hue="sky")) as ui: 

        # Announce the herald Application!
        gr.Markdown("# The Herald!")

        # Add query textbox 
        query_textbox = gr.Textbox(label="What would like to know about me ?")
        run_button = gr.Button("Run", variant="primary")

        # Place to get the answer for current query
        knowledge = gr.Markdown(label="Knowledge")

        # map everything! 
        run_button.click(fn=run_herald, inputs=query_textbox, outputs=knowledge)
        query_textbox.submit(fn=run_herald, inputs=query_textbox, outputs=knowledge)

    # launch gradio
    ui.launch(inbrowser=True)


async def terminal_ui():
    """For terminal based console."""

    console = Console()
    
    console.print(Panel.fit("🎺 The Herald", style="bold cyan"))
    console.print("Ask questions about the CV (type 'exit' to quit)\n", style="dim")
    
    while True:
        query = Prompt.ask("[bold green]Query[/bold green]")
        
        if query.lower() in ['exit', 'quit', 'q']:
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

        if browser_based == "yes":
            # ui_debug()
            gr.ChatInterface(HeraldApp().run).launch()

        else:  # Run on terminal
            asyncio.run(terminal_ui())
    
    finally:  # clean up workspace by removing the traces db after the run
        print("Cleaning up traces database...")
        for fname in ["herald_traces.db", "herald_traces.db-shm", "herald_traces.db-wal"]:
            if os.path.exists(fname):
                print(f"Cleaning up {fname}...")
                os.remove(fname)
