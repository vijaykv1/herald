"""Herald Main Application."""

import os
import asyncio
from contextlib import asynccontextmanager

import dotenv
import gradio as gr
from agents import SQLiteSession

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from herald.app import HeraldApp
from herald.context_manager.prompt_based import HeraldBasicPrompter
from herald.context_manager.rag_based import HeraldRAGContextManager
from herald.herald_route import herald_router, HERALD_DB_PATH
from herald.usage_tracker import UsageTracker

dotenv.load_dotenv()


def cleanup_traces_db():
    """Clean up traces database after the run."""
    print("Cleaning up traces database...")
    for fname in ["herald_traces.db", "herald_traces.db-shm", "herald_traces.db-wal"]:
        if os.path.exists(fname):
            print(f"Cleaning up {fname}...")
            os.remove(fname)


async def terminal_ui(prompt):
    """For terminal based console."""

    console = Console()
    session = SQLiteSession(session_id="terminal", db_path=HERALD_DB_PATH)
    app_instance = HeraldApp(prompt=prompt)

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

        async for chunk in app_instance.run(message=query, session=session):
            console.print(chunk)

        console.print()


@asynccontextmanager
async def lifespan_context(app: FastAPI):  # pylint: disable=unused-argument
    """Lifespan context manager for FastAPI application.
    
    :param FastAPI app: FastAPI application instance
    """
    print("Building the application context...")
    print(f"TOKEN_CTRL_ENABLED raw value: {os.getenv('TOKEN_CTRL_ENABLED')!r}")
    print(f"TOKEN_CTRL_ENABLED parsed: {os.getenv('TOKEN_CTRL_ENABLED', 'false').lower() == 'true'}")
    app.state.herald_prompt = HeraldRAGContextManager()  # or use HeraldBasicPrompter()
    app.state.herald_app = HeraldApp(prompt=app.state.herald_prompt)
    app.state.session_store = {}  # session_id → (SQLiteSession, last_active_monotonic)
    app.state.usage_tracker = UsageTracker()  # persistent per-user daily quota tracking
    yield
    cleanup_traces_db()


herald_app = FastAPI(lifespan=lifespan_context)

ALLOWED_ORIGINS = [
    origin for origin in [
        os.getenv("PORTFOLIO_ORIGIN", ""),   # e.g. https://yourportfolio.com
        "http://localhost:3000",             # Next.js local dev
    ] if origin
]

herald_app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

herald_app.include_router(herald_router)

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
        cleanup_traces_db()
