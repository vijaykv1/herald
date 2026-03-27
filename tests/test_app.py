"""Tests for the main Herald application."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from herald.app import HeraldApp
from herald.context_manager.prompt_based import HeraldBasicPrompter
from herald.context_manager.rag_based import HeraldRAGContextManager


class TestHeraldApp:
    """Test cases for HeraldApp."""

    def test_init_basic_prompt(self):
        """Test initialization with basic prompt stores the prompt."""
        mock_prompt = MagicMock(spec=HeraldBasicPrompter)
        mock_prompt.type = "basic_prompt"

        app = HeraldApp(prompt=mock_prompt)

        assert app.prompt == mock_prompt

    def test_init_rag_prompt(self):
        """Test initialization with RAG prompt stores the prompt."""
        mock_prompt = MagicMock(spec=HeraldRAGContextManager)
        mock_prompt.type = "rag_based"
        mock_prompt.context_store = MagicMock()

        app = HeraldApp(prompt=mock_prompt)

        assert app.prompt == mock_prompt

    @patch('herald.app.Agent')
    def test_herald_agent_basic(self, mock_agent):
        """Test herald_agent creation with basic prompt."""
        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        mock_prompt.get_system_instructions.return_value = "Test instructions"

        app = HeraldApp(prompt=mock_prompt)
        app.herald_agent()

        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs['name'] == "heralder"
        assert call_kwargs['instructions'] == "Test instructions"
        assert call_kwargs['model'] == "gemini-2.0-flash"
        assert 'tools' not in call_kwargs

    @patch('herald.app.Agent')
    def test_herald_agent_rag(self, mock_agent):
        """Test herald_agent creation with RAG prompt includes tools."""
        mock_tool = MagicMock()
        mock_context_store = MagicMock()
        mock_context_store.create_tool.return_value = mock_tool

        mock_prompt = MagicMock()
        mock_prompt.type = "rag_based"
        mock_prompt.context_store = mock_context_store
        mock_prompt.get_system_instructions.return_value = "RAG instructions"

        app = HeraldApp(prompt=mock_prompt)
        app.herald_agent()

        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        assert 'tools' in call_kwargs
        assert call_kwargs['tools'] == [mock_tool]

    @patch('herald.app.Runner')
    @patch('herald.app.Agent')
    @patch('herald.app.trace')
    @patch('herald.app.gen_trace_id')
    @pytest.mark.asyncio
    async def test_run_success(self, mock_gen_trace_id, mock_trace, mock_agent, mock_runner):
        """Test successful run of query with a session."""
        mock_gen_trace_id.return_value = "test_trace_id"
        mock_trace.return_value.__enter__ = MagicMock()
        mock_trace.return_value.__exit__ = MagicMock()

        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        mock_prompt.get_system_instructions.return_value = "Instructions"

        mock_result = MagicMock()
        mock_result.final_output = "Test response"
        mock_runner.run = AsyncMock(return_value=mock_result)

        mock_session = MagicMock()
        mock_session.session_id = "test-session-123"

        app = HeraldApp(prompt=mock_prompt)
        results = []
        async for chunk in app.run(message="Test query", session=mock_session):
            results.append(chunk)

        assert len(results) == 1
        assert results[0] == "Test response"
        mock_runner.run.assert_called_once()
        # Verify the session was passed through to Runner
        _, run_kwargs = mock_runner.run.call_args
        assert run_kwargs.get('session') == mock_session

    @patch('herald.app.Runner')
    @patch('herald.app.Agent')
    @patch('herald.app.trace')
    @patch('herald.app.gen_trace_id')
    @pytest.mark.asyncio
    async def test_run_uses_session_for_history(self, mock_gen_trace_id, mock_trace, mock_agent, mock_runner):
        """Test that run passes the session to Runner so history is maintained."""
        mock_gen_trace_id.return_value = "test_trace_id"
        mock_trace.return_value.__enter__ = MagicMock()
        mock_trace.return_value.__exit__ = MagicMock()

        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        mock_prompt.get_system_instructions.return_value = "Instructions"

        mock_result = MagicMock()
        mock_result.final_output = "Follow-up response"
        mock_runner.run = AsyncMock(return_value=mock_result)

        mock_session = MagicMock()
        mock_session.session_id = "follow-up-session"

        app = HeraldApp(prompt=mock_prompt)
        results = []
        async for chunk in app.run(message="Follow-up question", session=mock_session):
            results.append(chunk)

        assert len(results) == 1
        # The same session object must be passed — Runner handles history internally
        _, run_kwargs = mock_runner.run.call_args
        assert run_kwargs.get('session') is mock_session
