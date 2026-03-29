"""Tests for the main Herald application."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from herald.app import HeraldApp, _GROQ_MODEL
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

    @patch('herald.app._build_groq_model')
    @patch('herald.app.Agent')
    def test_herald_agent_basic(self, mock_agent, mock_build_model):
        """Test herald_agent creation with basic prompt."""
        mock_model = MagicMock(spec=OpenAIChatCompletionsModel)
        mock_model.model = _GROQ_MODEL
        mock_build_model.return_value = mock_model

        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        mock_prompt.get_system_instructions.return_value = "Test instructions"

        app = HeraldApp(prompt=mock_prompt)
        app.herald_agent()

        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs['name'] == "heralder"
        assert call_kwargs['instructions'] == "Test instructions"
        assert call_kwargs['model'] == mock_model
        assert 'tools' not in call_kwargs

    @patch('herald.app._build_groq_model')
    @patch('herald.app.Agent')
    def test_herald_agent_rag(self, mock_agent, mock_build_model):
        """Test herald_agent creation with RAG prompt includes tools."""
        mock_build_model.return_value = MagicMock(spec=OpenAIChatCompletionsModel)

        mock_tools = [MagicMock(), MagicMock()]
        mock_context_store = MagicMock()
        mock_context_store.create_tools.return_value = mock_tools

        mock_prompt = MagicMock()
        mock_prompt.type = "rag_based"
        mock_prompt.context_store = mock_context_store
        mock_prompt.get_system_instructions.return_value = "RAG instructions"

        app = HeraldApp(prompt=mock_prompt)
        app.herald_agent()

        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        assert 'tools' in call_kwargs
        assert call_kwargs['tools'] == mock_tools

    @patch('herald.app._build_groq_model')
    @patch('herald.app.Runner')
    @patch('herald.app.Agent')
    @pytest.mark.asyncio
    async def test_run_success(self, mock_agent, mock_runner, mock_build_model):
        """Test successful run of query with a session."""
        mock_build_model.return_value = MagicMock(spec=OpenAIChatCompletionsModel)
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

    @patch('herald.app._build_groq_model')
    @patch('herald.app.Runner')
    @patch('herald.app.Agent')
    @pytest.mark.asyncio
    async def test_run_uses_session_for_history(self, mock_agent, mock_runner, mock_build_model):
        """Test that run passes the session to Runner so history is maintained."""
        mock_build_model.return_value = MagicMock(spec=OpenAIChatCompletionsModel)
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
