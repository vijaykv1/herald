"""Tests for the main Herald application."""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from herald.app import HeraldApp
from herald.context_manager.prompt_based import HeraldBasicPrompter
from herald.context_manager.rag_based import HeraldRAGContextManager


class TestHeraldApp:
    """Test cases for HeraldApp."""

    @patch('herald.app.SQLiteSession')
    def test_init_basic_prompt(self, mock_session):
        """Test initialization with basic prompt."""
        mock_prompt = MagicMock(spec=HeraldBasicPrompter)
        mock_prompt.type = "basic_prompt"
        
        app = HeraldApp(prompt=mock_prompt)
        
        assert app.prompt == mock_prompt
        assert app.uuid is not None
        mock_session.assert_called_once()

    @patch('herald.app.SQLiteSession')
    def test_init_rag_prompt(self, mock_session):
        """Test initialization with RAG prompt."""
        mock_prompt = MagicMock(spec=HeraldRAGContextManager)
        mock_prompt.type = "rag_based"
        mock_prompt.context_store = MagicMock()
        
        app = HeraldApp(prompt=mock_prompt)
        
        assert app.prompt == mock_prompt
        assert app.uuid is not None

    @patch('herald.app.SQLiteSession')
    @patch('herald.app.Agent')
    def test_herald_agent_basic(self, mock_agent, mock_session):
        """Test herald_agent creation with basic prompt."""
        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        mock_prompt.get_system_instructions.return_value = "Test instructions"
        
        app = HeraldApp(prompt=mock_prompt)
        agent = app.herald_agent()
        
        # Verify Agent was called with correct parameters
        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs['name'] == "heralder"
        assert call_kwargs['instructions'] == "Test instructions"
        assert call_kwargs['model'] == "gpt-5-nano"
        assert 'tools' not in call_kwargs

    @patch('herald.app.SQLiteSession')
    @patch('herald.app.Agent')
    def test_herald_agent_rag(self, mock_agent, mock_session):
        """Test herald_agent creation with RAG prompt."""
        mock_tool = MagicMock()
        mock_context_store = MagicMock()
        mock_context_store.create_tool.return_value = mock_tool
        
        mock_prompt = MagicMock()
        mock_prompt.type = "rag_based"
        mock_prompt.context_store = mock_context_store
        mock_prompt.get_system_instructions.return_value = "RAG instructions"
        
        app = HeraldApp(prompt=mock_prompt)
        agent = app.herald_agent()
        
        # Verify Agent was called with tools
        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        assert 'tools' in call_kwargs
        assert call_kwargs['tools'] == [mock_tool]

    @patch('herald.app.Runner')
    @patch('herald.app.SQLiteSession')
    @patch('herald.app.Agent')
    @patch('herald.app.trace')
    @patch('herald.app.gen_trace_id')
    @pytest.mark.asyncio
    async def test_run_success(self, mock_gen_trace_id, mock_trace, mock_agent, mock_session, mock_runner):
        """Test successful run of query."""
        # Setup mocks
        mock_gen_trace_id.return_value = "test_trace_id"
        mock_trace.return_value.__enter__ = MagicMock()
        mock_trace.return_value.__exit__ = MagicMock()
        
        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        mock_prompt.get_system_instructions.return_value = "Instructions"
        
        mock_result = MagicMock()
        mock_result.final_output = "Test response"
        mock_runner.run = AsyncMock(return_value=mock_result)
        
        app = HeraldApp(prompt=mock_prompt)
        
        # Run the query
        result_gen = app.run(message="Test query", history=[])
        results = []
        async for chunk in result_gen:
            results.append(chunk)
        
        # Verify response
        assert len(results) == 1
        assert results[0] == "Test response"
        
        # Verify Runner.run was called
        mock_runner.run.assert_called_once()

    @patch('herald.app.Runner')
    @patch('herald.app.SQLiteSession')
    @patch('herald.app.Agent')
    @patch('herald.app.trace')
    @patch('herald.app.gen_trace_id')
    @pytest.mark.asyncio
    async def test_run_with_history(self, mock_gen_trace_id, mock_trace, mock_agent, mock_session, mock_runner):
        """Test run with conversation history."""
        mock_gen_trace_id.return_value = "test_trace_id"
        mock_trace.return_value.__enter__ = MagicMock()
        mock_trace.return_value.__exit__ = MagicMock()
        
        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        mock_prompt.get_system_instructions.return_value = "Instructions"
        
        mock_result = MagicMock()
        mock_result.final_output = "Response with context"
        mock_runner.run = AsyncMock(return_value=mock_result)
        
        app = HeraldApp(prompt=mock_prompt)
        
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        result_gen = app.run(message="Follow-up question", history=history)
        results = []
        async for chunk in result_gen:
            results.append(chunk)
        
        assert len(results) == 1
        mock_runner.run.assert_called_once()

    @patch('herald.app.SQLiteSession')
    def test_session_id_unique(self, mock_session):
        """Test that each app instance has unique session ID."""
        # Make each call to SQLiteSession return a different mock instance
        mock_session.side_effect = [MagicMock(), MagicMock()]
        
        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        
        app1 = HeraldApp(prompt=mock_prompt)
        app2 = HeraldApp(prompt=mock_prompt)
        
        assert app1.uuid != app2.uuid
        assert app1.session != app2.session

    @patch('herald.app.SQLiteSession')
    def test_sqlite_session_creation(self, mock_session):
        """Test that SQLite session is created with correct parameters."""
        mock_prompt = MagicMock()
        mock_prompt.type = "basic_prompt"
        
        app = HeraldApp(prompt=mock_prompt)
        
        # Verify SQLiteSession was called with correct db_path
        mock_session.assert_called_once()
        call_kwargs = mock_session.call_args[1]
        assert call_kwargs['db_path'] == "herald_traces.db"
        assert 'session_id' in call_kwargs
