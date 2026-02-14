"""Integration tests for Herald application."""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestIntegration:
    """Integration tests combining multiple components."""

    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_basic_prompter_end_to_end(self, mock_exists, mock_to_markdown):
        """Test basic prompter from initialization to instruction generation."""
        from herald.context_manager.prompt_based import HeraldBasicPrompter
        
        mock_exists.return_value = True
        mock_to_markdown.return_value = "# CV Content\n\nExperience with Python"
        
        with patch.dict('os.environ', {'ME': 'Integration Test User'}):
            prompter = HeraldBasicPrompter("test.pdf")
            instructions = prompter.get_system_instructions()
            
            assert prompter.type == "basic_prompt"
            assert "Integration Test User" in instructions
            assert "CV Content" in instructions
            assert "Python" in instructions

    @patch('herald.context_manager.rag_based.CVVectorStore')
    @patch('herald.context_manager.rag_based.LinkedInCVParser')
    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_rag_manager_end_to_end(self, mock_exists, mock_to_markdown, mock_parser, mock_vector_store):
        """Test RAG manager from initialization to instruction generation."""
        from herald.context_manager.rag_based import HeraldRAGContextManager
        
        mock_exists.return_value = True
        mock_to_markdown.return_value = "# CV\n\n## Skills\nPython, AWS"
        
        # Mock the parser to return empty chunks
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = []
        mock_parser.return_value = mock_parser_instance
        
        # Mock the vector store instance
        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance
        
        with patch.dict('os.environ', {'ME': 'RAG Test User'}):
            rag_manager = HeraldRAGContextManager("test.pdf")
            instructions = rag_manager.get_system_instructions()
            
            assert rag_manager.type == "rag_based"
            assert "RAG Test User" in instructions
            assert "retrieve_relevant_chunks" in instructions
            assert rag_manager.context_store == mock_store_instance

    @patch('herald.app.Runner')
    @patch('herald.app.SQLiteSession')
    @patch('herald.app.Agent')
    @patch('herald.app.trace')
    @patch('herald.app.gen_trace_id')
    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    @pytest.mark.asyncio
    async def test_full_query_flow_basic(
        self, mock_exists, mock_to_markdown, mock_gen_trace_id,
        mock_trace, mock_agent, mock_session, mock_runner
    ):
        """Test complete query flow with basic prompt."""
        from herald.app import HeraldApp
        from herald.context_manager.prompt_based import HeraldBasicPrompter
        
        # Setup mocks
        mock_exists.return_value = True
        mock_to_markdown.return_value = "# Test CV"
        mock_gen_trace_id.return_value = "trace_123"
        mock_trace.return_value.__enter__ = MagicMock()
        mock_trace.return_value.__exit__ = MagicMock()
        
        mock_result = MagicMock()
        mock_result.final_output = "I have 5 years of Python experience"
        
        from unittest.mock import AsyncMock
        mock_runner.run = AsyncMock(return_value=mock_result)
        
        # Create app with basic prompter
        with patch.dict('os.environ', {'ME': 'Test User'}):
            prompter = HeraldBasicPrompter("test.pdf")
            app = HeraldApp(prompt=prompter)
            
            # Run query
            results = []
            async for chunk in app.run(
                message="How many years of Python experience do you have?",
                history=[]
            ):
                results.append(chunk)
            
            assert len(results) == 1
            assert "Python experience" in results[0]

    @patch('herald.context_manager.rag.chromadb.Client')
    @patch('herald.context_manager.rag.OpenAI')
    def test_vector_store_workflow(self, mock_openai, mock_chromadb):
        """Test vector store creation and retrieval workflow."""
        from herald.context_manager.rag import CVVectorStore
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client
        
        mock_openai_instance = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai_instance.embeddings.create.return_value = mock_embedding
        mock_openai.return_value = mock_openai_instance
        
        mock_collection.query.return_value = {
            'documents': [['Found relevant info']],
            'distances': [[0.1]],
            'metadatas': [[{'topic': 'Skills'}]]
        }
        
        # Create vector store
        chunks = [
            {"topic": "Skills", "content": "Python, AWS"},
            {"topic": "Experience", "content": "5 years"}
        ]
        
        with patch('herald.context_manager.rag.tqdm.tqdm', lambda x, **kwargs: x):
            vector_store = CVVectorStore(chunks)
            vector_store.vectorize_chunks()
            results = vector_store.retrieve_relevant_chunks("Python skills", top_k=1)
        
        # Verify workflow
        assert mock_collection.add.call_count == len(chunks)
        assert isinstance(results, list)
