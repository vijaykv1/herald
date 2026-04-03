"""Tests for RAG vector store module."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from herald.context_manager.rag import CVVectorStore


class TestCVVectorStore:
    """Test cases for CVVectorStore."""

    @patch('herald.context_manager.rag.chromadb.Client')
    @patch('herald.context_manager.rag.OpenAI')
    def test_init(self, mock_openai, mock_chromadb, sample_cv_chunks):
        """Test initialization of CVVectorStore."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client
        
        vector_store = CVVectorStore(sample_cv_chunks)
        
        assert vector_store._CVVectorStore__cv_chunks == sample_cv_chunks
        mock_chromadb.assert_called_once()
        mock_client.create_collection.assert_called_once_with(name="cv_lookup")

    @patch('herald.context_manager.rag.chromadb.Client')
    @patch('herald.context_manager.rag.OpenAI')
    def test_init_uses_in_memory_client(self, mock_openai, mock_chromadb, sample_cv_chunks):
        """CVVectorStore uses an in-memory ChromaDB client (no persist_directory)."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client

        CVVectorStore(sample_cv_chunks)

        # Client must be called with no arguments (in-memory mode)
        mock_chromadb.assert_called_once_with()

    def test_normalize_chunk_string_content(self):
        """Test chunk normalization with string content."""
        chunk = {"topic": "Skills", "content": "Python, AWS, Docker"}
        
        # We need to access the private method
        with patch('herald.context_manager.rag.chromadb.Client'):
            with patch('herald.context_manager.rag.OpenAI'):
                vector_store = CVVectorStore([])
                normalized = vector_store._CVVectorStore__normalize_chunk(chunk)
        
        assert "Skills" in normalized
        assert "Python, AWS, Docker" in normalized
        assert "CV Section:" in normalized

    def test_normalize_chunk_dict_content(self):
        """Test chunk normalization with dictionary content."""
        chunk = {
            "topic": "Experience",
            "content": {
                "title": "Senior Developer",
                "company": "Company A",
                "duration": "2020-Present"
            }
        }
        
        with patch('herald.context_manager.rag.chromadb.Client'):
            with patch('herald.context_manager.rag.OpenAI'):
                vector_store = CVVectorStore([])
                normalized = vector_store._CVVectorStore__normalize_chunk(chunk)
        
        assert "Experience" in normalized
        assert "Senior Developer" in normalized
        assert "Company A" in normalized

    @patch('herald.context_manager.rag.tqdm.tqdm')
    @patch('herald.context_manager.rag.chromadb.Client')
    @patch('herald.context_manager.rag.OpenAI')
    def test_vectorize_chunks(self, mock_openai, mock_chromadb, mock_tqdm, sample_cv_chunks):
        """Test vectorizing and storing CV chunks."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client
        
        mock_openai_instance = MagicMock()
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai_instance.embeddings.create.return_value = mock_embedding_response
        mock_openai.return_value = mock_openai_instance
        
        # Make tqdm return the input iterable
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        vector_store = CVVectorStore(sample_cv_chunks)
        vector_store.vectorize_chunks()
        
        # Verify embeddings were created for each chunk
        assert mock_openai_instance.embeddings.create.call_count == len(sample_cv_chunks)
        
        # Verify chunks were added to collection
        assert mock_collection.add.call_count == len(sample_cv_chunks)

    @patch('herald.context_manager.rag.chromadb.Client')
    @patch('herald.context_manager.rag.OpenAI')
    def test_retrieve_relevant_chunks(self, mock_openai, mock_chromadb, sample_cv_chunks):
        """Test retrieving relevant chunks based on query."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client
        
        mock_openai_instance = MagicMock()
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai_instance.embeddings.create.return_value = mock_embedding_response
        mock_openai.return_value = mock_openai_instance
        
        # Mock collection query response
        mock_collection.query.return_value = {
            'documents': [['Document 1', 'Document 2']],
            'distances': [[0.1, 0.2]],
            'metadatas': [[{'topic': 'Skills'}, {'topic': 'Experience'}]]
        }
        
        vector_store = CVVectorStore(sample_cv_chunks)
        results = vector_store.retrieve_relevant_chunks("Python experience", top_k=2)
        
        # Verify query embedding was created
        mock_openai_instance.embeddings.create.assert_called()
        
        # Verify collection was queried
        mock_collection.query.assert_called_once()
        
        # Verify results
        assert isinstance(results, list)

    @patch('herald.context_manager.rag.chromadb.Client')
    @patch('herald.context_manager.rag.OpenAI')
    def test_retrieve_relevant_chunks_custom_top_k(self, mock_openai, mock_chromadb, sample_cv_chunks):
        """Test retrieving chunks with custom top_k value."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client
        
        mock_openai_instance = MagicMock()
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai_instance.embeddings.create.return_value = mock_embedding_response
        mock_openai.return_value = mock_openai_instance
        
        mock_collection.query.return_value = {
            'documents': [['Doc1', 'Doc2', 'Doc3']],
            'distances': [[0.1, 0.2, 0.3]],
            'metadatas': [[{'topic': 'A'}, {'topic': 'B'}, {'topic': 'C'}]]
        }
        
        vector_store = CVVectorStore(sample_cv_chunks)
        results = vector_store.retrieve_relevant_chunks("test query", top_k=3)
        
        # Check that query was called with correct n_results
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs['n_results'] == 3

    @patch('herald.context_manager.rag.function_tool')
    @patch('herald.context_manager.rag.chromadb.Client')
    @patch('herald.context_manager.rag.OpenAI')
    def test_create_tools(self, mock_openai, mock_chromadb, mock_function_tool, sample_cv_chunks):
        """Test creating topic-specific tools for agent use."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client

        mock_tool = MagicMock()
        mock_function_tool.return_value = mock_tool

        vector_store = CVVectorStore(sample_cv_chunks)
        tools = vector_store.create_tools()

        # Verify function_tool was called once per tool and a list is returned
        assert mock_function_tool.call_count == 6  # 6 topic-specific tools
        assert isinstance(tools, list)
        assert len(tools) == 6

    @patch('herald.context_manager.rag.chromadb.Client')
    @patch('herald.context_manager.rag.OpenAI')
    def test_create_tools_inner_function_delegates(self, mock_openai, mock_chromadb, sample_cv_chunks):
        """Test that calling an inner tool function delegates to retrieve_relevant_chunks."""
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
            'documents': [['Relevant chunk']],
            'distances': [[0.1]],
            'metadatas': [[{'topic': 'Skills'}]],
        }

        # Use function_tool as identity so the inner fns are directly callable
        with patch('herald.context_manager.rag.function_tool', side_effect=lambda f: f):
            vector_store = CVVectorStore(sample_cv_chunks)
            tools = vector_store.create_tools()
            # retrieve_experience_chunks is the second tool (index 1)
            retrieve_experience = tools[1]
            result = retrieve_experience(query="Python skills", top_k=1)

        assert result == ['Relevant chunk']
