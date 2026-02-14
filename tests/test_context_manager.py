"""Tests for context manager modules."""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from herald.context_manager.icontext import ContextInterface
from herald.context_manager.prompt_based import HeraldBasicPrompter
from herald.context_manager.rag_based import HeraldRAGContextManager


class TestContextInterface:
    """Test cases for ContextInterface."""

    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_prepare_cv_content_with_file(self, mock_exists, mock_to_markdown, sample_cv_content):
        """Test preparing CV content from PDF file."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        result = ContextInterface.prepare_cv_content("test.pdf")
        
        assert result == sample_cv_content
        mock_to_markdown.assert_called_once_with("test.pdf")

    @patch('os.path.exists')
    def test_prepare_cv_content_file_not_exists(self, mock_exists):
        """Test error when CV file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(ValueError, match="does not exist"):
            ContextInterface.prepare_cv_content("nonexistent.pdf")

    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    @patch.dict(os.environ, {'CV_PATH': '/path/to/cv.pdf'})
    def test_prepare_cv_content_from_env(self, mock_exists, mock_to_markdown, sample_cv_content):
        """Test preparing CV content from environment variable."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        result = ContextInterface.prepare_cv_content()
        
        assert result == sample_cv_content
        mock_to_markdown.assert_called_once_with('/path/to/cv.pdf')

    def test_basic_system_instructions(self, sample_cv_content):
        """Test basic system instructions generation."""
        
        class DummyContext(ContextInterface):
            def __init__(self):
                self._cv_pdf_file = None
                self._cv_md_content = sample_cv_content
            
            @property
            def type(self):
                return "dummy"
            
            def get_system_instructions(self):
                return self.basic_system_instructions()
        
        with patch.dict(os.environ, {'ME': 'John Doe'}):
            context = DummyContext()
            instructions = context.basic_system_instructions()
            
            assert "John Doe" in instructions
            assert "helpful assistant" in instructions
            assert "professional background" in instructions


class TestHeraldBasicPrompter:
    """Test cases for HeraldBasicPrompter."""

    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_init(self, mock_exists, mock_to_markdown, sample_cv_content):
        """Test initialization of HeraldBasicPrompter."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        prompter = HeraldBasicPrompter("test.pdf")
        
        assert prompter._cv_md_content == sample_cv_content
        assert prompter.type == "basic_prompt"

    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    @patch.dict(os.environ, {'ME': 'Test User'})
    def test_get_system_instructions(self, mock_exists, mock_to_markdown, sample_cv_content):
        """Test system instructions generation."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        prompter = HeraldBasicPrompter("test.pdf")
        instructions = prompter.get_system_instructions()
        
        # Should contain basic instructions
        assert "Test User" in instructions
        assert "helpful assistant" in instructions
        
        # Should contain CV content
        assert sample_cv_content in instructions
        assert "Knowledge Base" in instructions

    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_type_property(self, mock_exists, mock_to_markdown, sample_cv_content):
        """Test type property returns correct value."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        prompter = HeraldBasicPrompter("test.pdf")
        assert prompter.type == "basic_prompt"


class TestHeraldRAGContextManager:
    """Test cases for HeraldRAGContextManager."""

    @patch('herald.context_manager.rag_based.CVVectorStore')
    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_init(self, mock_exists, mock_to_markdown, mock_vector_store, sample_cv_content):
        """Test initialization of RAG context manager."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        rag_manager = HeraldRAGContextManager("test.pdf")
        
        assert rag_manager._cv_md_content == sample_cv_content
        assert rag_manager.type == "rag_based"
        assert rag_manager.vector_store is not None

    @patch('herald.context_manager.rag_based.CVVectorStore')
    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_type_property(self, mock_exists, mock_to_markdown, mock_vector_store, sample_cv_content):
        """Test type property returns correct value."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        rag_manager = HeraldRAGContextManager("test.pdf")
        assert rag_manager.type == "rag_based"

    @patch('herald.context_manager.rag_based.CVVectorStore')
    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_context_store_property(self, mock_exists, mock_to_markdown, mock_vector_store, sample_cv_content):
        """Test context_store property returns vector store."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        rag_manager = HeraldRAGContextManager("test.pdf")
        assert rag_manager.context_store == rag_manager.vector_store

    @patch('herald.context_manager.rag_based.CVVectorStore')
    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    @patch.dict(os.environ, {'ME': 'Test User'})
    def test_get_system_instructions(self, mock_exists, mock_to_markdown, mock_vector_store, sample_cv_content):
        """Test system instructions for RAG-based approach."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        rag_manager = HeraldRAGContextManager("test.pdf")
        instructions = rag_manager.get_system_instructions()
        
        # Should contain instructions about using the retrieval tool
        assert "Test User" in instructions
        assert "retrieve_relevant_chunks" in instructions
        assert "tool" in instructions.lower()
        assert "retrieval" in instructions.lower()

    @patch('herald.context_manager.rag_based.LinkedInCVParser')
    @patch('herald.context_manager.rag_based.CVVectorStore')
    @patch('herald.context_manager.icontext.pymupdf4llm.to_markdown')
    @patch('os.path.exists')
    def test_prepare_vector_store_called(
        self, mock_exists, mock_to_markdown, mock_vector_store, mock_parser, sample_cv_content
    ):
        """Test that vector store is prepared during initialization."""
        mock_exists.return_value = True
        mock_to_markdown.return_value = sample_cv_content
        
        # Setup mock parser
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = []
        mock_parser.return_value = mock_parser_instance
        
        rag_manager = HeraldRAGContextManager("test.pdf")
        
        # Vector store should be created
        assert rag_manager.vector_store is not None
