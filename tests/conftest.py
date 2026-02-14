"""Pytest configuration and shared fixtures."""

import os
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_cv_content():
    """Sample CV content for testing."""
    return """# John Doe

## Summary
Experienced software engineer with 5 years in Python development.

## Experience

### Senior Developer
**Company A** | 2020 - Present
- Led development of microservices
- Implemented CI/CD pipelines

### Junior Developer
**Company B** | 2018 - 2020
- Developed REST APIs
- Worked with Python and Django

## Skills
Python, AWS, Docker, Kubernetes

## Education
BS in Computer Science, University XYZ, 2018
"""


@pytest.fixture
def sample_linkedin_cv():
    """Sample LinkedIn CV in markdown format."""
    return """# John Doe

## Contact
john.doe@email.com
linkedin.com/in/johndoe

## Skills
Python, AWS, Machine Learning

## Experience

### Senior Software Engineer
Company A
January 2020 - Present

- Developed microservices architecture
- Led team of 5 developers

### Software Engineer
Company B
June 2018 - December 2019

- Built REST APIs
- Implemented testing frameworks

## Education

### Bachelor of Science in Computer Science
University XYZ
2014 - 2018
"""


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = MagicMock()
    
    # Mock embeddings API
    mock_embedding = MagicMock()
    mock_embedding.data = [MagicMock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_embedding
    
    return mock_client


@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB collection."""
    mock_collection = MagicMock()
    mock_collection.add = MagicMock()
    mock_collection.query = MagicMock(return_value={
        'documents': [['Sample document 1', 'Sample document 2']],
        'distances': [[0.1, 0.2]],
        'metadatas': [[{'topic': 'Experience'}, {'topic': 'Skills'}]]
    })
    return mock_collection


@pytest.fixture
def mock_chromadb_client(mock_chromadb_collection):
    """Mock ChromaDB client."""
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_chromadb_collection
    return mock_client


@pytest.fixture
def temp_cv_pdf(tmp_path):
    """Create a temporary PDF file path."""
    cv_path = tmp_path / "test_cv.pdf"
    # Don't create actual PDF, just return path for mocking
    return str(cv_path)


@pytest.fixture
def sample_cv_chunks():
    """Sample CV chunks for vector store testing."""
    return [
        {"topic": "name", "content": "John Doe"},
        {"topic": "Summary", "content": "Experienced software engineer"},
        {"topic": "Experience", "content": {
            "title": "Senior Developer",
            "company": "Company A",
            "duration": "2020 - Present",
            "description": "Led development of microservices"
        }},
        {"topic": "Skills", "content": "Python, AWS, Docker"}
    ]


@pytest.fixture
def mock_env_vars(monkeypatch, temp_cv_pdf):
    """Mock environment variables."""
    monkeypatch.setenv("CV_PATH", temp_cv_pdf)
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("ME", "Test User")
    monkeypatch.setenv("WITH_BROWSER", "no")
    monkeypatch.setenv("PROMPT_OPTION", "basic")
