# 🎺 Herald

[![Pylint](https://github.com/vijaykv1/herald/actions/workflows/pylint.yml/badge.svg)](https://github.com/vijaykv1/herald/actions/workflows/pylint.yml)

**Herald** is an intelligent AI-powered chatbot that acts as your personal representative, answering questions about your 
professional background, skills, and experience based on your CV or LinkedIn profile.

## 🌟 Features

- **Dual Interface Support**: Choose between browser-based (Gradio) or terminal-based UI
- **Multiple Context Strategies**:
  - **Basic Prompt**: Direct CV content injection into the system prompt
  - **RAG-based**: Retrieval-Augmented Generation using vector embeddings for more accurate, context-aware responses
- **LinkedIn CV Parser**: Specialized parser for LinkedIn-generated CVs with intelligent section extraction
- **First-Person Responses**: The bot speaks as you, providing natural conversational experiences
- **Conversation History**: Maintains session context using SQLite for coherent multi-turn conversations
- **Trace Management**: Built-in tracing for debugging and monitoring LLM interactions

## 🏗️ Architecture

```
herald/
├── app.py                    # Main Herald application logic
├── cv_parser/                # CV parsing modules
│   ├── iparser.py           # Parser interface
│   └── linkedin.py          # LinkedIn CV parser implementation
└── context_manager/          # Context management strategies
    ├── icontext.py          # Context interface
    ├── prompt_based.py      # Basic prompt-based context
    ├── rag_based.py         # RAG-based context with vector search
    └── rag.py               # Vector store implementation using ChromaDB
```

## 📋 Prerequisites

- Python >= 3.12
- OpenAI API key
- PDF CV file (or markdown/text format)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/vijaykv1/herald.git
cd herald
```

2. Install dependencies using [uv](https://github.com/astral-sh/uv) (recommended) or pip:
```bash
# Using uv (recommended)
uv venv
uv sync

# Or using pip
pip install -e .
```

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```env
# Required: Path to your CV (PDF format)
CV_PATH=/path/to/your/cv.pdf

# Required: Your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Your name (used in prompts)
ME=Your Name

# Optional: UI mode - "yes" for browser, "no" for terminal (default: "no")
WITH_BROWSER=no

# Optional: Context strategy - "basic" or "rag" (default: "basic")
PROMPT_OPTION=basic
```

## 🎯 Usage

### Terminal Mode (Default)

```bash
python main.py
```

This launches an interactive terminal interface where you can ask questions about the CV:

```
🎺 The Herald
Ask questions about the CV (type 'exit' to quit)

Query: What is your experience with Python?
Answer: I have worked with Python for...
```

### Browser Mode (Gradio)

Set `WITH_BROWSER=yes` in your `.env` file, then run:

```bash
python main.py
```

This launches a web interface accessible at `http://localhost:7860` (or another port shown in the terminal).

## 🧠 Context Strategies

### Basic Prompt (`PROMPT_OPTION=basic`)

- Injects the entire CV content directly into the system prompt
- Simple and fast
- Best for smaller CVs
- No additional setup required

### RAG-based (`PROMPT_OPTION=rag`)

- Uses vector embeddings and semantic search
- Retrieves only relevant CV sections for each query
- Better for larger CVs
- More accurate for specific queries
- Automatically creates a local ChromaDB vector store

## 🛠️ Development

### Running Tests

```bash
# Using uv
uv run pytest

# Using pip
pytest
```

### Code Quality

The project uses Pylint for code quality checks:

```bash
# Using uv
uv run pylint $(git ls-files '*.py')

# Using pip
pylint $(git ls-files '*.py')
```

Configuration can be found in `pyproject.toml`:
- Max line length: 120 characters
- Disabled checks: `fixme`

### Project Structure

- **`main.py`**: Entry point for the application
- **`herald/app.py`**: Core application logic and agent setup
- **`herald/cv_parser/`**: CV parsing implementations
  - LinkedIn CV parser with section extraction
  - Experience parsing with job details
- **`herald/context_manager/`**: Different context strategies
  - Basic prompt-based context
  - RAG-based context with vector retrieval
  - Vector store using ChromaDB and OpenAI embeddings

## 📦 Dependencies

Key dependencies include:

- **openai-agents**: For building the conversational agent
- **gradio**: Browser-based UI
- **chromadb**: Vector database for RAG
- **pymupdf4llm**: PDF to markdown conversion
- **langchain-text-splitters**: Text chunking for RAG
- **rich**: Beautiful terminal output
- **pydantic**: Data validation

See `pyproject.toml` for the complete list.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Built with [OpenAI Agents](https://github.com/openai/openai-agents-python)
- UI powered by [Gradio](https://gradio.app/)
- Vector storage with [ChromaDB](https://www.trychroma.com/)

## 📞 Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/vijaykv1/herald).

---

Made with ❤️ by the Herald team
