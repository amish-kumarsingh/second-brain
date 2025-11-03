# 🧠 Second Brain

A personal knowledge management system powered by RAG (Retrieval-Augmented Generation) that helps you organize, retrieve, and reason over your personal notes and documents.

## Overview

Second Brain is an intelligent assistant that combines vector search, LLM reasoning, and conversation memory to help you interact with your personal knowledge base. It ingests your notes, stores them in a vector database, and uses AI to answer questions based on your stored knowledge while maintaining context from past conversations.

## Features

- **📚 Document Ingestion**: Automatically processes and stores text files from your notes folder
- **🔍 Vector Search**: Semantic search using ChromaDB and sentence transformers for finding relevant information
- **🤖 AI Reasoning**: Uses Pydantic AI with Google Gemini to synthesize information and answer questions
- **💭 Conversation Memory**: Maintains context from previous interactions for more coherent conversations
- **🛡️ PII Protection**: Automatically detects and redacts personally identifiable information (PII)
- **📊 Observability**: Built-in OpenTelemetry tracing and Logfire integration for monitoring
- **✅ Evaluation Framework**: Test and evaluate agent responses and retrieval quality
- **🎯 Interactive CLI**: User-friendly command-line interface for managing your knowledge base

## Architecture

The system consists of several key components:

- **RAGManager**: Handles document ingestion, vector storage (ChromaDB), and semantic retrieval
- **ThoughtAgent**: AI agent that combines RAG retrieval with LLM reasoning using Pydantic AI
- **MemoryManager**: Stores and retrieves conversation history for context-aware responses
- **PIIGuard**: Protects sensitive information using Guardrails AI or regex-based detection
- **Evaluation System**: Test datasets and evaluators for measuring system performance

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd second_brain
```

2. Install dependencies:
```bash
uv sync
```

3. Create a `.env` file with your configuration:
```bash
# Optional: Set LLM model (defaults to google-gla:gemini-2.5-pro)
LLM_MODEL=google-gla:gemini-2.5-pro

# Optional: Enable/disable PII guardrails (defaults to true)
GUARDRAILS_ENABLED=true
```

4. Add your notes to the `data/notes/` folder (or use your own folder path)

## Usage

### Interactive Mode

Run the main application:
```bash
uv run second-brain
```

The interactive menu provides the following options:

1. **Ingest all data into memory** - Process all `.txt` files from `data/notes/` and store them in ChromaDB
2. **Ask your Second Brain a question** - Query your knowledge base with natural language
3. **Reset (delete) all stored data** - Clear the ChromaDB collection (requires confirmation)
4. **Test Thought Agent** - Directly interact with the AI agent for reasoning tasks
5. **Clear Memory** - Clear conversation history (requires confirmation)
6. **Exit** - Close the application

### Example Workflow

1. Place your text files in `data/notes/`
2. Run the application and select option 1 to ingest your notes
3. Ask questions like:
   - "What are my learning goals?"
   - "Suggest me some travel ideas"
   - "What were the key points from the AI team meeting?"
4. The system will retrieve relevant context and provide AI-generated answers

### Programmatic Usage

```python
from second_brain.agents.ingestor import RAGManager
from second_brain.agents.thought_agent import ThoughtAgent

# Ingest documents
rag_manager = RAGManager()
rag_manager.ingest_folder("data/notes")

# Query documents
results = rag_manager.query_notes("What are my goals?")

# Use the Thought Agent
agent = ThoughtAgent()
response = agent.run("What should I focus on learning next?")
print(response)
```

## Evaluation

The project includes an evaluation framework for testing agent performance:

```bash
# Run agent evaluation
uv run python -m second_brain.eval.run_eval

# Or run specific evaluations
uv run python -c "from second_brain.eval import run_agent_evaluation; run_agent_evaluation()"
```

The evaluation system tests:
- Agent response quality using LLM-based judges
- Retrieval accuracy for knowledge base queries
- Handling of different query types and difficulties

## Project Structure

```
second_brain/
├── data/
│   └── notes/              # Place your .txt files here
├── src/
│   └── second_brain/
│       ├── agents/
│       │   ├── ingestor.py        # RAGManager: document ingestion & retrieval
│       │   ├── memory_manager.py   # Conversation memory management
│       │   └── thought_agent.py   # AI agent with RAG + LLM reasoning
│       ├── eval/                  # Evaluation framework
│       ├── memory/                # Memory storage (JSON)
│       ├── utils/
│       │   ├── guardrails.py      # PII detection and redaction
│       │   └── otel_setup.py      # OpenTelemetry configuration
│       └── main.py                # CLI entry point
├── pyproject.toml
└── README.md
```

## Technologies

- **Pydantic AI**: AI agent framework for LLM interactions
- **ChromaDB**: Vector database for document storage and retrieval
- **Sentence Transformers**: Embedding generation (all-MiniLM-L6-v2)
- **LangChain**: Text splitting utilities
- **Guardrails AI**: PII detection and redaction
- **OpenTelemetry / Logfire**: Observability and distributed tracing
- **Google Gemini**: LLM for reasoning and synthesis

## Configuration

### Environment Variables

- `LLM_MODEL`: LLM model identifier (default: `google-gla:gemini-2.5-pro`)
- `GUARDRAILS_ENABLED`: Enable/disable PII protection (default: `true`)

### Data Storage

- ChromaDB collection: Stored in `.chromadb/` (excluded from git)
- Memory: Stored in `src/second_brain/memory/memory_data.json`
- Sentence Transformer cache: `~/.cache/sentence_transformers/`

## Privacy & Security

The system includes built-in PII protection:
- Automatically detects and redacts sensitive information (emails, phone numbers, SSN, credit cards, etc.)
- Uses Guardrails AI when available, falls back to regex-based detection
- All user inputs and stored memories are sanitized before processing and storage

## Development

### Running Tests

```bash
# Run evaluation suite
uv run python -m second_brain.eval.run_eval
```

### Adding New Features

1. Agents: Add new agent types in `src/second_brain/agents/`
2. Evaluations: Add test cases in `src/second_brain/eval/`
3. Utilities: Add helper functions in `src/second_brain/utils/`

---

Built with ❤️ for personal knowledge management

