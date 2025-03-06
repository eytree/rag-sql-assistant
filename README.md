# RAG-powered SQL Assistant

This project demonstrates how to implement a Retrieval-Augmented Generation (RAG) system using a local Large Language Model (LLM) to convert natural language queries into SQL for a trading database.

## What is RAG?

Retrieval-Augmented Generation (RAG) is an approach that enhances the capabilities of large language models by incorporating external knowledge retrieval. Instead of relying solely on the model's parametric knowledge, RAG systems:

1. Retrieve relevant information from a knowledge base
2. Augment the prompt with this contextual information
3. Generate responses based on both the retrieved context and the original query

This approach significantly improves accuracy and reduces hallucinations, especially for domain-specific tasks like SQL generation.

## Project Overview

This project implements a RAG-based natural language to SQL converter for a trading database. The system:

- Maintains a knowledge base of database schema information
- Processes user questions about trading data
- Retrieves relevant schema information
- Constructs a specialized prompt for the LLM
- Generates accurate SQL queries with performance insights

## Getting Started

### Prerequisites

- Windows OS
- Python 3.10+ 
- Git (optional)

### Directory Structure

```
rag-sql-assistant/
│
├── data/
│   ├── schema.json         # Database schema information
│   └── sample_data.sql     # Sample data for the trading database
│
├── src/
│   ├── __init__.py
│   ├── database.py         # Database connection and operations
│   ├── embeddings.py       # Vector embeddings for schema elements
│   ├── knowledge_base.py   # RAG knowledge base management
│   ├── llm.py              # Local LLM integration
│   └── sql_generator.py    # SQL generation with RAG
│
├── examples/
│   └── example_queries.md  # Example queries and explanations
│
├── .gitignore
├── requirements.txt        # Project dependencies
├── setup_db.py             # Database setup script
└── app.py                  # Main application entry point
```

### Installation

1. Clone the repository (or create the directory structure manually):

```bash
git clone https://github.com/yourusername/rag-sql-assistant.git
cd rag-sql-assistant
```

2. Create and activate a virtual environment:

Using standard venv:
```bash
python -m venv venv
venv\Scripts\activate
```

Or using `uv` (faster alternative to pip):
```bash
pip install uv
uv venv
.venv\Scripts\activate
```

3. Install dependencies:

Using pip:
```bash
pip install -r requirements.txt
```

Or using `uv`:
```bash
uv pip install -r requirements.txt
```

4. Initialize the database:

```bash
python setup_db.py
```

5. Run the application:

```bash
python app.py
```

## Project Components

### Database Schema

The trading database models a simplified trading system with the following tables:

- `users`: People who can place trades
- `instruments`: Financial instruments that can be traded
- `trades`: Records of trade transactions
- `portfolios`: Collections of instruments owned by users
- `price_history`: Historical price data for instruments

The schema details are stored in `data/schema.json` and used to create the RAG knowledge base.

### RAG Implementation

The RAG system works through these components:

1. **Knowledge Base** (`knowledge_base.py`):
   - Processes database schema information
   - Creates and manages vector embeddings
   - Handles retrieval of relevant schema information
   
2. **Embeddings** (`embeddings.py`):
   - Creates embeddings for schema elements using SentenceTransformers
   - Performs vector similarity search
   
3. **Local LLM** (`llm.py`):
   - Integrates with a locally-run LLM using llama.cpp
   - Manages prompt construction and response generation
   
4. **SQL Generator** (`sql_generator.py`):
   - Combines retrieved context with user query
   - Creates specialized prompts for SQL generation
   - Processes LLM responses into SQL and performance insights

### Example Usage

The `app.py` script provides a simple command-line interface for interacting with the system:

```bash
python app.py
```

Input a natural language query:
```
Show me all trades made by user 'john_doe' last month for tech stocks with values over $10,000
```

The system will:
1. Parse your query
2. Retrieve relevant schema information
3. Construct a specialized LLM prompt
4. Generate SQL with performance insights
5. Display the results

## Dependencies

Major dependencies include:

- `sqlite3`: For the database backend
- `sentence-transformers`: For creating and managing embeddings
- `llama-cpp-python`: For running the local LLM
- `faiss-cpu`: For efficient vector search
- `rich`: For improved console output

See `requirements.txt` for the complete list.

## Notes on LLM Selection

This project uses llama.cpp to run models locally. Some good model options include:

- Llama-3-8B-Instruct (good balance of performance and resource usage)
- Mistral-7B-Instruct-v0.2 (efficient and good at structured tasks)
- Phi-3-mini-4k-instruct (lightweight option for basic queries)

You'll need to download your preferred model separately and configure its path in the application.

## Performance Considerations

- Local LLMs require significant RAM and CPU/GPU resources
- Embedding creation is computationally intensive but only performed once during setup
- Consider using a GPU for improved performance if available

## Next Steps

Potential enhancements:

- Web UI for easier interaction
- Support for more database types (PostgreSQL, MySQL, etc.)
- Query execution and result visualization
- Fine-tuning the LLM specifically for SQL generation