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

## Technical Implementation

The RAG system is implemented through a series of key components that work together to convert natural language queries into SQL:

### 1. Knowledge Base (`knowledge_base.py`)
The knowledge base is the core of the RAG system, responsible for:
- Loading and managing database schema information from `schema.json`
- Extracting relevant entities (tables, columns) from user queries
- Retrieving context through semantic search
- Formatting context for LLM prompts

Key methods:
```python
def retrieve_context(self, query: str) -> Dict[str, Any]:
    # 1. Extract mentioned entities
    entities = self.extract_query_entities(query)
    
    # 2. Get context for mentioned tables
    for table_name in entities["tables"]:
        table_info = self.db.get_table_schema(table_name)
        context["tables"].append(table_info)
    
    # 3. Perform semantic search
    search_results = self.search(query)
    
    # 4. Process and combine results
    return context
```

### 2. Embeddings (`embeddings.py`)
Handles vector embeddings for semantic search:
- Uses SentenceTransformers for creating embeddings
- Manages FAISS index for efficient similarity search
- Persists embeddings to disk for reuse

Key methods:
```python
def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    # 1. Generate query embedding
    query_embedding = self.model.encode([query])
    
    # 2. Search in FAISS index
    distances, indices = self.index.search(query_embedding, top_k)
    
    # 3. Return relevant schema elements
    return results
```

### 3. Local LLM (`llm.py`)
Manages the language model interaction:
- Loads and manages the GGUF model
- Constructs specialized prompts with RAG context
- Processes model responses into SQL

Key methods:
```python
def generate_sql(self, query: str, context: str) -> Dict[str, str]:
    # 1. Create prompt with context
    prompt = self.create_sql_prompt(query, context)
    
    # 2. Generate response
    response = self.generate(prompt)
    
    # 3. Extract SQL and explanations
    return {
        "sql": sql_query,
        "explanation": explanation,
        "performance_notes": performance_notes
    }
```

### 4. SQL Generator (`sql_generator.py`)
Orchestrates the RAG pipeline:
- Coordinates between knowledge base and LLM
- Handles query generation and execution
- Provides performance analysis

Key methods:
```python
def generate_sql(self, query: str) -> Dict[str, Any]:
    # 1. Retrieve context
    context = self.knowledge_base.retrieve_context(query)
    
    # 2. Format context
    formatted_context = self.knowledge_base.format_context_for_prompt(context)
    
    # 3. Generate SQL with LLM
    result = self.llm.generate_sql(query, formatted_context)
    
    # 4. Analyze performance
    performance_info = self.analyze_query(result["sql"])
    
    return result
```

### RAG Pipeline Flow

1. **Query Processing**:
   - User submits natural language query
   - System extracts mentioned tables and columns

2. **Context Retrieval**:
   - Knowledge base performs semantic search
   - Retrieves relevant schema information
   - Combines explicit and implicit context

3. **Prompt Construction**:
   - System formats context into structured prompt
   - Includes schema details, relationships, and performance notes
   - Prepares specialized prompt for SQL generation

4. **SQL Generation**:
   - LLM generates SQL query with context
   - System validates and analyzes query
   - Returns SQL with explanations and performance insights

5. **Query Execution**:
   - System executes generated SQL
   - Displays results in formatted table
   - Provides performance feedback

This implementation ensures accurate SQL generation by combining the LLM's capabilities with domain-specific knowledge from the database schema.

## Getting Started

### Prerequisites

- Windows OS
- **Python 3.12** (important: the app is not compatible with Python 3.13 due to typing module changes)
- Git (optional)

### Directory Structure

```
rag-sql-assistant/
│
├── data/
│   ├── schema.json         # Database schema information
│   ├── trading.db          # SQLite database (created by setup_db.py)
│   └── embeddings/         # Directory for storing vector embeddings
│
├── models/                 # Directory for LLM model files
│   └── [your-model].gguf   # Your downloaded language model file
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
├── .gitignore              # Git ignore file
├── .env                    # Environment configuration (created from .env.example)
├── .env.example            # Example environment configuration
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

5. **Download a Language Model**:

This step is **required** before running the application. You need to download a language model in GGUF format. There are two ways to do this:

**Option A: Using Hugging Face CLI (Recommended)**

The Hugging Face CLI tool makes downloading models easier:

```bash
# Install the Hugging Face CLI
pip install huggingface_hub

# Create a models directory
mkdir -p models

# Download a model using the CLI
# For Llama-3-8B-Instruct (Q5_K_M variant, ~5GB)
# huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./models/Meta-Llama-3-8B-Instruct

# For a smaller model (Phi-3-mini-4k-instruct, Q4_K_M variant, ~2GB)
# huggingface-cli download TheBloke/phi-3-mini-4k-instruct-GGUF phi-3-mini-4k-instruct.Q4_K_M.gguf --local-dir ./models

# For Mistral-7B-Instruct (Q5_K_M variant, ~5GB)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q5_K_M.gguf --local-dir ./models
```

**Option B: Manual Download**

Alternatively, download the model file manually from one of these sources:
- [Llama-3-8B-Instruct (recommended)](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF)
- [Mistral-7B-Instruct-v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- [Phi-3-mini-4k-instruct](https://huggingface.co/TheBloke/phi-3-mini-4k-instruct-GGUF)

Place the downloaded file in the `models` directory.

6. Create an environment configuration file:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to specify your model path (use the actual filename you downloaded)
# For example, if you downloaded llama-3-8b-instruct.Q5_K_M.gguf:
# LLM_MODEL_PATH=./models/llama-3-8b-instruct.Q5_K_M.gguf
```

Edit the `.env` file to point to your downloaded model. The content should look like:

```
# Path to the LLM model file
LLM_MODEL_PATH=./models/llama-3-8b-instruct.Q5_K_M.gguf

# Database configuration (optional - only if you want to change the default)
# DB_PATH=./data/trading.db
```

7. Run the application:

```bash
python app.py interactive
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

### Important: Model Download and Configuration

The language model is a critical component and must be downloaded separately:

1. **Where to download models**: Visit [Hugging Face](https://huggingface.co/) to find quantized models in GGUF format. Repositories by "TheBloke" offer many compatible options.

2. **Quantization levels**: Models come in different quantization levels (Q4_K_M, Q5_K_M, Q8_0, etc.). Lower quantization = smaller file size but potentially lower quality:
   - Q4 variants: ~4GB, fastest, lowest quality
   - Q5 variants: ~5GB, good balance of speed and quality
   - Q8 variants: ~8GB, highest quality, slowest

3. **System requirements**:
   - 8GB+ RAM for Q4 models
   - 16GB+ RAM for Q5/Q8 models
   - GPU acceleration is supported with CUDA if available

4. **Configuration**:
   - Place the model file in the `models` directory
   - Update the `.env` file with the correct path
   - You can adjust context length and other parameters in `src/llm.py`

If you encounter the error "Model path does not exist," it means you need to download a model and correctly configure the path in your `.env` file.

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