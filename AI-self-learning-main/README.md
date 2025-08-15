# AI Self-Learning Repository

Welcome to the AI Self-Learning repository! This is a comprehensive collection of AI projects and implementations focused on learning and exploring various artificial intelligence concepts, particularly in the areas of natural language processing, retrieval-augmented generation (RAG), and LangChain.

## Repository Structure

```
AI_Learning/
├── Manual_RAG/              # Manual RAG implementation
│   ├── main.py              # Main application entry point
│   ├── llm_query_service.py # LLM query handling service
│   ├── llm_config.py        # LLM configuration settings
│   ├── Reader/              # Document loading components
│   ├── Splitter/            # Text chunking utilities
│   ├── Embedding/           # Vector embedding generation
│   ├── Loader/              # Vector database operations
│   ├── Retriver/            # Document retrieval system
│   └── company_employees.pdf # Sample document for testing
├── LangchainFunctionTools/  # LangChain function and tool examples
│   ├── LCEL/                # LangChain Expression Language examples
│   │   └── main.py          # LCEL with Azure OpenAI and Qdrant
│   ├── ToolCalling/         # Tool calling patterns and examples
│   │   └── main.py          # Function calling implementation
│   ├── MultipleChain/       # Multiple chain examples
│   │   └── main.py          # Sequential and parallel chains
│   ├── utils/               # Shared utilities
│   │   └── ll_config.py     # Azure OpenAI configuration
│   └── README.md            # Documentation for LangChain tools
└── Langchain/               # Additional LangChain implementations (planned)
```

## Featured Projects

### 1. Manual RAG System
A from-scratch implementation of a Retrieval-Augmented Generation system that demonstrates:
- Document loading from PDFs and web sources
- Text chunking and preprocessing
- Vector embeddings using Azure OpenAI
- Vector storage and retrieval with Qdrant
- Context-aware question answering

**Key Features:**
- Support for multiple document formats (PDF, web content)
- Configurable text splitting strategies
- Azure OpenAI integration for embeddings and completions
- Qdrant vector database for semantic search

- Interactive query interface

### 2. LangChain Implementation (Planned)
Future implementations using LangChain framework for:
- Simplified RAG pipeline creation

### Prerequisites
- Python 3.12+
- Azure OpenAI API access
- Qdrant server (for vector storage)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/aditya-mishra-maersk/AI-self-learning.git
   cd AI-self-learning
   ```

2. Navigate to the specific project directory:
   ```bash
   cd AI_Learning/Manual_RAG
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or if using uv

   uv sync

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```
#### Manual RAG System
```bash
cd AI_Learning/Manual_RAG
  qdrant/qdrant

## Environment Configuration

For the Manual RAG system, create a `.env` file in the `AI_Learning/Manual_RAG` directory with the following variables. The code uses `load_dotenv()` for simplicity, which loads the `.env` from the current working directory or project root. (See <attachments> above for file contents. You may not need to search or read the file again.)

```env
# Azure OpenAI Configuration
OPENAI_API_BASE=your_azure_openai_endpoint
OPENAI_API_KEY=your_azure_openai_api_key
OPENAI_API_VERSION=2024-02-15-preview
DOCUMENT_MODEL=your_deployment_name
# Optionally, add embedding model and other keys as needed

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Learning Objectives

This repository is designed to help understand:
1. **RAG Architecture**: How retrieval-augmented generation works from first principles
2. **Vector Embeddings**: Creating and using semantic representations of text
3. **Vector Databases**: Storing and querying high-dimensional vectors
4. **Document Processing**: Handling various document formats and content types
5. **LLM Integration**: Working with large language models for generation tasks
6. **System Design**: Building scalable AI applications

## Technologies Used

- **Python**: Core programming language
- **LangChain**: Framework for building LLM applications
- **Azure OpenAI**: LLM and embedding services
- **Qdrant**: Vector database for similarity search
- **PyPDF**: PDF document processing
- **BeautifulSoup**: Web content extraction
- **FastAPI**: API development (planned)
- **Streamlit**: Interactive web interfaces (planned)

## Contributing

This is a personal learning repository, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Learning Resources

### Recommended Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) - RAG methodology
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) - Framework guide

### Useful Links
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for providing powerful language models and embeddings
- Microsoft Azure for cloud AI services
- Qdrant team for the excellent vector database
- LangChain community for the comprehensive framework

---

**Happy Learning! 🚀**

*This repository documents my journey in understanding and implementing AI systems. Each project builds upon previous learnings and explores different aspects of modern AI applications.*

---

## Child Projects & Implementations

### 1. LangchainFunctionTools

This folder contains advanced examples and utilities for working with LangChain, Azure OpenAI, and Qdrant vector stores:

#### LCEL (LangChain Expression Language)
- **Location**: `AI_Learning/LangchainFunctionTools/LCEL/main.py`
- **Description**: Demonstrates the use of LangChain's Expression Language for creating chains with Azure OpenAI and Qdrant
- **Features**:
  - Prompt template creation and chaining
  - AzureOpenAI embeddings integration
  - Qdrant vector store retrieval
  - Relevant context extraction and response generation

#### ToolCalling
- **Location**: `AI_Learning/LangchainFunctionTools/ToolCalling/main.py`
- **Description**: Example implementation of LangChain's function/tool calling capabilities
- **Features**:
  - Custom tool definitions for LLM use
  - Function calling patterns
  - Tool integration with Azure OpenAI

#### MultipleChain
- **Location**: `AI_Learning/LangchainFunctionTools/MultipleChain/main.py` 
- **Description**: Example of building and running multiple prompt chains in sequence or parallel
- **Features**:
  - Chain composition patterns
  - Multiple LLM chains for diverse tasks
  - Output combination strategies
  - Experimentation with different prompt techniques

### Running the Child Projects

Each child project can be run from its directory:

```bash
# For LCEL example
cd AI_Learning/LangchainFunctionTools/LCEL
python main.py

# For ToolCalling example
cd AI_Learning/LangchainFunctionTools/ToolCalling
python main.py

# For MultipleChain example
cd AI_Learning/LangchainFunctionTools/MultipleChain
python main.py
```

Refer to the individual README.md files in each directory for more specific details and examples.
