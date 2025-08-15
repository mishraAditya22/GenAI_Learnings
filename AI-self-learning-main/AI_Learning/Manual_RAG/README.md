# LangChain RAG Application for Document Querying

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Azure OpenAI services to query and retrieve information from various document sources.

## Project Overview

The application allows users to:
1. Load documents from PDF files and web URLs
2. Split text into manageable chunks
3. Generate embeddings using Azure OpenAI
4. Store and query vectors in Qdrant (vector database)
5. Retrieve relevant document sections based on semantic similarity
6. Generate accurate responses using LLMs

## Project Structure

```
ChatWithData/
├── main.py                  # Main script coordinating all components
├── llm_query_service.py     # Service for querying LLMs with context
├── Reader/                  # Document loading components
│   └── main.py              # Loads PDFs and web content
├── Splitter/                # Text chunking components
│   └── main.py              # Splits documents into chunks
├── Embedding/               # Vector embedding components
│   └── main.py              # Generates embeddings using Azure OpenAI
├── Loader/                  # Vector database components
│   └── main.py              # Manages Qdrant client operations
├── Retriver/                # Retrieval components
│   └── main.py              # Facilitates document retrieval
└── company_employees.pdf    # Sample PDF document
```

## Prerequisites

- Python 3.12+
- Qdrant server running (default: localhost:6333)
- Azure OpenAI API credentials

## Dependencies

- langchain, langchain-community, langchain-openai
- openai
- python-dotenv
- qdrant-client
- pypdf
- beautifulsoup4
- tiktoken
- pydantic

## Setup

1. Clone the repository
2. Create a `.env` file with your Azure OpenAI credentials:

```
DOCUMENT_MODEL=<your-azure-deployment-name>
EMBEDDING_MODEL=<your-embedding-model-name>
EMBEDDING_OPENAI_API_BASE=<your-azure-openai-endpoint>
EMBEDDING_OPENAI_API_KEY=<your-api-key>
EMBEDDING_OPENAI_API_VERSION=<your-api-version>
OPENAI_API_BASE=<your-azure-openai-endpoint>
OPENAI_API_KEY=<your-api-key>
OPENAI_API_VERSION=<your-api-version>
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Uncomment the data processing sections in `main.py`
2. Add your documents to the project
3. Update the document paths in `main.py`
4. Run the application:

```bash
python main.py
```

5. Modify the query in `main.py` to ask questions about your documents

## How It Works

1. **Document Loading**: The `DocumentLoader` class loads content from PDF files and web URLs
2. **Text Splitting**: `GetTextSplitters` breaks documents into manageable chunks
3. **Embedding Generation**: `EmbeddingManager` converts text chunks into vector embeddings
4. **Vector Storage**: `QdrantClientManager` stores and retrieves vectors from Qdrant
5. **Query Processing**: User queries are converted to embeddings and matched with stored vectors
6. **Response Generation**: `LLMQueryService` uses the retrieved context to generate responses

## Example

```python
query = "What do you know about httpstreamable MCP servers?"
query_vector = embedding_manager.embed_query(query)
search_results = QdrantClientManager.search(collection_name, query_vector, limit=5)
llm_service = LLMQueryService()
llm_response = llm_service.query_llm(query, search_results)
print(llm_response)
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
