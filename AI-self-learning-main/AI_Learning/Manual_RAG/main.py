import sys
import os
from Reader.main import DocumentLoader
from Splitter.main import GetTextSplitters
from Loader.main import QdrantClientManager

# Import embedding manager class
from Embedding.main import EmbeddingManager
from utils.llm_query_service import LLMQueryService

# Get the loaded documents in text format
doc_loader = DocumentLoader("company_employees.pdf", "https://modelcontextprotocol.io/docs/getting-started/intro")
pdf_text, web_text = doc_loader.load_all()
print("completed loading documents")

# Split the web text into chunks
text_splitter = GetTextSplitters(doc=pdf_text, chunk_size=40)
splitted_by_space = text_splitter.splitByCharacterTextSplitter()
print("completed splitting text into chunks")

# Generate embeddings for each chunk using EmbeddingManager
embedding_manager = EmbeddingManager()
vectors = embedding_manager.embed_documents(splitted_by_space)
# print("completed generating embeddings for chunks")

# Prepare data for Qdrant (id, vector, payload)

# Example meta-data: source, chunk_index, length
data = [
    {
        "id": idx,
        "vector": vec,
        "payload": {
            "text": text,
            "meta": {
                "source": "pdf_text",
                "chunk_index": idx,
                "length": len(text)
            }
        }
    }
    for idx, (vec, text) in enumerate(zip(vectors, splitted_by_space))
]

# Initialize the QdrantClient and upload the embeddings to Qdrant
qdrant_client = QdrantClientManager.initialize_client()
collection_name = "Employee_data"
QdrantClientManager.create_collection(collection_name, vector_size=len(vectors[0]))
QdrantClientManager.upload_data(collection_name, data)

# Example search query
query = "What do you know about httpstreamable MCP servers?, help me write a mcp server with fastapi , give me detailed answer"
query_vector = embedding_manager.embed_query(query)
search_results = QdrantClientManager.search(collection_name, query_vector, limit=5)

# print(f"Search results for query '{query}':")
# for result in search_results:
#     print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")

llm_service = LLMQueryService()
llm_response = llm_service.query_llm(query, search_results)
print("LLM Response:")
print(llm_response)


if __name__ == "__main__":
    print("Script executed successfully.")