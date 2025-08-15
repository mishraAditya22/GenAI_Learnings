from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

class EmbeddingManager:
    def __init__(self):
        os.environ.pop("OPENAI_API_BASE", None)
        self.embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),      # Azure deployment name
            azure_endpoint=os.getenv("EMBEDDING_OPENAI_API_BASE"),  # instead of openai_api_base
            openai_api_key=os.getenv("EMBEDDING_OPENAI_API_KEY"),
            openai_api_version=os.getenv("EMBEDDING_OPENAI_API_VERSION"),
            chunk_size=1
        )

    def embed_documents(self, documents):
        return self.embeddings.embed_documents(documents)

    def embed_query(self, query):
        return self.embeddings.embed_query(query)