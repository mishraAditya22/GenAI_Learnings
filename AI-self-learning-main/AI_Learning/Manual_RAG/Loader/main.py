import sys
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class QdrantClientManager:
    _client = None

    @classmethod
    def initialize_client(cls, host='localhost', port=6333):
        if cls._client is None:
            cls._client = QdrantClient(host=host, port=port)
        return cls._client

    @classmethod
    def create_collection(cls, collection_name, vector_size=1536):
        existing_collections = [c.name for c in cls._client.get_collections().collections]
        if collection_name not in existing_collections:
            print(f"Creating collection: {collection_name}")
            cls._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        else:
            print(f"Collection {collection_name} already exists.")

    
    @classmethod
    def upload_data(cls, collection_name, data):
        """
        Uploads data (list of dicts or points) to the specified Qdrant collection.
        Args:
            collection_name (str): Name of the collection.
            data (list): List of points to upload. Each point should be a dict with 'id', 'vector', and optionally 'payload'.
        """
        client = cls._client
        if client is None:
            client = cls.initialize_client()
        try:
            client.upsert(
                collection_name=collection_name,
                points=data
            )
            print(f"Uploaded {len(data)} points to collection '{collection_name}'.")
        except Exception as e:
            print(f"Error uploading data: {e}")

    @classmethod
    def search(cls, collection_name, query_vector, limit=10):
        """
        Searches for the nearest neighbors of a query vector in the specified collection.
        Args:
            collection_name (str): Name of the collection.
            query_vector (list): The vector to search for.
            limit (int): Number of nearest neighbors to return.
        Returns:
            List of points that are the nearest neighbors.
        """
        client = cls._client
        if client is None:
            client = cls.initialize_client()
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results