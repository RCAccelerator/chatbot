"""Vector database client for RAG operations."""

from typing import List
import chainlit as cl
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException

from rca_accelerator_chatbot.config import config


class VectorStore:
    """Abstract interface for vector storage and retrieval operations."""

    def search(
        self, embedding: List[float], similarity_threshold: float,
        collection_name: str, top_n: int = config.search_top_n,
    ) -> list:
        """
        Search for similar vectors in the database.

        Args:
            embedding: Vector embedding to search with
            similarity_threshold: Minimum similarity score threshold
            collection_name: Name of the collection to search in
            top_n: The maximum number of results to return.

        Returns:
            List of search results with scores and metadata
        """
        raise NotImplementedError

    def get_collections(self) -> list[str]:
        """
        Fetches collection names from Qdrant and adds default collections
        from the configuration if they exist.

        Returns:
            List of collection names
        """
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore interface."""

    def __init__(self):
        """Initialize the vector database client."""
        self.client = QdrantClient(
            config.vectordb_url,
            api_key=config.vectordb_api_key,
            port=config.vectordb_port,
        )
        cl.logger.info("Qdrant client initialized successfully.")

    def get_collections(self) -> list[str]:
        """
        Fetches collection names from Qdrant.

        Returns:
            List of collection names
        """
        collections = []
        try:
            qdrant_collections = self.client.get_collections().collections
            # Add fetched collections, avoiding duplicates
            for col in qdrant_collections:
                if col.name not in collections:
                    collections.append(col.name)
        except ApiException as e:
            cl.logger.error("Failed to connect to Qdrant to list collections: %s", str(e))
        if not collections:
            cl.logger.error("No collections found in Qdrant.")
        return collections

    def search(
        self, embedding: List[float], similarity_threshold: float,
        collection_name: str, top_n: int = config.search_top_n,
    ) -> list:
        """
        Search for similar vectors in the database.

        Args:
            embedding: Vector embedding to search with
            similarity_threshold: Minimum similarity score threshold
            collection_name: Name of the collection to search in
            top_n: The maximum number of results to return.

        Returns:
            List of search results with scores and metadata
        """
        results = []
        if self.client is None:
            cl.logger.error("Vector database client is not available")
            return results

        try:
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=top_n,
            )

            for res in search_results:
                if res.score >= similarity_threshold:
                    results.append(
                        {
                            "score": res.score,
                            "url": res.payload["url"],
                            "kind": res.payload["kind"],
                            "text": res.payload["text"],
                            "components": res.payload["components"],
                        }
                    )

            return results
        except ApiException as e:
            cl.logger.error("Error in vector search: %s", str(e))
            return results


# Create singleton instance
vector_store = QdrantVectorStore()
