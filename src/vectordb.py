"""Vector database client for RAG operations."""

from typing import List
import chainlit as cl
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException

from config import config


class VectorStore:
    """Abstract interface for vector storage and retrieval operations."""

    def search(
        self, embedding: List[float], top_n: int, similarity_threshold: float
    ) -> list:
        """
        Search for similar vectors in the database.

        Args:
            embedding: Vector embedding to search with
            top_n: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of search results with scores and metadata
        """
        raise NotImplementedError

    def health_check(self):
        """
        Check if the vector database connection is healthy.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore interface."""

    def __init__(self):
        """Initialize the vector database client."""
        self.client = None
        try:
            self.client = QdrantClient(
                config.vectordb_url,
                api_key=config.vectordb_api_key,
                port=config.vectordb_port,
            )
            if self.health_check():
                cl.logger.info("Successfully connected to Qdrant vector " + "database")
            else:
                cl.logger.error("Vector database client is not healthy")
        except ApiException as e:
            cl.logger.error("Failed to connect to Qdrant: %s", str(e))

    def health_check(self):
        """
        Check if the vector database connection is healthy.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        if self.client is None:
            return False

        # Attempt to get collection info to verify connection
        self.client.get_collection(config.vectordb_collection_name)
        return True

    def search(
        self, embedding: List[float], top_n: int, similarity_threshold: float
    ) -> list:
        """
        Search for similar vectors in the database.

        Args:
            embedding: Vector embedding to search with
            top_n: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of search results with scores and metadata
        """
        results = []
        if self.client is None:
            cl.logger.error("Vector database client is not available")
            return results

        try:
            search_results = self.client.search(
                collection_name=config.vectordb_collection_name,
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
