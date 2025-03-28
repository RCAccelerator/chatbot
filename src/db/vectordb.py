"""Vector database client for RAG operations."""
import chainlit as cl
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException

from config import VECTORDB_URL, VECTORDB_PORT, VECTORDB_COLLECTION_NAME


class VectorDBClient:
    """Vector database client for similarity search."""

    def __init__(self):
        """Initialize the vector database client."""
        try:
            self.client = QdrantClient(VECTORDB_URL, port=VECTORDB_PORT)
            cl.logger.info("Successfully connected to Qdrant vector database")
        except ApiException as e:
            cl.logger.error("Failed to connect to Qdrant: %s", str(e))
            self.client = None

    def health_check(self):
        """
        Check if the vector database connection is healthy.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        if self.client is None:
            return False

        try:
            # Attempt to get collection info to verify connection
            self.client.get_collection(VECTORDB_COLLECTION_NAME)
            return True
        except ApiException:
            return False

    def search(self, embedding, top_n, similarity_threshold):
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
            cl.logger.error("Vector database client not available")
            return results

        try:
            search_results = self.client.search(
                collection_name=VECTORDB_COLLECTION_NAME,
                query_vector=embedding,
                limit=top_n
            )

            for res in search_results:
                if res.score >= similarity_threshold:
                    results.append({
                        "score": res.score,
                        "url": res.payload['url']
                    })
            return results
        except ApiException as e:
            cl.logger.error("Error in vector search: %s", str(e))
            return results


# Create singleton instance
vectordb_client = VectorDBClient()
