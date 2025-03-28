"""Embedding generation and vector search functionality."""
import chainlit as cl
from openai import AsyncOpenAI, OpenAIError
from qdrant_client.http.exceptions import ApiException

from config import (
    EMB_LLM_API_URL, EMB_LLM_API_KEY, EMBEDDINGS_MODEL,
    SEARCH_INSTRUCTION, SEARCH_TOP_N, SEARCH_SIMILARITY_THRESHOLD
)
from db.vectordb import vectordb_client

# Initialize embedding LLM client
emb_llm = AsyncOpenAI(
    base_url=EMB_LLM_API_URL,
    organization='',
    api_key=EMB_LLM_API_KEY
)


async def generate_embedding(text, model_name=EMBEDDINGS_MODEL):
    """Generate embeddings for the given text using the specified model."""
    try:
        embedding_response = await emb_llm.embeddings.create(
            model=model_name,
            input=text,
            encoding_format='float'
        )

        if not embedding_response:
            cl.logger.error("Failed to get embeddings: " +
                            "No response from model %s", model_name)
            return None
        if not embedding_response.data or len(embedding_response.data) == 0:
            cl.logger.error("Failed to get embeddings: " +
                            "Empty response for model %s", model_name)
            return None

        return embedding_response.data[0].embedding
    except OpenAIError as e:
        cl.logger.error("Error generating embeddings: %s", str(e))
        return None


async def search_similar_content(
        search_string, model_name=EMBEDDINGS_MODEL,
        top_n=SEARCH_TOP_N,
        similarity_threshold=SEARCH_SIMILARITY_THRESHOLD):
    """
    Search for similar content in the vector database.

    Args:
        search_string: The query to search for
        model_name: The model to use for creating embeddings
        top_n: Maximum number of results to return
        similarity_threshold: Minimum similarity score threshold

    Returns:
        List of search results with text and metadata
    """
    try:
        embedding = await generate_embedding(search_string, model_name)
        if embedding is None:
            return []

        # Search vector database using the embedding
        results = vectordb_client.search(embedding, top_n,
                                         similarity_threshold)
        return results
    except (ApiException, OpenAIError, ValueError, KeyError) as e:
        cl.logger.error("Error in search_similar_content: %s", str(e))
        return []


async def db_lookup(search_string, model_name=EMBEDDINGS_MODEL,
                    search_top_n=SEARCH_TOP_N,
                    search_sensitive=SEARCH_SIMILARITY_THRESHOLD):
    """
    Legacy compatibility wrapper for search_similar_content.
    Search the vector database for relevant content based on the input query.
    """
    search_query = SEARCH_INSTRUCTION + search_string
    return await search_similar_content(
        search_query, model_name, search_top_n, search_sensitive
    )
