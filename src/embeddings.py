"""Embedding generation and vector search functionality."""
from urllib.parse import urlparse

import chainlit as cl
import httpx
from openai import AsyncOpenAI, OpenAIError
from qdrant_client.http.exceptions import ApiException

from config import config
from vectordb import vector_store

# Initialize embedding LLM client
emb_llm = AsyncOpenAI(
    base_url=config.embeddings_llm_api_url,
    organization='',
    api_key=config.embeddings_llm_api_key
)


async def generate_embedding(text, model_name=config.embeddings_model):
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
        search_string, model_name=config.embeddings_model,
        top_n=config.search_top_n,
        similarity_threshold=config.search_similarity_threshold):
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
        results = vector_store.search(embedding, top_n,
                                      similarity_threshold)
        return results
    except (ApiException, OpenAIError, ValueError, KeyError) as e:
        cl.logger.error("Error in search_similar_content: %s", str(e))
        return []


async def get_num_tokens(
        prompt: str, model=config.embeddings_model,
        llm_url=config.embeddings_llm_api_url,
        api_key=config.embeddings_llm_api_key) -> int:
    """Retrieve the number of tokens required to process the prompt.

    This function calls the /tokenize API endpoint to get the number of
    tokens the input will be transformed into when processed by the specified
    model (default is the embedding model).

    Args:
        prompt: The input text for which to calculate the token count.
        model: The model to use for tokenization.
        llm_url: The URL of the model.
        api_key: The API key used for authentication.

    Raises:
        HTTPStatusError: If the response from the /tokenize API endpoint is
            not 200 status code.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "prompt": prompt,
    }

    llm_url_parse = urlparse(llm_url)
    tokenize_url = f"{llm_url_parse.scheme}://{llm_url_parse.netloc}/tokenize"

    async with httpx.AsyncClient() as client:
        response = await client.post(tokenize_url, headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            return response_data["count"]

        response.raise_for_status()

    return 0
