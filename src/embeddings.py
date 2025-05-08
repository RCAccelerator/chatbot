"""Embedding generation and vector search functionality."""

from typing import List
from urllib.parse import urlparse

import chainlit as cl
import httpx
from openai import AsyncOpenAI, OpenAIError

from config import config
from generation import extract_model_ids

# Initialize embedding LLM client
emb_llm = AsyncOpenAI(
    base_url=config.embeddings_llm_api_url,
    organization="",
    api_key=config.embeddings_llm_api_key,
)

async def discover_embeddings_model_names() -> List[str]:
    """Discover available embedding LLM models."""
    models = await emb_llm.models.list()
    return extract_model_ids(models)

async def get_default_embeddings_model_name() -> str:
    """Get name of the default embeddings model."""
    models = await discover_embeddings_model_names()
    return models[0]

async def generate_embedding(
    text: str, model_name: str
) -> None | List[float]:
    """Generate embeddings for the given text using the specified model."""
    try:
        embedding_response = await emb_llm.embeddings.create(
            model=model_name, input=text, encoding_format="float"
        )

        if not embedding_response:
            cl.logger.error(
                "Failed to get embeddings: " + "No response from model %s", model_name
            )
            return None
        if not embedding_response.data or len(embedding_response.data) == 0:
            cl.logger.error(
                "Failed to get embeddings: " + "Empty response for model %s", model_name
            )
            return None

        return embedding_response.data[0].embedding
    except OpenAIError as e:
        cl.logger.error("Error generating embeddings: %s", str(e))
        return None


async def get_num_tokens(
    prompt: str,
    model: str,
    llm_url: str = config.embeddings_llm_api_url,
    api_key: str = config.embeddings_llm_api_key,
) -> int:
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
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

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

async def get_rerank_score(
        query_text: str,
        search_content: str,
        model: str = config.reranking_model_name,
        reranking_model_url: str = config.reranking_model_api_url,
        reranking_model_api_key: str = config.reranking_model_api_key,
) -> float:
    """Contact a re-rank model and get a more precise re-ranking score for the search content.

    This function calls the /rerank API endpoint to calculate a new more accurate
    score for the search content. First it chunks the search content to fit
    the context of the re-rank model, and then it calculates the score for each
    such a chunk. The final score is the maximum re-rank score out of all the
    chunks.

    Args:
        query_text: query text that the search content should be related to.
        search_content: Is a chunk retrieved from the vector database.
        model: Name of the model to use for re-ranking.
        reranking_model_url: URL of the re-rank model.
        reranking_model_api_key: API key for the re-rank model.

    Raises:
        HTTPStatusError: If the response from the /rerank API endpoint is
            not 200 status code.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {reranking_model_api_key}"
    }

    # If the search_content is too big, we have to split it. We use half of the
    # reranking_model_max_content because we have to leave space for the user's
    # input.
    max_chunk_size = config.reranking_model_max_context // 2
    sub_chunks = [
        search_content[i:i + max_chunk_size]
        for i in range(0, len(search_content), max_chunk_size)
    ]

    data = {
        "model": model,
        "query": query_text,
        "documents": sub_chunks,
    }
    rerank_url = f"{reranking_model_url}/rerank"
    async with httpx.AsyncClient() as client:
        response = await client.post(rerank_url, headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if len(response_data["results"]) == 0:
                return .0
            return response_data["results"][0].get("relevance_score", .0)

        response.raise_for_status()

    return .0
