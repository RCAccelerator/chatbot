"""Provider for the embedding model."""

import chainlit as cl
from openai import OpenAIError

from rca_accelerator_chatbot.config import config
from rca_accelerator_chatbot.models.model import ModelProvider


class EmbeddingsModelProvider(ModelProvider):
    """Embeddings model provider"""

    def __init__(self,
                 base_url: str = config.embeddings_llm_api_url,
                 api_key: str = config.embeddings_llm_api_key):
        super().__init__(base_url, api_key)

    async def generate_embedding(self, text: str, model_name: str) -> None | list[float]:
        """Generate embeddings for the given text using the specified model."""
        try:
            embedding_response = await self.llm.embeddings.create(
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

embed_model_provider = EmbeddingsModelProvider()
