"""Provider for the rerank model"""
import httpx

from rca_accelerator_chatbot.config import config
from rca_accelerator_chatbot.models.model import ModelProvider


class RerankModelProvider(ModelProvider):
    """Rerank model provider"""

    def __init__(self,
                 base_url: str = config.reranking_model_api_url,
                 api_key: str = config.reranking_model_api_key):
        super().__init__(base_url, api_key)

    async def get_rerank_score(
            self,
            query_text: str,
            search_content: str,
            model: str,
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

        Raises:
            HTTPStatusError: If the response from the /rerank API endpoint is
                not 200 status code.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # If the search_content is too big, we have to split it. We use half of the
        # reranking_model_max_content because we have to leave space for the user's
        # input.
        max_chunk_size = self.get_context_size(model) // 2
        sub_chunks = [
            search_content[i:i + max_chunk_size]
            for i in range(0, len(search_content), max_chunk_size)
        ]

        data = {
            "model": model,
            "query": query_text,
            "documents": sub_chunks,
        }
        rerank_url = f"{self.base_url}/rerank"
        async with httpx.AsyncClient() as client:
            response = await client.post(rerank_url, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                if len(response_data["results"]) == 0:
                    return .0
                return response_data["results"][0].get("relevance_score", .0)

            response.raise_for_status()

        return .0

rerank_model_provider = RerankModelProvider()
