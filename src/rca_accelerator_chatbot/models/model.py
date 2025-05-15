"""Base class for the model providers"""
import time
from functools import cached_property
from urllib.parse import urlparse

from openai import AsyncOpenAI
import httpx

from rca_accelerator_chatbot.config import config


class ModelProvider:
    """Base class for the model providers

    It is required to call init() after the class is initialized to ensure that
    all the fields are populated and all functionality is available.
    """

    def __init__(self, base_url: str, api_key: str, cache_timeout: int = 60):
        self.base_url = base_url
        self.api_key = api_key
        self.llm = AsyncOpenAI(
            base_url=self.base_url,
            organization='',
            api_key=self.api_key,
        )

        self._api_response = None
        self._api_response_time = 0
        self.cache_timeout = cache_timeout

    async def init(self) -> None:
        """Initialize the object with the data from the /models api response

        This function must be called after the object is instantiated to ensure
        that all the functionality is unlocked. It caches the response from /models
        api response.
        """
        cache_expired = time.time() > self._api_response_time + self.cache_timeout
        if self._api_response is not None and not cache_expired:
            return

        # List available models at the API endpoint
        models_page = await self.llm.models.list()
        self._api_response_time = time.time()
        self._api_response = [model.model_dump() for model in models_page.data]
        if not self._api_response:
            raise RuntimeError(f"No model discovered at {self.base_url}")

    @cached_property
    def all_model_names(self) -> list[str]:
        """Return all models available at the base_url endpoint."""
        if self._api_response is None:
            raise RuntimeError(f"{__class__} was not initialized - init().")

        return [response["id"] for response in self._api_response]

    def get_context_size(self, model_name: str) -> int:
        """Return max context size of the model"""
        if self._api_response is None:
            raise RuntimeError(f"{__class__} was not initialized - init().")

        if model_name not in self.all_model_names:
            raise RuntimeError(f"{model_name} model is not available at {self.base_url}")

        for response in self._api_response:
            if response["id"] == model_name:
                return response["max_model_len"]

        return 0

    async def get_num_tokens(self, prompt: str, model_name: str) -> int:
        """Retrieve the number of tokens required to process the prompt.

        This function calls the /tokenize API endpoint to get the number of
        tokens the input will be transformed into when processed by the specified
        model (default is the embedding model).

        Args:
            prompt: The input text for which to calculate the token count.
            model_name: The name of the model whose tokenizer we will use

        Raises:
            HTTPStatusError: If the response from the /tokenize API endpoint is
                not 200 status code.
        """
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        data = {
            "model": model_name,
            "prompt": prompt,
        }

        llm_url_parse = urlparse(self.base_url)
        tokenize_url = f"{llm_url_parse.scheme}://{llm_url_parse.netloc}/tokenize"

        async with httpx.AsyncClient() as client:
            response = await client.post(tokenize_url, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                return response_data["count"]

            response.raise_for_status()

        return 0

    async def check_message_length(self, message_content: str, model_name: str) -> tuple[bool, str]:
        """Check if the message content exceeds the token limit.

        Args:
            message_content: The content to check
            model_name: The name of the model that should be used for tokenization

        Returns:
            A tuple containing:
            - bool: True if the message is within length limits, False otherwise
            - str: Error message if the length check fails, empty string otherwise
        """
        try:
            num_required_tokens = await self.get_num_tokens(message_content, model_name)
        except httpx.HTTPStatusError:
            return False, "We've encountered an issue. Please try again later ..."

        embed_model_max_context = self.get_context_size(model_name)
        if num_required_tokens > embed_model_max_context:
            # Calculate the maximum character limit estimation for the embedding model.
            approx_max_chars = round(
                embed_model_max_context * config.chars_per_token_estimation, -2)

            error_message = (
                "⚠️ **Your input is too lengthy!**\n We can process inputs of up "
                f"to approximately {approx_max_chars} characters. The exact limit "
                "may vary depending on the input type. For instance, plain text "
                "inputs can be longer compared to logs or structured data "
                "containing special characters (e.g., `[`, `]`, `:`, etc.).\n\n"
                "To proceed, please:\n"
                "  - Focus on including only the most relevant details, and\n"
                "  - Shorten your input if possible."
                " \n\n"
                "To let you continue, we will reset the conversation history.\n"
                "Please start over with a shorter input."
            )
            return False, error_message

        return True, ""
