"""Provider for the generative model."""
import chainlit as cl
from openai import OpenAIError

from rca_accelerator_chatbot.settings import ModelSettings, ThreadMessages
from rca_accelerator_chatbot.config import config
from rca_accelerator_chatbot.models.model import ModelProvider


class GenerativeModelProvider(ModelProvider):
    """Generative model provider"""

    def __init__(self,
                 base_url: str = config.generation_llm_api_url,
                 api_key: str = config.generation_llm_api_key):
        super().__init__(base_url, api_key)

    # pylint: disable=too-many-arguments
    async def get_response(self,
                           user_message: ThreadMessages,
                           response_msg: cl.Message,
                           model_settings: ModelSettings,
                           is_api: bool = False,
                           stream_response: bool = True,
                           step: cl.Step = None) -> bool:
        """Send a user's message and generate a response using the LLM.

        Args:
            user_message: The user's input message object.
            response_msg: The message object to populate with the LLM's
                generated response or an error message if something goes wrong.
            model_settings: A dictionary containing LLM configuration.
            stream_response: Indicates whether we want to stream the response or
                get the process in a single chunk.
            is_api: Indicates whether the function is called from the API or not.
            step: Optional step object to stream reasoning content to.

        Returns:
            bool indicating whether the function was successful or not.
        """
        is_error = True

        try:
            if stream_response:
                async for stream_resp in await self.llm.chat.completions.create(
                    messages=user_message, stream=stream_response,
                    **model_settings
                ):
                    if stream_resp.choices and len(stream_resp.choices) > 0:
                        delta = stream_resp.choices[0].delta

                        # Stream content to the response message
                        if token := delta.content or "":
                            await response_msg.stream_token(token)

                        # Stream reasoning content to the step if it exists
                        if step and hasattr(delta, "reasoning_content") and delta.reasoning_content:
                            await step.stream_token(delta.reasoning_content)
            else:
                response = await self.llm.chat.completions.create(
                    messages=user_message, stream=stream_response,
                    **model_settings
                )
                response_msg.content = response.choices[0].message.content or ""

                # If we have a step and reasoning content, update the step output
                message = response.choices[0].message
                if (step and hasattr(message, "reasoning_content")
                        and message.reasoning_content):
                    step.output = response.choices[0].message.reasoning_content

            is_error = False
        except OpenAIError as e:
            err_msg = GenerativeModelProvider._handle_context_size_limit(e, is_api)
            if not is_api:
                cl.logger.error("Error in process_message_and_get_response: %s",
                                err_msg)
            response_msg.content = (
                f"I encountered an error while generating a response: {err_msg}."
            )
        return is_error

    @staticmethod
    def _handle_context_size_limit(err: OpenAIError,
                                   is_api: bool = False) -> str:
        if 'reduce the length of the messages or completion' in err.message:
            if not is_api :
                cl.user_session.set('message_history', '')
            return 'Request size with history exceeded limit, ' \
                   'Please start a new thread.'
        return str(err)

gen_model_provider = GenerativeModelProvider()
