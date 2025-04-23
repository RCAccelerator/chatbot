"""Text generation with large language models."""

import chainlit as cl
from openai import AsyncOpenAI, OpenAIError

from settings import HistorySettings, ModelSettings
from config import config

# Initialize generative LLM client
gen_llm = AsyncOpenAI(
    base_url=config.generation_llm_api_url,
    organization='',
    api_key=config.generation_llm_api_key,
)


async def discover_generative_model_names() -> str:
    """Discover available generative LLM models."""
    models = await gen_llm.models.list()
    return extract_model_ids(models)


def extract_model_ids(models) -> list[str]:
    """Extracts model IDs from the models list."""
    model_ids = []
    for model in models.data:
        model_ids.append(model.id)
    if not model_ids:
        cl.logger.error("No models available.")
        return []
    return model_ids


def _handle_context_size_limit(err: OpenAIError) -> str:
    if 'reduce the length of the messages or completion' in err.message:
        cl.user_session.set('message_history', '')
        return 'Request size with history exceeded limit, ' \
               'Please start a new thread.'
    return str(err)


async def get_response(history_settings: HistorySettings, # pylint: disable=too-many-arguments
                       user_message: cl.Message, response_msg: cl.Message,
                       model_settings: ModelSettings,
                       product_name: str,
                       stream_response: bool = True) -> bool:
    """Process the user's message and generate a response using the LLM.

    This function constructs the prompt from the LLM by compbining the system
    prompt with the user's input. It then sends the prepared message to the LLM
    for response generation.

    Args:
        user_message: The user's input message object.
        response_msg: The message object to populate with the LLM's
            generated response or an error message if something goes wrong.
        model_settings: A dictionary containing LLM configuration.
        stream_response: Indicates whether we want to stream the response or
            get the process in a single chunk.
    """
    is_error = True
    message_history = history_settings.get('message_history', [])
    system_prompt_with_product_name = config.system_prompt.replace(
        "PRODUCT_NAME", product_name
    )
    if not message_history:
        message_history = [
            {"role": "system", "content": system_prompt_with_product_name}]

    message_history.append({"role": "user", "content": user_message.content})
    try:
        if stream_response:
            async for stream_resp in await gen_llm.chat.completions.create(
                messages=message_history, stream=stream_response,
                **model_settings
            ):
                if stream_resp.choices and len(stream_resp.choices) > 0:
                    if token := stream_resp.choices[0].delta.content or "":
                        await response_msg.stream_token(token)
        else:
            response = await gen_llm.chat.completions.create(
                messages=message_history, stream=stream_response,
                **model_settings
            )
            response_msg.content = response.choices[0].message.content or ""
        message_history.append({"role": "assistant",
                                "content": response_msg.content})
        if history_settings.get('keep_history', True):
            cl.user_session.set('message_history', message_history)
        is_error = False
    except OpenAIError as e:
        err_msg = _handle_context_size_limit(e)

        cl.logger.error("Error in process_message_and_get_response: %s",
                        err_msg)
        response_msg.content = (
            f"I encountered an error while generating a response: {err_msg}."
        )
    return is_error
