"""Text generation with large language models."""

import chainlit as cl
from openai import AsyncOpenAI, OpenAIError

from rca_accelerator_chatbot.settings import ModelSettings, ThreadMessages
from rca_accelerator_chatbot.config import config
from rca_accelerator_chatbot.constants import (
    DOCS_PROFILE, RCA_FULL_PROFILE, CI_LOGS_PROFILE
)

# Initialize generative LLM client
gen_llm = AsyncOpenAI(
    base_url=config.generation_llm_api_url,
    organization='',
    api_key=config.generation_llm_api_key,
)


async def discover_generative_model_names() -> list[str]:
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


def _handle_context_size_limit(err: OpenAIError,
                               is_api: bool = False) -> str:
    if 'reduce the length of the messages or completion' in err.message:
        if not is_api :
            cl.user_session.set('message_history', '')
        return 'Request size with history exceeded limit, ' \
               'Please start a new thread.'
    return str(err)


async def get_response(user_message: ThreadMessages, # pylint: disable=too-many-arguments
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
            async for stream_resp in await gen_llm.chat.completions.create(
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
            response = await gen_llm.chat.completions.create(
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
        err_msg = _handle_context_size_limit(e, is_api)
        if not is_api:
            cl.logger.error("Error in process_message_and_get_response: %s",
                            err_msg)
        response_msg.content = (
            f"I encountered an error while generating a response: {err_msg}."
        )
    return is_error


def get_system_prompt_per_profile(profile_name: str) -> str:
    """Get the system prompt for the specified profile.

    Args:
        profile_name: The name of the profile for which to get the system prompt.
    Returns:
        The system prompt for the specified profile.
    """
    if profile_name == DOCS_PROFILE:
        return config.docs_system_prompt
    if profile_name in [CI_LOGS_PROFILE, RCA_FULL_PROFILE]:
        return config.ci_logs_system_prompt + config.jira_formatting_syntax_prompt
    return config.ci_logs_system_prompt
