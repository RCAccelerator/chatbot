"""A module responsible for building the prompt for the generative model."""
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
)

from config import config
from constants import (
    NO_RESULTS_FOUND, SEARCH_RESULTS_TEMPLATE, SEARCH_RESULT_TRUNCATED_CHUNK
)
from settings import HistorySettings, ThreadMessages
from generation import get_system_prompt_per_profile


def search_result_to_str(search_result: dict) -> str:
    """Convert a search result to a string."""
    components = "NO VALUE"
    if search_result.get('components', []):
        components = ",".join([str(e) for e in search_result.get('components')])

    search_result_chunk = SEARCH_RESULTS_TEMPLATE.format(
        kind=search_result.get('kind', "NO VALUE"),
        text=search_result.get('text', "NO VALUE"),
        score=search_result.get('score', "NO VALUE"),
        components=components,
    )

    search_result_chunk += "\n".join(
        [
            f"{k}: {v}" for k, v in search_result.items()
            if k not in ['kind', 'text', 'score', 'components']
        ])
    search_result_chunk += "\n---\n"

    return search_result_chunk

# pylint: disable=R0914
async def build_prompt(
        search_results: list[dict],
        user_message: str,
        profile_name: str,
        history_settings: HistorySettings,
) -> (bool, ThreadMessages):
    """Generate a full prompt that gets sent to the generative model.

    The full prompt consists out of these three parts:
        1. System prompt
        2. User message
        3. Text representation of the data retrieved from vector database

    The sections #2 and #2 may repeat if history is enabled. If for whatever
    reason the full prompt exceeds the maximum context length of the generative
    model (set in config.py), the new part #2 of the system prompt is truncated.
    The user message is always appended to the full prompt in its full length.

    Args:
        search_results: A list of results obtained from the vector db
        user_message: The user's message content
        history_settings: Settings for the message history
        profile_name: The name of the profile for which to generate the prompt

    Returns:
        List of messages that make up the full prompt. Each return value contains
        at least one system prompt and user message. And indication whether an
        error occurred during the process of building of the prompt.
    """
    # 1. Build system prompt if it is not part of the history already
    is_error = False
    full_prompt_len = 0
    full_prompt: ThreadMessages = []
    message_history = history_settings.get('message_history', [])

    if not message_history:
        full_prompt.append(ChatCompletionSystemMessageParam(
            role="system",
            content=get_system_prompt_per_profile(profile_name),
        ))
    else:
        full_prompt = message_history

    full_prompt_len += len(str(full_prompt))

    # NOTE: On average, a single token corresponds to approximately 4 characters.
    # Because logs often require more tokens to process, we estimate 3
    # characters per token. Also, we do not want to use the full context
    # of the model as trying to use the full context of the model might lead
    # to decreased performance (0.75 constant).
    approx_max_chars = config.generative_model_max_context * 3 * 0.75

    # If no information was retrieved from the vector database, end the generation
    # of the prompt.
    if not search_results:
        full_prompt.append(ChatCompletionUserMessageParam(
            role="user",
            content=config.prompt_header + NO_RESULTS_FOUND + "\n" + user_message,
        ))
        return is_error, full_prompt


    full_user_message = config.prompt_header
    full_prompt_len += len(full_user_message)

    # 2. Add a user's message into the prompt
    full_user_message += "\n" + user_message


    # 3. Add search results into the conversations
    for res in search_results:
        search_result_chunk = search_result_to_str(res)

        # If there is not enough space, for search result truncate it and finish
        # the generation of the prompt.
        current_prompt_len = (
            full_prompt_len + len(search_result_chunk) + len(user_message)
        )

        if current_prompt_len > approx_max_chars:
            # Calculate how many characters we have to remove from the search
            # result
            trim_len = int(current_prompt_len - approx_max_chars)
            truncated_search_result = SEARCH_RESULT_TRUNCATED_CHUNK.format(
                text=search_result_chunk[:-trim_len]
            )

            full_user_message += "\n" + truncated_search_result
            full_prompt_len += len(truncated_search_result)

            is_error = True
            break

        full_user_message += search_result_chunk
        full_prompt_len += len(search_result_chunk)

    full_prompt.append(ChatCompletionUserMessageParam(
        role="user",
        content=full_user_message,
    ))

    return is_error, full_prompt
