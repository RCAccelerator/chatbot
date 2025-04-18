"""Handler for chat messages and responses."""
from dataclasses import dataclass
import chainlit as cl
import httpx

from openai.types.chat import ChatCompletionMessageParam
from generation import get_response
from embeddings import search_similar_content, get_num_tokens
from settings import ModelSettings
from config import config
from constants import (
    SUGGESTED_MINIMUM_SIMILARITY_THRESHOLD,
    SEARCH_RESULTS_TEMPLATE,
    NO_RESULTS_FOUND
    )


# Create mock message and response objects
@dataclass
class MockMessage:
    """
    A dictionary type that defines a mock message.

    Attributes:
        content: The content of the message.
        urls: The list of Jira urls
    """
    content: str
    urls: list


async def perform_search(user_content: str,
                         model_name: str,
                         similarity_threshold: float,
                         collection_name: str = config.vectordb_collection_name) -> list[dict]:
    """
    Perform search inside of the vector DB to find information that might
    relate to the problem described by the user.

    Args:
        user_content: User's input query
        similarity_threshold: Minimum similarity score threshold

    Returns:
        List of unique search results sorted by relevance
    """

    # Search based on user query first
    search_results = await search_similar_content(
        search_string=user_content,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        collection_name=collection_name
    )

    unique_results: dict = {}
    for result in search_results:
        key = (result.get('url'), result.get('kind'))
        if key in unique_results:
            # Keep the result with higher score
            if result.get('score', 0) > unique_results[key].get('score', 0):
                unique_results[key] = result
        else:
            unique_results[key] = result
    return sorted(list(unique_results.values()),
                  key=lambda x: x.get('score', 0), reverse=True)


def build_prompt(search_results: list[dict]) -> str:
    """
    Generate a prompt based on the information we retrieved from the vector
    database.

    Args:
        search_results: A list of results obtained from the vector db

    Returns:
        Formatted string with search results
    """
    if not search_results:
        return config.prompt_header + NO_RESULTS_FOUND

    formatted_results = []

    for res in search_results:
        components = "NO VALUE"
        if res.get('components', []):
            components = ",".join([str(e) for e in res.get('components')])

        formatted_results.append(SEARCH_RESULTS_TEMPLATE.format(
            kind=res.get('kind', "NO VALUE"),
            text=res.get('text', "NO VALUE"),
            score=res.get('score', "NO VALUE"),
            components=components
        ))

    return config.prompt_header + "\n" + "\n".join(formatted_results)


def append_searched_urls(search_results, resp, urls_as_list=False):
    """
    Append search urls.

    Args:
        search_results: List of search results
        resp: The response message object to populate
        urls_as_list: Whether to return URLs as a list in `resp.urls`
        or as a string in `resp.content`.
    """
    search_message = ""
    deduped_urls: list = []

    # Deduplicate jira urls
    for result in search_results:
        url = result.get('url')
        if url not in deduped_urls:
            score = result.get('score', 0)
            search_message += f'🔗 {url}, Similarity Score: {score}\n'
            deduped_urls.append(url)
    if urls_as_list and deduped_urls:
        if hasattr(resp, 'urls'):
            resp.urls = deduped_urls
    elif search_message:
        resp.content += "\n\nTop similar bugs:\n" + search_message


def update_msg_count():
    """Update the number of messages in the conversation."""
    counter = cl.user_session.get("counter", 0)
    counter += 1
    cl.user_session.set("counter", counter)


async def check_message_length(message_content: str) -> tuple[bool, str]:
    """
    Check if the message content exceeds the token limit.

    Args:
        message_content: The content to check

    Returns:
        A tuple containing:
        - bool: True if the message is within length limits, False otherwise
        - str: Error message if the length check fails, empty string otherwise
    """
    try:
        num_required_tokens = await get_num_tokens(message_content)
    except httpx.HTTPStatusError as e:
        cl.logger.error(e)
        return False, "We've encountered an issue. Please try again later ..."

    if num_required_tokens > config.embeddings_llm_max_context:
        # On average, a single token corresponds to approximately 4 characters.
        # Because logs often require more tokens to process, we estimate 3
        # characters per token.
        approx_max_chars = round(
            config.embeddings_llm_max_context * 3, -2)

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


async def print_debug_content(
        settings: dict,
        search_content: str,
        search_results: list[dict]) -> None:
    """Print debug content if user requested it.

    Args:
        settings: The settings user provided through the UI.
        search_results: The results we obtained from the vector database.
    """
    # Initialize debug_content with all settings
    debug_content = ""
    if settings:
        debug_content = "#### Current Settings:\n"
        for key, value in settings.items():
            debug_content += f"- {key}: {value}\n"
    debug_content += "\n\n"

    # Display the search content
    debug_content += (
        f"#### Search Content:\n"
        f"```\n"
        f"{search_content}\n"
        f"```\n"
    )

    # Display the number of tokens in the search content
    num_t = await get_num_tokens(search_content)
    debug_content += f"**Number of tokens in search content:** {num_t}\n\n"

    # Display vector DB debug information if debug mode is enabled
    if search_results:
        debug_content += "#### Vector DB Search Results:\n"
        for i, result in enumerate(search_results[:config.search_top_n], 1):
            debug_content += (
                f"**Result {i}**\n"
                f"- Score: {result.get('score', 0)}\n"
                f"- URL: {result.get('url', 'N/A')}\n\n"
                f"Preview:\n"
                f"```\n"
                f"{result.get('text', 'N/A')[:500]} ...\n"
                f"```\n\n"
            )
    await cl.Message(content=debug_content, author="debug").send()


def _build_search_content_from_history(
        message_history: list[ChatCompletionMessageParam]) -> str:
    previous_message_content = ""
    if message_history:
        for message in message_history:
            if message['role'] == 'user':
                previous_message_content += f"\n{message['content']}"
    return previous_message_content


def _filter_debug_messages(
        message_history: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
    """Remove all debug messages from history.
    """
    if message_history:
        message_history = [
            message for message in message_history
            if message.get("name", "system") != "debug"]

    return message_history


async def handle_user_message(message: cl.Message, debug_mode=False):
    """
    Main handler for user messages.

    Args:
        message: The user's input message
        debug_mode: Whether to show debug information
    """
    settings = cl.user_session.get("settings")
    resp = cl.Message(content="")

    message_history = cl.user_session.get('message_history')
    message_history = _filter_debug_messages(message_history)

    previous_messages = _build_search_content_from_history(message_history)
    search_content = previous_messages + message.content

    try:
        if message.elements and message.elements[0].path:
            with open(message.elements[0].path, 'r', encoding='utf-8') as file:
                search_content += file.read()
    except OSError as e:
        cl.logger.error(e)
        resp.content = "An error occurred while processing your file."
        await resp.send()
        return

    # Check message length
    is_valid_length, error_message = await check_message_length(
        search_content)
    if not is_valid_length:
        resp.content = error_message
        # Reset message history to let the user try again
        cl.user_session.set("message_history", [])
        await resp.send()
        return

    if message.content:
        search_results = await perform_search(user_content=search_content,
                                              model_name=get_embeddings_model_name(),
                                              similarity_threshold=get_similarity_threshold(),
                                              collection_name=get_collection_name())
        if debug_mode:
            await print_debug_content(settings, search_content,
                                      search_results)

        message.content += build_prompt(search_results)
        # Process user message and get AI response
        is_error = await get_response(
            {
                "message_history": message_history,
                "keep_history": settings.get("keep_history", True)
            },
            message,
            resp,
            {
                "model": settings["generative_model"],
                "max_tokens": settings["max_tokens"],
                "temperature": settings["temperature"]
            },
            stream_response=settings.get("stream", True)
        )

        if not is_error:
            # Extend response with searched jira urls
            append_searched_urls(search_results, resp)

    update_msg_count()
    await resp.send()


async def handle_user_message_api(
    message_content: str,
    similarity_threshold: float,
    generative_model_settings: ModelSettings,
    embeddings_model_settings: ModelSettings,
    vectordb_collection: str
    ) -> str:
    """
    API handler for user messages without Chainlit context.

    Args:
        message_content: The user's input message as a string

    Returns:
        The AI generated response as a string
    """
    response = MockMessage(content="", urls=[])

    # Check message length
    is_valid_length, error_message = await check_message_length(message_content)
    if not is_valid_length:
        response.content = error_message
        return response

    # Perform search and build prompt
    search_results = await perform_search(
        user_content=message_content,
        model_name=embeddings_model_settings["model"],
        similarity_threshold=similarity_threshold,
        collection_name=vectordb_collection
    )

    message = MockMessage(content=message_content + build_prompt(search_results), urls=[])

     # Process user message and get AI response
    is_error = await get_response(
        {"keep_history": False}, message, response, generative_model_settings,
        stream_response=False,
    )

    if not is_error:
        # Extend response with searched jira urls
        append_searched_urls(search_results, response, urls_as_list=True)

    return response


def get_similarity_threshold() -> float:
    """
    Get the similarity threshold from user settings or default config.

    Returns:
        Similarity threshold value
    """
    settings = cl.user_session.get("settings")
    if not settings:
        return config.search_similarity_threshold
    # Get threshold from settings or fall back to config default
    threshold = settings.get("search_similarity_threshold",
                             config.search_similarity_threshold)

    # Ensure threshold is within valid range
    if threshold < SUGGESTED_MINIMUM_SIMILARITY_THRESHOLD:
        # If default config is also below minimum, use it anyway
        if (config.search_similarity_threshold <
                SUGGESTED_MINIMUM_SIMILARITY_THRESHOLD):
            return config.search_similarity_threshold
        return SUGGESTED_MINIMUM_SIMILARITY_THRESHOLD

    # If threshold is above 1, cap it at 1
    return min(threshold, 1.0)

def get_collection_name() -> str:
    """Get name of database collection for retrieval."""

    settings = cl.user_session.get("settings")

    return settings.get("collection_name", config.vectordb_collection_name)

def get_embeddings_model_name() -> str:
    """Get name of the embeddings model."""

    settings = cl.user_session.get("settings")

    return settings.get("embeddings_model", config.embeddings_model)
