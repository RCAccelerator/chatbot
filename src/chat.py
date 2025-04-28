"""Handler for chat messages and responses."""
from dataclasses import dataclass
import chainlit as cl
import httpx
from openai.types.chat import ChatCompletionMessageParam

from vectordb import vector_store
from generation import get_response
from embeddings import get_num_tokens, generate_embedding
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


async def perform_multi_collection_search(
    message_content: str,
    embeddings_model_name: str,
    similarity_threshold: float,
    collections: list[str],
    top_n: int = config.search_top_n
) -> list[dict]:
    """
    Search multiple collections using a generated embedding from message_content.
    Returns aggregated and sorted results from all collections.
    """
    embedding = await generate_embedding(message_content, embeddings_model_name)
    if embedding is None:
        return []
    all_results = []
    for collection in collections:
        results = vector_store.search(
            embedding, top_n, similarity_threshold, collection
        )
        for r in results:
            r['collection'] = collection
        all_results.extend(results)

    return sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)


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
        result = SEARCH_RESULTS_TEMPLATE.format(
            kind=res.get('kind', "NO VALUE"),
            text=res.get('text', "NO VALUE"),
            score=res.get('score', "NO VALUE"),
            components=components
        )

        # Append additional fields
        result += "\n".join(
            [
                f"{k}: {v}" for k, v in res.items()
                if k not in ['kind', 'text', 'score', 'components']
            ])
        result += "\n---"

        formatted_results.append(result)

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
        resp.content += "\n\nTop related knowledge:\n" + search_message


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
        search_results: list[dict],
        context: str) -> None:
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
    debug_content += f"\n```{context}```"
    cl.logger.debug(debug_content)
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

    try:
        if message.elements and message.elements[0].path:
            with open(message.elements[0].path, 'r', encoding='utf-8') as file:
                message.content += file.read()
    except OSError as e:
        cl.logger.error(e)
        resp.content = "An error occurred while processing your file."
        await resp.send()
        return

    search_content = _build_search_content_from_history(message_history) + message.content

    # Check message length
    is_valid_length, error_message = await check_message_length(
        search_content)
    if not is_valid_length:
        resp.content = error_message
        # Reset message history to let the user try again
        cl.user_session.set("message_history", [])
        await resp.send()
        return

    # Get collections from settings
    collections = [
        settings["jira_collection_name"],
        settings["errata_collection_name"],
        settings["documentation_collection_name"],
    ]

    if message.content:
        # Search all collections with the same embedding (embedding now generated inside)
        search_results = await perform_multi_collection_search(
            search_content,
            get_embeddings_model_name(),
            get_similarity_threshold(),
            collections
        )
        message.content += build_prompt(search_results)

        if debug_mode:
            await print_debug_content(settings, search_content,
                                      search_results, message.content)

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
            cl.user_session.get("chat_profile"),
            stream_response=settings.get("stream", True)
        )

        if not is_error:
            # Extend response with searched jira urls
            append_searched_urls(search_results, resp)

    update_msg_count()
    await resp.send()


async def handle_user_message_api( # pylint: disable=too-many-arguments
    message_content: str,
    similarity_threshold: float,
    generative_model_settings: ModelSettings,
    embeddings_model_settings: ModelSettings,
    vectordb_collections: list[str],
    product_name: str,
    ) -> str:
    """
    API handler for user messages without Chainlit context.
    """
    response = MockMessage(content="", urls=[])

    # Check message length
    is_valid_length, error_message = await check_message_length(message_content)
    if not is_valid_length:
        response.content = error_message
        return response

    # Perform search in all collections (embedding generated inside)
    search_results = await perform_multi_collection_search(
        message_content,
        embeddings_model_settings["model"],
        similarity_threshold=similarity_threshold,
        collections=vectordb_collections
    )

    message = MockMessage(content=message_content + build_prompt(search_results), urls=[])

    # Process user message and get AI response
    is_error = await get_response(
        {"keep_history": False}, message, response, generative_model_settings,
        product_name, stream_response=False
    )
    if not is_error:
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


def get_embeddings_model_name() -> str:
    """Get name of the embeddings model."""

    settings = cl.user_session.get("settings")

    return settings.get("embeddings_model", config.embeddings_model)
