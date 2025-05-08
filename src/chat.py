"""Handler for chat messages and responses."""
from dataclasses import dataclass
import chainlit as cl
from chainlit.context import ChainlitContextException
import httpx
from openai.types.chat import ChatCompletionAssistantMessageParam

import constants
from prompt import build_prompt
from vectordb import vector_store
from generation import get_response
from embeddings import (
    get_num_tokens, generate_embedding,
    get_rerank_score, get_default_embeddings_model_name
)
from settings import ModelSettings, HistorySettings, ThreadMessages
from config import config
from constants import (
    DOCS_PROFILE,
    RCA_FULL_PROFILE,
    TEXT_UPLOAD_TEMPLATE,
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
    settings: dict,
) -> list[dict]:
    """Search multiple collections using a generated embedding from message_content.

    The function first queries all the collections from the vector database and
    then sorts them based on the rerank score. The function returns the top n
    results.

    Args:
         message_content: The content of the user's message.
         embeddings_model_name: The name of the embedding model to use.
         similarity_threshold: The similarity threshold to use.
         collections: A list of collections to search.
         settings: The settings user provided through the UI.
    """
    embedding = await generate_embedding(message_content, embeddings_model_name)
    if embedding is None:
        return []

    all_results = []
    for collection in collections:
        results = vector_store.search(
            embedding, similarity_threshold, collection
        )

        for r in results:
            r['collection'] = collection
            if settings['enable_rerank']:
                r['rerank_score'] = await get_rerank_score(message_content, r['text'])
            else:
                r['rerank_score'] = None

        all_results.extend(results)

    sort_key = 'rerank_score' if settings['enable_rerank'] else 'score'
    sorted_results = sorted(all_results, key=lambda x: x.get(sort_key, 0), reverse=True)

    rerank_top_n = settings.get('rerank_top_n', config.rerank_top_n)
    if not isinstance(rerank_top_n, int):
        rerank_top_n = config.rerank_top_n

    return sorted_results[:rerank_top_n]


def append_searched_urls(search_results, resp, enable_rerank, urls_as_list=False):
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
            score_key = "score" if not enable_rerank else "rerank_score"
            rerank_score = result.get(score_key, 0)
            search_message += f'🔗 {url}, (_Similarity Score_: {rerank_score})\n'
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
        num_required_tokens = await get_num_tokens(message_content,
                                                   await get_embeddings_model_name())
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
        message_content: ThreadMessages) -> None:
    """Print debug content if the user requested it.

    Args:
        settings: The settings user provided through the UI.
        search_results: The results we got from the vector database.
        search_content: The content we used to search the vector database.
        message_content: The content of the user's message.
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
    num_t = await get_num_tokens(search_content,
                                 await get_embeddings_model_name())
    debug_content += f"**Number of tokens in search content:** {num_t}\n\n"

    # Display vector DB debug information if debug mode is enabled
    if search_results:
        debug_content += "#### Vector DB Search Results:\n"
        for i, result in enumerate(search_results, 1):
            score_key = "score" if not settings.get("enable_rerank") else "rerank_score"
            debug_content += (
                f"**Result {i}**\n"
                f"- Cosine similarity: {result.get('score', 0)}\n"
                f"- Rerank score: {result.get(score_key, 0)}\n"
                f"- URL: {result.get('url', 'N/A')}\n\n"
                f"Preview:\n"
                f"```\n"
                f"{result.get('text', 'N/A')[:500]} ...\n"
                f"```\n\n"
            )
    # Escaping markdown
    debug_content += f"\n#### Full user message:\n```\n{str(message_content)}\n```"

    cl.logger.debug(debug_content)
    async with cl.Step(name="debug") as debug_step:
        debug_step.output = debug_content


def _build_search_content(message_history: ThreadMessages,
                          current_message: str,
                          settings: dict) -> str:
    """Build search text from the history and current message."""
    previous_message_content = ""
    if message_history:
        for message in message_history:
            if message['role'] == 'user':
                previous_message_content += f"\n{message['content']}"

    # Limit the size of the message content as this is passed as query to
    # the reranking model. This is brute truncation, but can be improved
    # when we handle message history better.
    if settings['enable_rerank']:
        max_text_len_from_history = (
        config.reranking_model_max_context // 2 - len(
            current_message)) - 1
        return previous_message_content[
            :max_text_len_from_history] + current_message
    return previous_message_content + current_message


async def handle_user_message( # pylint: disable=too-many-locals,too-many-statements
    message: cl.Message,
    debug_mode=False):
    """
    Main handler for user messages.

    Args:
        message: The user's input message
        debug_mode: Whether to show debug information
    """
    settings = cl.user_session.get("settings")
    resp = cl.Message(content="")

    message_history = cl.user_session.get('message_history')

    try:
        if message.elements and message.elements[0].path:
            with open(message.elements[0].path, 'r', encoding='utf-8') as file:
                message.content += TEXT_UPLOAD_TEMPLATE.format(text=file.read())
    except OSError as e:
        cl.logger.error(e)
        resp.content = "An error occurred while processing your file."
        await resp.send()
        return

    search_content = _build_search_content(message_history,
                                           message.content,
                                           settings)

    # Check message length
    is_valid_length, error_message = await check_message_length(
        search_content)
    if not is_valid_length:
        resp.content = error_message
        # Reset message history to let the user try again
        cl.user_session.set("message_history", [])
        await resp.send()
        return

    collections = get_collections_per_profile(
        cl.user_session.get("chat_profile")
    )
    error_message = check_collections(collections)
    if error_message:
        resp.content = error_message
        await resp.send()
        return

    if message.content:
        async with cl.Step(name="searching") as search_step:
            search_step.output = "Searching for relevant information in our knowledge base..."
            # Search all collections with the same embedding (embedding now generated inside)
            try:
                search_results = await perform_multi_collection_search(
                    search_content,
                    await get_embeddings_model_name(),
                    get_similarity_threshold(),
                    collections,
                    settings,
                )
            except httpx.HTTPStatusError as e:
                cl.logger.error(e)
                resp.content = "An error occurred while searching the vector database."
                await resp.send()
                return
        await search_step.remove()

        async with cl.Step(name="building a prompt") as prompt_step:
            prompt_step.output = "Generating a full prompt on the system prompt, " \
                                 "user message, and search results..."
            is_error_prompt, full_prompt = await build_prompt(
                search_results,
                message.content,
                cl.user_session.get("chat_profile"),
                HistorySettings(
                    keep_history=settings["keep_history"],
                    message_history=message_history,
                ),
            )
            if is_error_prompt:
                await cl.Message(content=constants.WARNING_MESSAGE_TRUNCATED_TEXT).send()
        await prompt_step.remove()

        if debug_mode:
            await print_debug_content(settings, search_content,
                                      search_results, full_prompt)

        async with cl.Step(name="thinking and generating a response") as resp_step:
            # Process user message and get AI response
            is_error = await get_response(
                full_prompt,
                resp,
                {
                    "model": settings["generative_model"],
                    "max_tokens": settings["max_tokens"],
                    "temperature": settings["temperature"]
                },
                is_api=False,
                stream_response=settings.get("stream", True),
                step=resp_step,
            )

        if settings["keep_history"]:
            full_prompt.append(ChatCompletionAssistantMessageParam(
                role="assistant",
                content=resp.content,
            ))
            cl.user_session.set('message_history', full_prompt)

        if not is_error:
            # Extend response with searched jira urls
            append_searched_urls(search_results, resp, settings.get("enable_rerank", True))

    update_msg_count()
    await resp.send()


async def handle_user_message_api( # pylint: disable=too-many-arguments
    message_content: str,
    similarity_threshold: float,
    generative_model_settings: ModelSettings,
    embeddings_model_settings: ModelSettings,
    profile_name: str,
    enable_rerank: bool = True,
    ) -> MockMessage:
    """
    API handler for user messages without Chainlit context.
    """
    response = MockMessage(content="", urls=[])

    # Check message length
    is_valid_length, error_message = await check_message_length(message_content)
    if not is_valid_length:
        response.content = error_message
        return response

    collections = get_collections_per_profile(profile_name)
    error_message = check_collections(collections)
    if error_message:
        response.content = error_message
        return response

    # Perform search in all collections (embedding generated inside)
    try:
        search_results = await perform_multi_collection_search(
            message_content,
            embeddings_model_settings["model"],
            similarity_threshold=similarity_threshold,
            collections=collections,
            settings={
                "enable_rerank": enable_rerank,
                "rerank_top_n": config.rerank_top_n,
            },
        )
    except httpx.HTTPStatusError:
        response.content = "An error occurred while searching the vector database."
        return response

    is_error_prompt, full_prompt = await build_prompt(
        search_results,
        message_content,
        profile_name,
        HistorySettings(
            keep_history=False,
            message_history=[],
        ),
    )
    # Process user message and get AI response
    is_error = await get_response(
        full_prompt,
        response,
        generative_model_settings,
        is_api=True,
        stream_response=False
    )

    if not is_error:
        append_searched_urls(search_results, response, enable_rerank, urls_as_list=True)

    if is_error_prompt:
        response.content = ("Warning! The content from the vector database "
                            "has been truncated.\n" + response.content)

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

    # If threshold is above 1, cap it at 1
    return min(threshold, 1.0)

async def get_embeddings_model_name() -> str:
    """Get name of the embeddings model."""
    try:
        settings = cl.user_session.get("settings")
        return settings.get("embeddings_model")
    except ChainlitContextException:
        return await get_default_embeddings_model_name()

def check_collections(collections_to_check: list[str]) -> str:
    """
    Verify if the specified collections exist in the vector store.

    Args:
        collections_to_check: A list of collection names to verify.

    Returns:
        An error message string listing missing collections, or an empty string
        if all collections exist.
    """
    available_collections = vector_store.get_collections()
    missing_collections = [
        collection for collection in collections_to_check
        if collection not in available_collections
    ]

    if missing_collections:
        return (f"The following collections are configured but not found in "
                f"the vector store: {', '.join(missing_collections)}. "
                f"Please ensure they are created or update the configuration.")
    return ""


def get_collections_per_profile(
    profile_name: str
) -> list[str]:
    """
    Get the collections associated with a specific profile.

    Args:
        profile_name: The name of the profile

    Returns:
        A list of collections associated with the profile
    """
    if profile_name == DOCS_PROFILE:
        return [
            config.vectordb_collection_name_documentation,
            config.vectordb_collection_name_errata,
            config.vectordb_collection_name_solutions,
        ]
    if profile_name == RCA_FULL_PROFILE:
        return [
            config.vectordb_collection_name_jira,
            config.vectordb_collection_name_documentation,
            config.vectordb_collection_name_errata,
            config.vectordb_collection_name_solutions,
        ]

    return [
        config.vectordb_collection_name_jira,
        config.vectordb_collection_name_ci_logs,
    ]
