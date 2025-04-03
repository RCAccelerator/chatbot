"""Handler for chat messages and responses."""
import chainlit as cl

from generation import process_message_and_get_response
from embeddings import search_similar_content
from config import config


async def perform_search(user_content: str) -> list[dict]:
    """
    Perform search inside of the vector DB to find information that might
    relate to the problem described by the user.

    Args:
        user_content: User's input query

    Returns:
        List of unique search results sorted by relevance
    """

    # Search based on user query first
    search_results_query = await search_similar_content(user_content)
    search_results = []
    search_results.extend(search_results_query)

    # Remove duplicates (based on URL) and sort by score
    unique_results: dict = {}
    for result in search_results:
        url = result.get('url')
        if url in unique_results:
            # Keep the result with higher score
            if result.get('score', 0) > unique_results[url].get('score', 0):
                unique_results[url] = result
        else:
            unique_results[url] = result

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
        return config.prompt_header + "NO relevant Jira tickets found."

    prompt = [
        f"{res.get('text')}, Similarity Score: {res.get('score', 0)}"
        for res in search_results
    ]

    return config.prompt_header + "\n" + "\n".join(prompt)


def append_searched_urls(search_results, resp):
    """
    Append search urls.

    Args:
        search_results: List of search results
        resp: The response message object to populate
    """
    search_message = ""
    for result in search_results:
        score = result.get('score', 0)
        search_message += f'🔗 {result["url"]}, Similarity Score: {score}\n'
    if search_message != "":
        resp.content += "\n\nTop similar bugs:\n" + search_message


def update_msg_count():
    """Update the number of messages in the conversation."""
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)


async def handle_user_message(message: cl.Message):
    """
    Main handler for user messages.

    Args:
        message: The user's input message
    """
    model_settings = cl.user_session.get("model_settings")

    resp = cl.Message(content="")

    if message.elements and message.elements[0].path:
        with open(message.elements[0].path, 'r', encoding='utf-8') as file:
            message.content += file.read()

    if message.content:
        search_results = await perform_search(message.content)
        message.content += build_prompt(search_results)

    # Process user message and get AI response
    await process_message_and_get_response(message, resp, model_settings)

    # Extend response with searched jira urls
    append_searched_urls(search_results, resp)

    update_msg_count()
    await resp.send()
