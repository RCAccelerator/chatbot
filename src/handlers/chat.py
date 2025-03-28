"""Handler for chat messages and responses."""
import chainlit as cl

from llm.generation import process_message_and_get_response
from llm.embeddings import db_lookup
from db.mongodb import mongodb_client
from config import SEARCH_INSTRUCTION, EMBEDDINGS_MODEL


async def perform_search(user_content, response_content):
    """
    Perform search with user input and AI response.

    Args:
        user_content: User's input query
        response_content: AI's response content

    Returns:
        List of unique search results sorted by relevance
    """
    search_results = []

    # Search based on user query first
    if user_content:
        search_query = SEARCH_INSTRUCTION + user_content
        search_results_query = await db_lookup(search_query, EMBEDDINGS_MODEL)
        search_results.extend(search_results_query)

    # Only search based on response if we have content
    if response_content:
        search_summ_message = SEARCH_INSTRUCTION + response_content
        search_results_sum = await db_lookup(search_summ_message,
                                             EMBEDDINGS_MODEL)
        search_results.extend(search_results_sum)

    # Remove duplicates (based on URL) and sort by score
    unique_results = {}
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


async def display_search_results(search_results):
    """Display search results to the user."""
    search_message = ""
    for result in search_results:
        score = result.get('score', 0)
        if score >= 0.8:
            search_message += f'🔗 {result["url"]}, Similarity Score: {score}\n'
    if search_message != "":
        await cl.Message(content="Top similar bugs:\n" + search_message).send()


async def handle_user_message(message: cl.Message):
    """
    Main handler for user messages.

    Args:
        message: The user's input message
    """
    model_settings = cl.user_session.get("model_settings")

    # Create response message with feedback actions
    actions = [
        cl.Action(name="feedback", label="Affirmative",
                  payload={"feedback": "positive"}),
        cl.Action(name="feedback", label="Negative",
                  payload={"feedback": "negative"})
    ]
    msg = cl.Message(content="", actions=actions)

    # Process user message and get AI response
    await process_message_and_get_response(message, msg, model_settings)

    # Perform search and display results
    search_results = await perform_search(message.content, msg.content)
    await display_search_results(search_results)

    # Save conversation data
    user_id = cl.user_session.get("id")
    conversation_data = {
        "user_id": user_id,
        "user_message": message.content,
        "ai_response": msg.content,
        "model_settings": model_settings,
        "search_results": search_results
    }
    mongodb_client.save_conversation(conversation_data)

    await msg.send()
