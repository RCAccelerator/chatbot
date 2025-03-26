"""
Chainlit-based chatbot for Root Cause Analysis assistance with RAG
capabilities.
"""
# Standard libs
import os

# Third-party libs
try:
    from openai import AsyncOpenAI
    import chainlit as cl
    import bcrypt
    from chainlit.context import ChainlitContextException
    from chainlit.input_widget import Select, Switch, Slider
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, PyMongoError
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import ApiException
except ImportError as import_exception:
    print(f"Error importing required libraries: {import_exception}")
    raise

SEARCH_INSTRUCTION = "Represent this sentence " \
                "for searching relevant passages: "

llm_api_url = os.environ.get("LLM_API_URL", 'http://<changeme>/v1')
llm_api_key = os.environ.get("OPENAI_API_KEY")
generative_model = os.environ.get("DEFAULT_MODEL_NAME",
                                  'Mistral-7B-Instruct-v0.2')
embeddings_model = os.environ.get("DEFAULT_EMBEDDINGS_MODEL",
                                  'bge-large-en-v1.5')

default_temperature = os.environ.get("DEFAULT_MODEL_TEMPERATURE", 0.7)
default_max_tokens = os.environ.get("DEFAULT_MODEL_MAX_TOKENS", 1024)
default_top_p = os.environ.get("DEFAULT_MODEL_TOP_P", 1)
default_n = os.environ.get("DEFAULT_MODEL_N", 1)

db_url = os.environ.get("DB_URL",
                        'mongodb://mongoadmin:<changeme>@<changeme>:27017/')
default_db_name = os.environ.get("DEFAULT_DB_NAME", 'conversations')
default_collection_name = os.environ.get("DEFAULT_COLLECTION_NAME",
                                         'debug-collection')
vectordb_url = os.environ.get("VECTORDB_URL", '<changeme>')

# Vector Storage client
vectordb_client = QdrantClient(vectordb_url, port=6333)
vectordb_collection_name = os.environ.get("VECTORDB_COLLECTION_NAME",
                                          'all-jira-tickets')

# Message history storage
try:
    DB_CLIENT = MongoClient(db_url, serverSelectionTimeoutMS=5000)
    # Validate connection immediately
    DB_CLIENT.admin.command('ping')
    CONV_DB = DB_CLIENT[default_db_name]
    COLLECTION = CONV_DB[default_collection_name]
    DB_AVAILABLE = True
    cl.logger.info("Successfully connected to MongoDB")
except (ConnectionFailure, PyMongoError) as exception:
    cl.logger.error("Failed to connect to MongoDB: %s", str(exception))
    DB_CLIENT = None
    CONV_DB = None
    COLLECTION = None
    DB_AVAILABLE = False

llm = AsyncOpenAI(
    base_url=llm_api_url,
    organization='',
    api_key=llm_api_key)


async def db_lookup(search_string: str,
                    model_name: str, search_top_n: int = 5,
                    search_sensitive: float = 0.8) -> list:
    """
    Search the vector database for relevant content based on the input query.
    Args:
        search_string: The query to search for
        model_name: The model to use for creating embeddings
        search_top_n: Maximum number of results to return
        search_sensitive: Minimum similarity score threshold
    Returns:
        List of search results with text and metadata
    """
    results = []
    try:
        embedding_response = await llm.embeddings.create(
            model=model_name,
            input=search_string,
            encoding_format='float'
        )

        if not embedding_response:
            cl.logger.error("Failed to get embeddings: " +
                            "No response from model %s", model_name)
            return results
        if not embedding_response.data or len(embedding_response.data) == 0:
            cl.logger.error("Failed to get embeddings: " +
                            "Empty response for model %s", model_name)
            return results

        embedding = embedding_response.data[0].embedding

        search_results = vectordb_client.search(
            collection_name=vectordb_collection_name,
            query_vector=embedding,
            limit=search_top_n)

        for res in search_results:
            if res.score >= search_sensitive:
                results.append(
                    {
                        "score": res.score,
                        "url": res.payload['url']
                    })
        return results
    except (ApiException, ValueError, KeyError) as e:
        cl.logger.error("Error in db_lookup: %s", str(e))
        # Return empty results on error instead of crashing
        return results


@cl.on_chat_start
async def init_chat():
    """
    Initialize the chat session with default settings and user interface
    elements.
    Sets up model selection, parameters, and initial message history.
    """

    # Get the username if available
    username = cl.user_session.get("username", "User")
    # Send a welcome message to the authenticated user
    await cl.Message(
        content=f"Welcome, {username}! How can I help with CI " +
        "troubleshooting today?"
    ).send()

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Chat - Model",
                values=[generative_model],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Model Temperature",
                initial=default_temperature,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="max_tokens",
                label="Max Tokens",
                initial=default_max_tokens,
                min=1,
                max=1024,
                step=1,
            ),
            Switch(id="stream", label="Stream a response", initial=True)
            ]
    ).send()
    cl.user_session.set("model_settings", settings)


@cl.action_callback("feedback")
async def on_action(action):
    """
    Handle user feedback on chat responses.
    Updates the database with the feedback value.
    """
    if not DB_AVAILABLE or COLLECTION is None:
        cl.logger.warning("MongoDB is not available - feedback " +
                          "will not be saved")
        return

    try:
        filter_msg = {"message_id": action.forId}
        # Use action.value from payload instead of directly accessing it
        value = {"$set": {"feedback": action.payload.get("feedback")}}
        COLLECTION.update_one(filter_msg, value)
    except PyMongoError as e:
        cl.logger.error("Failed to save feedback: %s", str(e))


@cl.on_message
async def main(message: cl.Message):
    """
    Main message handler that processes user input, searches for context,
    and generates AI responses.
    """
    model_settings = cl.user_session.get("model_settings")

    # Create response message with feedback actions
    actions = [
            cl.Action(name="feedback",
                      label="Affirmative",
                      payload={"feedback": "positive"}),
            cl.Action(name="feedback",
                      label="Negative",
                      payload={"feedback": "negative"})
        ]
    msg = cl.Message(content="", actions=actions)

    # Process user message and get AI response
    await process_message_and_get_response(message, msg, model_settings)

    # Perform search and display results
    search_results = await perform_search(message.content, msg.content)
    await display_search_results(search_results)

    # Save conversation data
    save_conversation_data(message, msg, model_settings, search_results)

    await msg.send()


async def process_message_and_get_response(user_message, response_msg,
                                           model_settings):
    """Helper function to process the message and get AI response"""
    message_history = [{"role": "system",
                        "content": "You are an CI assistant. "
                        "You help with CI failures and help define RCA."}]
    message_history.append({"role": "user", "content": user_message.content})

    if model_settings['stream']:
        async for stream_resp in await llm.chat.completions.create(
            messages=message_history, **model_settings
        ):
            # Check if choices exists and has at least one element
            if stream_resp.choices and len(stream_resp.choices) > 0:
                if token := stream_resp.choices[0].delta.content or "":
                    await response_msg.stream_token(token)
    else:
        content = await llm.chat.completions.create(
            messages=message_history, **model_settings)
        response_msg.content = content.choices[0].message.content


async def perform_search(user_content, response_content):
    """Helper function to perform search with user input and AI response"""
    search_results = []

    # Search based on user query first
    if user_content:
        search_query = SEARCH_INSTRUCTION + user_content
        search_results_query = await db_lookup(search_query, embeddings_model)
        search_results.extend(search_results_query)

    # Only search based on response if we have content
    if response_content:
        search_summ_message = SEARCH_INSTRUCTION + response_content
        search_results_sum = await db_lookup(search_summ_message,
                                             embeddings_model)
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
    """Helper function to display search results"""
    search_message = ""
    for result in search_results:
        score = result.get('score', 0)
        if score >= 0.8:
            search_message += f'🔗 {result["url"]}, Similarity Score: {score}\n'
    if search_message != "":
        await cl.Message(content="Top similar bugs:\n" + search_message).send()


def save_conversation_data(user_message, response_msg, model_settings,
                           search_results):
    """Helper function to save conversation data to database"""
    if not DB_AVAILABLE or COLLECTION is None:
        cl.logger.warning("MongoDB is not available - " +
                          "conversation data will not be saved")
        return

    try:
        user_id = cl.user_session.get("id")
        record = {
            "conversation_id": user_id,
            "create_at": response_msg.created_at,
            "message_id": response_msg.id,
            "prompt": user_message.content,
            "search_results": search_results,
            "model_reply": response_msg.content,
            "settings": model_settings,
            "feedback": None
        }
        COLLECTION.insert_one(record)
    except PyMongoError as e:
        cl.logger.error("Failed to save conversation data: %s", str(e))
        # Continue execution without crashing


def verify_credentials(username: str, password: str) -> bool:
    """
    Verify user credentials against the database using bcrypt.
    Args:
        username: The username to verify
        password: The password to verify
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Look up the user in the database
    user_collection = DB_CLIENT[default_db_name]["users"]
    user = user_collection.find_one({"username": username})

    if user and user.get("password"):
        # Verify the password using bcrypt
        stored_hash = user.get("password")
        # Convert password to bytes if it's not already
        password_bytes = (password.encode('utf-8')
                          if isinstance(password, str)
                          else password)
        # Check if the password matches the stored hash
        return bcrypt.checkpw(password_bytes, stored_hash)

    return False


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> bool:
    """
    Authenticate users with username and password.
    Args:
        username: The username provided by the user
        password: The password provided by the user
    Returns:
        bool: True if authentication is successful, False otherwise
    """
    cl.logger.info("Authentication attempt for user: %s", username)

    is_valid = verify_credentials(username, password)

    if is_valid:
        try:
            cl.user_session.set("username", username)
            cl.logger.info("Authentication successful for user: %s", username)
        except ChainlitContextException as context_exception:
            cl.logger.warning("Failed to set username in session: %s",
                              str(context_exception))
    else:
        cl.logger.warning("Authentication failed for user: %s", username)

    return is_valid
