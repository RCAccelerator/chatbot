"""
Chainlit-based chatbot for Root Cause Analysis assistance with RAG
capabilities.
"""
# Standard libs
import os

# Third-party libs
try:
    import openai
    from openai import OpenAI, AsyncOpenAI

    client = OpenAI(
        base_url=os.environ.get("OPENAI_API_BASE", "http://<changeme>/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "RedHat")
    )
    aclient = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_API_BASE", "http://<changeme>/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "RedHat")
    )
    import chainlit as cl
    from chainlit.input_widget import Select, Switch, Slider
    from pymongo import MongoClient
    from qdrant_client import QdrantClient
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("openai chainlit pymongo qdrant-client")
    raise


default_model = os.environ.get("DEFAULT_MODEL_NAME")
if default_model is None:
    default_model = []
    model_list = client.models.list()
    for model_name in model_list.data:
        if "completions" in model_name["id"].split("/"):
            default_model.append(model_name["id"])
else:
    default_model = [default_model]

default_embeddings_model = os.environ.get("DEFAULT_EMBEDDINGS_MODEL")
if default_embeddings_model is None:
    default_embeddings_model = []
    model_list = client.models.list()
    for model_name in model_list.data:
        if "embeddings" in model_name["id"].split("/"):
            default_embeddings_model.append(model_name["id"])
else:
    default_embeddings_model = [default_embeddings_model]

default_temperature = os.environ.get("DEFAULT_MODEL_TEMPERATURE", 0.7)
default_max_tokens = os.environ.get("DEFAULT_MODEL_MAX_TOKENS", 1024)
default_top_p = os.environ.get("DEFAULT_MODEL_TOP_P", 1)
default_n = os.environ.get("DEFAULT_MODEL_N", 1)

db_url = os.environ.get("DB_URL", "mongodb://<changeme>")
default_db_name = os.environ.get("DEFAULT_DB_NAME", "conversations")
default_collection_name = os.environ.get("DEFAULT_COLLECTION_NAME",
                                         "conv-collection")
vectordb_url = os.environ.get("VECTORDB_URL", "localhost")

# Vector Storage client
vectordb_client = QdrantClient(vectordb_url, port=6333)

# Message history storage
db_client = MongoClient(db_url)
conv_db = db_client[default_db_name]
collection = conv_db[default_collection_name]

default_logo_url = os.environ.get(
    "DEFAULT_LOGO_URL",
    "https://www.redhat.com/rhdc/managed-files/" +
    "Asset-Red_Hat-Logo_page-General-This-RGB.svg"
)
default_organization = os.environ.get(
    "DEFAULT_ORGANIZATION", "Red Hat"
)


def db_lookup(search_string: str, embedding_model: str, search_top_n: int = 3,
              search_sensitive: float = 0.83) -> list:
    """
    Search the vector database for relevant content based on the input query.

    Args:
        search_string: The query to search for
        embedding_model: The model to use for creating embeddings
        search_top_n: Maximum number of results to return
        search_sensitive: Minimum similarity score threshold
    Returns:
        List of search results with text and metadata
    """
    results = []
    embedding = client.embeddings.create(
        input=search_string,
        model=embedding_model
    )["data"][0]["embedding"]
    search_results = vectordb_client.search(collection_name="rca",
                                            query_vector=embedding,
                                            limit=search_top_n)
    for res in search_results:
        if res.score >= search_sensitive:
            res = {
                "text": res.payload.get("page_content", ""),
                "url": res.payload.get("url", None),
                "image": res.payload.get("image", None),
                "screen": res.payload.get("image", None),
            }
            results.append(res)
    return results


@cl.on_chat_start
async def init_chat():
    """
    Initialize the chat session with default settings and user interface
    elements.
    Sets up model selection, parameters, and initial message history.
    """
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Chat - Model",
                values=default_model,
                initial_index=0,
            ),
            Select(
                id="embeddings_model",
                label="Embeddings - Model",
                values=default_embeddings_model,
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
    cl.user_session.set(
        "message_history",
        [
            {
                "role": "system",
                "content": "You are an CI assistant. " +
                "You help with CI failures and help define RCA."
            }
        ],
    )
    await cl.Avatar(
        name=default_organization,
        url=default_logo_url,
    ).send()


@cl.action_callback("feedback")
async def on_action(action):
    """
    Handle user feedback on chat responses.
    Updates the database with the feedback value.
    """
    query_filter = {"message_id": action.forId}
    value = {"$set": {"feedback": action.value}}
    collection.update_one(query_filter, value)
    await cl.Message(
        content="Thank you! Your feedback will be used for future " +
        "improvements.").send()


@cl.on_message
async def main(message: str):
    """
    Main message handler that processes user input, searches for context,
    and generates AI responses.
    """
    message_history = cl.user_session.get("message_history")
    model_settings = cl.user_session.get("model_settings")
    constructed_prompt = ""

    try:
        search_results = db_lookup(message, model_settings["embeddings_model"])
        if len(search_results) > 0:
            constructed_prompt = ("Answer the question based only on the "
                                  "following context:")
            for s_result in search_results:
                if s_result["text"] != "":
                    constructed_prompt += s_result["text"] + "\n"
            constructed_prompt += "Question: " + message
    except (openai.OpenAIError, ValueError, KeyError) as e:
        search_results = ["Search error"]
        cl.logger.debug(f"Search error: {e}")

    if constructed_prompt != "":
        message_history.append({"role": "user", "content": constructed_prompt})
        model_settings["temperature"] = 0.1
    else:
        message_history.append({"role": "user", "content": message})

    actions = [
            cl.Action(
                name="feedback",
                value="positive",
                label="Awesome!",
                description="Positive feedback"
            ),
            cl.Action(
                name="feedback",
                value="negative",
                label="Report a message",
                description="Negative feedback"
            )
        ]
    msg = cl.Message(content="", actions=actions)
    if model_settings["stream"]:
        async for stream_resp in aclient.chat.completions.create(
            messages=message_history,
            stream=True,
            **model_settings
        ):
            token = stream_resp.choices[0]["delta"].get("content", "")
            await msg.stream_token(token)
    else:
        content = await aclient.chat.completions.create(
            messages=message_history, **model_settings)
        msg.content = content.choices[0]["message"].get("content", "")
    message_history.append({"role": "assistant", "content": msg.content})

    user_id = cl.user_session.get("id")

    record = {
        "conversation_id": user_id,
        "create_at": msg.created_at,
        "message_id": msg.id,
        "prompt": message,
        "search_results": search_results,
        "model_reply": msg.content,
        "settings": model_settings,
        "feedback": None
    }
    collection.insert_one(record)
    await msg.send()
