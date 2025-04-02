"""
Chainlit-based chatbot for Root Cause Analysis assistance
with RAG capabilities.
"""
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from config import config, SUGGESTED_MINIMUM_SIMILARITY_THRESHOLD
from chat import handle_user_message
from auth import authentification


@cl.on_chat_start
async def init_chat():
    """
    Initialize the chat session with default settings and user interface
    elements.
    Sets up model selection, parameters, and initial message history.
    """

    app_user = cl.user_session.get("user")
    await cl.Message(
        content=(f"Hello {app_user.identifier}! " +
                 config.welcome_message)
    ).send()
    cl.user_session.set("counter", 0)

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Chat - Model",
                values=[config.generative_model],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Model Temperature",
                initial=config.default_temperature,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="max_tokens",
                label="Max Tokens",
                initial=config.default_max_tokens,
                min=1,
                max=1024,
                step=1,
            ),
            Slider(
                id="search_similarity_threshold",
                label="Search Similarity Threshold",
                initial=config.search_similarity_threshold,
                min=min(
                    config.search_similarity_threshold,
                    SUGGESTED_MINIMUM_SIMILARITY_THRESHOLD
                ),
                max=1,
                step=0.05,
            ),
            Switch(id="stream", label="Stream a response", initial=True)
        ]
    ).send()
    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):
    """Main message handler that processes user input."""
    await handle_user_message(message)


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    """
    Authentication callback to validate user credentials.
    Returns True if authentication is successful, False otherwise.
    """
    return authentification.authenticate(username, password)


@cl.on_chat_resume
async def on_chat_resume():
    """
    Handle chat resume event.
    This function can be used to restore the chat state or perform any
    necessary actions when the chat is resumed.
    """
    pass  # pylint: disable=unnecessary-pass


@cl.on_chat_end
async def end_chat():
    """
    Handle chat end event.
    This function can be used to perform cleanup or logging when the chat
    ends.
    """
    await cl.Message(content="Thank you for using the chatbot!").send()
