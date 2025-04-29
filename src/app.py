"""
Chainlit-based chatbot for Root Cause Analysis assistance
with RAG capabilities.
"""
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from config import config
import constants
from chat import handle_user_message
from auth import authentification
from generation import discover_generative_model_names
from embeddings import discover_embeddings_model_names


@cl.set_chat_profiles
async def chat_profile():
    """
    Define the chat profile for the application.
    This function sets up the chat profile with a name, description,
    icon, and a list of starters.
    The profile is used to customize the chat experience for users.
    """
    return [
        cl.ChatProfile(
            name=constants.CI_LOGS_PROFILE,
            markdown_description="Root Cause Analysis for CI logs",
            icon="/public/ci-logs.png",
            starters=[
                cl.Starter(
                    label="Help me with CI job RCA",
                    message="Explain me how to get help with CI failures",
                    icon="/public/debug.svg",
                ),
            ],
        ),
        cl.ChatProfile(
            name=constants.DOCS_PROFILE,
            markdown_description="Chat with documentation and errata",
            icon="/public/book.png",
            starters=[
                cl.Starter(
                    label="How to collect diagnostic information",
                    message="How to collecting diagnostic information for support" +
                            " Red Hat OpenStack Services on OpenShift",
                    icon="/public/debug.svg",
                ),
            ],
        ),
    ]


@cl.on_chat_start
async def init_chat():
    """
    Initialize the chat session with default settings and user interface
    elements.
    Sets up model selection, parameters, and initial message history.
    """

    cl.user_session.set("counter", 0)
    await setup_chat_settings()


async def setup_chat_settings():
    """
    Set up the chat settings interface with model selection,
    temperature, token limits, and other configuration options.
    """
    generative_model_names = await discover_generative_model_names()
    embeddings_model_names = await discover_embeddings_model_names()
    if not generative_model_names or not embeddings_model_names:
        cl.Message(
            content="No generative or embeddings model found. "
                    "Please check your configuration."
        ).send()
        return

    settings = await cl.ChatSettings(
        [
            Select(
                id="generative_model",
                label="Generative LLM Model",
                values=generative_model_names,
                initial_index=0,
            ),
            Select(
                id="embeddings_model",
                label="Embeddings LLM Model",
                values=embeddings_model_names,
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
                min=0,
                max=1,
                step=0.05,
            ),
            Switch(id="stream", label="Stream a response", initial=True),
            Switch(id="debug_mode", label="Debug Mode", initial=False),
            Switch(id="keep_history", label="Keep message history in thread", initial=True)
        ]
    ).send()
    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):
    """Main message handler that processes user input."""
    settings = cl.user_session.get("settings")
    await handle_user_message(message,
                              debug_mode=settings.get("debug_mode", False))


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
    await setup_chat_settings()


@cl.on_chat_end
async def end_chat():
    """
    Handle chat end event.
    This function can be used to perform cleanup or logging when the chat
    ends.
    """
    pass  # pylint: disable=unnecessary-pass
