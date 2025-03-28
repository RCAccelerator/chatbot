"""
Chainlit-based chatbot for Root Cause Analysis assistance
with RAG capabilities.
"""
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from config import (
    GENERATIVE_MODEL, DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS
)
from handlers.chat import handle_user_message
from handlers.feedback import handle_feedback


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
                values=[GENERATIVE_MODEL],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Model Temperature",
                initial=DEFAULT_TEMPERATURE,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="max_tokens",
                label="Max Tokens",
                initial=DEFAULT_MAX_TOKENS,
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
    """Handle feedback actions from users."""
    await handle_feedback(action)


@cl.on_message
async def main(message: cl.Message):
    """Main message handler that processes user input."""
    await handle_user_message(message)
