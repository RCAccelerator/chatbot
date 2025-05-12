"""Common type definitions shared across modules."""
from typing import TypedDict

from openai.types.chat import ChatCompletionMessageParam

# A list of messages in a single thread.
ThreadMessages = list[ChatCompletionMessageParam]

class HistorySettings(TypedDict):
    """
    A dictionary type that defines the settings for message history.

    Attributes:
        keep_history: Whether to keep the message history.
        message_history: The list of messages in the history.
    """
    keep_history: bool
    message_history: ThreadMessages


class ModelSettings(TypedDict):
    """
    A dictionary type that defines the settings for a model.

    Attributes:
        model: The name of the model.
        temperature: The temperature to use for generation.
        max_tokens: The maximum number of tokens in the output.
    """
    model: str
    temperature: float
    max_tokens: int
