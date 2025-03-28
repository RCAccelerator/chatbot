"""Text generation with large language models."""
import chainlit as cl
from openai import AsyncOpenAI, OpenAIError

from config import (GEN_LLM_API_URL, GEN_LLM_API_KEY,
                    SYSTEM_PROMPT)

# Initialize generative LLM client
gen_llm = AsyncOpenAI(
    base_url=GEN_LLM_API_URL,
    organization='',
    api_key=GEN_LLM_API_KEY
)


async def generate_response(user_message, model_settings, stream=True):
    """
    Generate a response from the LLM based on the user message.

    Args:
        user_message: The user's input message
        model_settings: Dictionary containing model configuration parameters
        stream: Whether to stream the response or return it all at once

    Returns:
        Generated response text or stream
    """
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    message_history.append({"role": "user", "content": user_message})

    try:
        if stream:
            return await gen_llm.chat.completions.create(
                messages=message_history, stream=True, **model_settings
            )
        response = await gen_llm.chat.completions.create(
            messages=message_history, stream=False, **model_settings
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        cl.logger.error("Error generating response: %s", str(e))
        return "I'm sorry, I encountered an error while generating a response."


async def process_message_and_get_response(user_message, response_msg,
                                           model_settings):
    """
    Process the user message and generate an AI response.

    Args:
        user_message: The user's input message object
        response_msg: The response message object to populate
        model_settings: Dictionary containing model configuration parameters
    """
    try:
        if model_settings.get('stream', True):
            async for stream_resp in await generate_response(
                user_message.content, model_settings, stream=True
            ):
                if stream_resp.choices and len(stream_resp.choices) > 0:
                    if token := stream_resp.choices[0].delta.content or "":
                        await response_msg.stream_token(token)
        else:
            content = await generate_response(
                user_message.content, model_settings, stream=False
            )
            response_msg.content = content
    except OpenAIError as e:
        cl.logger.error("Error in process_message_and_get_response: %s",
                        str(e))
        response_msg.content = (
            "I encountered an error while generating a response."
        )
