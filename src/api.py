"""
FastAPI endpoints for the RCAccelerator API.
"""
from fastapi import FastAPI, Body

from chat import handle_user_message_api

app = FastAPI(title="RCAccelerator API")


@app.post("/prompt")
async def process_prompt(message_data: dict = Body(...)) -> dict[str, str]:
    """
    FastAPI endpoint that processes a message and returns an answer.

    Args:
        message_data: Dictionary containing the message content

    Returns:
        The response generated by the chat handler
    """
    message = message_data.get("content")
    response = await handle_user_message_api(message)

    return {"response": response if response else ""}
