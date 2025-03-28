"""Handler for user feedback on responses."""
import chainlit as cl

from db.mongodb import mongodb_client


async def handle_feedback(action):
    """
    Handle user feedback on chat responses.
    Updates the database with the feedback value.

    Args:
        action: The feedback action from the user
    """
    message_id = action.forId
    feedback_value = action.payload.get("feedback")

    success = mongodb_client.update_feedback(message_id, feedback_value)

    if not success:
        cl.logger.warning("Failed to save feedback for message %s",
                          str(message_id))
