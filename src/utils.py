"""Utility functions shared across modules."""

import chainlit as cl

def extract_model_ids(models) -> list[str]:
    """Extracts model IDs from the models list."""
    model_ids = []
    for model in models.data:
        model_ids.append(model.id)
    if not model_ids:
        cl.logger.error("No models available.")
        return []
    return model_ids
