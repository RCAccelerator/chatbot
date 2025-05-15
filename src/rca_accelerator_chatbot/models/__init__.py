"""Communicate with various models needed by the Chatbot.

This package initializes three singleton instances used to communicate with:
    - the generative model
    - the embedding model
    - the rerank model

These singletons are initialized using values provided by the user via the
environment variables (see config module). Note that init_model_providers()
must be called by the consumer of this package in order for the all provider's
functionality to be fully available.
"""
import asyncio
from rca_accelerator_chatbot.models.generative import gen_model_provider
from rca_accelerator_chatbot.models.embeddings import embed_model_provider
from rca_accelerator_chatbot.models.rerank import rerank_model_provider


async def init_model_providers():
    """Initialize all providers"""
    await asyncio.gather(
        gen_model_provider.init(),
        embed_model_provider.init(),
        rerank_model_provider.init(),
    )
