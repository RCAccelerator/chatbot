"""
FastAPI endpoints for the RCAccelerator API.
"""
from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from constants import OPENSTACK_PROFILE
from chat import handle_user_message_api
from config import config
from settings import ModelSettings
from vectordb import vector_store
from generation import discover_generative_model_names
from embeddings import discover_embeddings_model_names

app = FastAPI(title="RCAccelerator API")


class ChatRequest(BaseModel):
    """
    Represents the parameters for a chat request, including
    message content, similarity threshold, temperature, and max token limit.
    """
    content: str
    similarity_threshold: float = Field(
        config.search_similarity_threshold,
        gte=-1.0,
        le=1.0
        )
    temperature: float = Field(
        config.default_temperature,
        gte=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        config.default_max_tokens,
        gt=1,
        le=1024
    )
    vectordb_collection_jira: str = Field(
        config.vectordb_collection_name_jira,
        description="The name of the vector database collection for Jira to use."
    )
    vectordb_collection_errata: str = Field(
        config.vectordb_collection_name_errata,
        description="The name of the vector database collection for Errata to use."
    )
    vectordb_collection_documentation: str = Field(
        config.vectordb_collection_name_documentation,
        description="The name of the vector database collection for Documentation to use."
    )
    generative_model_name: str = Field(
        config.generative_model,
        description="The name of the generative model to use."
    )
    embeddings_model_name: str = Field(
        config.embeddings_model,
        description="The name of the embeddings model to use."
    )
    product_name: str = Field(
        OPENSTACK_PROFILE,
        description="The name of the product to use."
    )

    @field_validator('vectordb_collection_jira', 'vectordb_collection_errata',
                     'vectordb_collection_documentation', mode='after')
    @classmethod
    def validate_vectordb_collections(cls, value: str) -> str:
        """Validate the vector database collection names."""
        # Only validate if value is not None or empty
        if not value:
            return value
        available_collections = vector_store.get_collections()
        if not available_collections:
            raise ValueError("No collections found in the vector database.")
        if value not in available_collections:
            raise ValueError(f"Invalid collection name: {value}. " +
                             f"Available collections are: {available_collections}")
        return value

    @field_validator('generative_model_name', mode='after')
    @classmethod
    def validate_generative_model_name(cls, value: str) -> str:
        """Validate the generative model name."""
        available_generative_models = discover_generative_model_names()
        if value not in available_generative_models:
            raise ValueError("Invalid generative model name. Available models are: " +
                             f"{available_generative_models}")
        return value

    @field_validator('embeddings_model_name', mode='after')
    @classmethod
    def validate_embeddings_model_name(cls, value: str) -> str:
        """Validate the embeddings model name."""
        available_embeddings_models = discover_embeddings_model_names()
        if value not in available_embeddings_models:
            raise ValueError("Invalid embeddings model name. Available models are: " +
                             f"{available_embeddings_models}")
        return value

    @field_validator('product_name', mode='after')
    @classmethod
    def validate_product_name(cls, value: str) -> str:
        """Validate the product name."""
        if value != OPENSTACK_PROFILE:
            raise ValueError(f"Invalid product name: {value}. " +
                             f"Available product names for now: {OPENSTACK_PROFILE}")
        return value


@app.post("/prompt")
async def process_prompt(message_data: ChatRequest) -> Dict[str, Any]:
    """
    FastAPI endpoint that processes a message and returns an answer.
    """
    generative_model_settings: ModelSettings = {
        "model": message_data.generative_model_name,
        "max_tokens": message_data.max_tokens,
        "temperature": message_data.temperature,
    }
    embeddings_model_settings: ModelSettings = {
        "model": message_data.embeddings_model_name,
    }

    vectordb_collections = [
        c for c in [message_data.vectordb_collection_jira,
                    message_data.vectordb_collection_errata,
                    message_data.vectordb_collection_documentation] if c
    ]
    if not vectordb_collections:
        return {"error": "No collections specified. Please specify at least one collection."}

    response = await handle_user_message_api(
        message_data.content,
        message_data.similarity_threshold,
        generative_model_settings,
        embeddings_model_settings,
        vectordb_collections,
        message_data.product_name,
        )

    return  {
        "response": getattr(response, "content", ""),
        "urls": getattr(response, "urls", [])
    }
