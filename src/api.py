"""
FastAPI endpoints for the RCAccelerator API.
"""
from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from constants import CI_LOGS_PROFILE, DOCS_PROFILE
from chat import handle_user_message_api
from config import config
from settings import ModelSettings
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
    generative_model_name: str = Field(
        config.generative_model,
        description="The name of the generative model to use."
    )
    embeddings_model_name: str = Field(
        config.embeddings_model,
        description="The name of the embeddings model to use."
    )
    profile_name: str = Field(
        CI_LOGS_PROFILE,
        description="The name of the profile to use."
    )

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

    @field_validator('profile_name', mode='after')
    @classmethod
    def validate_product_name(cls, value: str) -> str:
        """Validate the product name."""
        if value not in [CI_LOGS_PROFILE, DOCS_PROFILE]:
            raise ValueError("Invalid profile name. Available profiles are: " +
                             f"{[CI_LOGS_PROFILE, DOCS_PROFILE]}")
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

    response = await handle_user_message_api(
        message_data.content,
        message_data.similarity_threshold,
        generative_model_settings,
        embeddings_model_settings,
        message_data.profile_name,
        )

    return  {
        "response": getattr(response, "content", ""),
        "urls": getattr(response, "urls", [])
    }
