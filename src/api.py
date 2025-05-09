"""
FastAPI endpoints for the RCAccelerator API.
"""
import asyncio
from typing import Dict, Any, List
import re
import httpx
from httpx_gssapi import HTTPSPNEGOAuth, OPTIONAL
from bs4 import BeautifulSoup
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from constants import CI_LOGS_PROFILE, DOCS_PROFILE, RCA_FULL_PROFILE
from chat import handle_user_message_api
from config import config
from settings import ModelSettings
from generation import discover_generative_model_names
from embeddings import discover_embeddings_model_names

app = FastAPI(title="RCAccelerator API")

class BaseModelSettings(BaseModel):
    """Base model with common settings for model configuration."""
    similarity_threshold: float = Field(
        config.search_similarity_threshold, ge=-1.0, le=1.0)
    temperature: float = Field(
        config.default_temperature, ge=0.0, le=1.0)
    max_tokens: int = Field(config.default_max_tokens, gt=1, le=1024)
    generative_model_name: str = Field("")
    embeddings_model_name: str = Field("")
    profile_name: str = Field(CI_LOGS_PROFILE)
    enable_rerank: bool = Field(config.enable_rerank)


class ChatRequest(BaseModelSettings):
    """Request model for the chat endpoint."""
    content: str


class RcaRequest(BaseModelSettings):
    """Request model for the RCA endpoint."""
    tempest_report_url: HttpUrl = Field(..., description="URL of the Tempest report HTML file.")


async def validate_settings(request: BaseModelSettings) -> BaseModelSettings:
    """Validate the settings for any request.
    This function performs checks to ensure the API request is valid.
    Some checks are performed asynchronously which is why we don't use
    the built-in Pydantic validators.
    """
    available_generative_models = await discover_generative_model_names()
    if not request.generative_model_name:
        request.generative_model_name = available_generative_models[0]
    elif request.generative_model_name not in available_generative_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid generative model. Available: {available_generative_models}"
        )

    available_embedding_models = await discover_embeddings_model_names()
    if not request.embeddings_model_name:
        request.embeddings_model_name = available_embedding_models[0]
    elif request.embeddings_model_name not in available_embedding_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid embeddings model. Available: {available_embedding_models}"
        )

    if request.profile_name not in [CI_LOGS_PROFILE, DOCS_PROFILE, RCA_FULL_PROFILE]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile name. Allowed: {[CI_LOGS_PROFILE, DOCS_PROFILE,
                                                      RCA_FULL_PROFILE]}"
        )

    return request


async def validate_chat_settings(request: ChatRequest) -> ChatRequest:
    """Type-specific validation for ChatRequest."""
    return await validate_settings(request)


async def validate_rca_settings(request: RcaRequest) -> RcaRequest:
    """Type-specific validation for RcaRequest."""
    return await validate_settings(request)


class RcaResponseItem(BaseModel):
    """Response item for a single RCA."""
    test_name: str
    response: str
    urls: List[str]


def _extract_test_name(test_name_part: str) -> str:
    """Extract the test name from the text before the traceback."""
    # Extract the test name using a regex pattern
    test_name_match = re.search(r'ft\d+\.\d+:\s*(.*?)\)?testtools', test_name_part)
    if test_name_match:
        test_name = test_name_match.group(1).strip()
        if test_name.endswith('('):
            test_name = test_name[:-1].strip()
    else:
        # Try alternative pattern for different formats
        test_name_match = re.search(r'ft\d+\.\d+:\s*(.*?)$', test_name_part)
        if test_name_match:
            test_name = test_name_match.group(1).strip()
        else:
            test_name = "Unknown Test Name"

    # Remove any content within square brackets
    # e.g. test_tagged_boot_devices[id-a2e65a6c,image,network,slow,volume]
    # becomes test_tagged_boot_devices
    test_name = re.sub(r'\[.*?\]', '', test_name).strip()

    # Remove any content within parentheses
    test_name = re.sub(r'\(.*?\)', '', test_name).strip()

    return test_name


async def fetch_and_parse_tempest_report(url: str) -> List[Dict[str, str]]:
    """Fetches and parses the Tempest HTML report to extract test names and tracebacks."""
    async with httpx.AsyncClient(verify=False, follow_redirects=True) as client:
        try:
            response = await client.get(url, auth=HTTPSPNEGOAuth(mutual_authentication=OPTIONAL))
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=400, detail=f"Error fetching URL: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Error response {exc.response.status_code} " +
                                f"while requesting {exc.request.url!r}.") from exc

    soup = BeautifulSoup(response.text, 'html.parser')
    failed_test_rows = soup.find_all('tr', id=re.compile(r'^ft\d+\.\d+'))

    results = []
    for row in failed_test_rows:
        row_text = row.get_text().strip()

        traceback_start_marker = "Traceback (most recent call last):"
        traceback_start_index = row_text.find(traceback_start_marker)

        if traceback_start_index != -1:
            test_name_part = row_text[:traceback_start_index].strip()
            test_name = _extract_test_name(test_name_part)

            traceback_text = row_text[traceback_start_index:]
            end_marker_index = traceback_text.find("}}}")
            if end_marker_index != -1:
                traceback_text = traceback_text[:end_marker_index].strip()
            else:
                traceback_text = traceback_text.strip()

            results.append({"test_name": test_name, "traceback": traceback_text})

    if not results:
        pass

    return results


@app.post("/prompt")
async def process_prompt(
        message_data: ChatRequest = Depends(validate_chat_settings)
    ) -> Dict[str, Any]:
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
        message_data.enable_rerank,
        )

    return  {
        "response": getattr(response, "content", ""),
        "urls": getattr(response, "urls", [])
    }


@app.post("/rca-from-tempest", response_model=List[RcaResponseItem])
async def process_rca(
        request: RcaRequest = Depends(validate_rca_settings)
    ) -> List[RcaResponseItem]:
    """
    FastAPI endpoint that extracts Root Cause Analyses (RCAs) from a Tempest report URL.
    """
    traceback_items = await fetch_and_parse_tempest_report(str(request.tempest_report_url))

    if not traceback_items:
        raise HTTPException(status_code=404, detail="No tracebacks found in " +
                            "the provided Tempest report URL.")

    generative_model_settings: ModelSettings = {
        "model": request.generative_model_name,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    embeddings_model_settings: ModelSettings = {
        "model": request.embeddings_model_name,
    }

    unique_items = {}
    for item in traceback_items:
        # If we've seen this test name before, skip it
        if item['test_name'] not in unique_items:
            unique_items[item['test_name']] = item

    tasks = []
    for test_name, item in unique_items.items():
        message = f"Test: {test_name}\n\n{item['traceback']}"
        task = handle_user_message_api(
            message_content=message,
            similarity_threshold=request.similarity_threshold,
            generative_model_settings=generative_model_settings,
            embeddings_model_settings=embeddings_model_settings,
            profile_name=request.profile_name,
            enable_rerank=request.enable_rerank,
        )
        tasks.append((test_name, task))

    raw_results = await asyncio.gather(*[task for _, task in tasks])

    response_list = [
        RcaResponseItem(
            test_name=test_name,
            response=getattr(res, "content", "Error generating RCA."),
            urls=getattr(res, "urls", [])
        )
        for (test_name, res) in zip([t[0] for t in tasks], raw_results)
    ]

    return response_list
