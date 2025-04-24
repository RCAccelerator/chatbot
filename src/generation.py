"""Text generation with large language models."""

import chainlit as cl
from openai import AsyncOpenAI, OpenAIError

from embeddings import get_num_tokens
from settings import HistorySettings, ModelSettings
from config import config
from utils import extract_model_ids

# Initialize generative LLM client
gen_llm = AsyncOpenAI(
    base_url=config.generation_llm_api_url,
    organization='',
    api_key=config.generation_llm_api_key,
)


async def discover_generative_model_names() -> str:
    """Discover available generative LLM models."""
    models = await gen_llm.models.list()
    return extract_model_ids(models)


def _handle_context_size_limit(err: OpenAIError) -> str:
    if 'reduce the length of the messages or completion' in err.message:
        cl.user_session.set('message_history', '')
        return 'Request size with history exceeded limit, ' \
               'Please start a new thread.'
    return str(err)


async def get_response(history_settings: HistorySettings, # pylint: disable=too-many-arguments
                       user_message: cl.Message, response_msg: cl.Message,
                       model_settings: ModelSettings,
                       product_name: str,
                       stream_response: bool = True) -> bool:
    """Process the user's message and generate a response using the LLM.

    This function constructs the prompt from the LLM by compbining the system
    prompt with the user's input. It then sends the prepared message to the LLM
    for response generation.

    Args:
        user_message: The user's input message object.
        response_msg: The message object to populate with the LLM's
            generated response or an error message if something goes wrong.
        model_settings: A dictionary containing LLM configuration.
        stream_response: Indicates whether we want to stream the response or
            get the process in a single chunk.
    """
    is_error = True
    message_history = history_settings.get('message_history', [])
    system_prompt_with_product_name = config.system_prompt.replace(
        "PRODUCT_NAME", product_name
    )
    if not message_history:
        message_history = [
            {"role": "system", "content": system_prompt_with_product_name}]

    message_history.append({"role": "user", "content": user_message.content})
    try:
        if stream_response:
            async for stream_resp in await gen_llm.chat.completions.create(
                messages=message_history, stream=stream_response,
                **model_settings
            ):
                if stream_resp.choices and len(stream_resp.choices) > 0:
                    if token := stream_resp.choices[0].delta.content or "":
                        await response_msg.stream_token(token)
        else:
            response = await gen_llm.chat.completions.create(
                messages=message_history, stream=stream_response,
                **model_settings
            )
            response_msg.content = response.choices[0].message.content or ""
        message_history.append({"role": "assistant",
                                "content": response_msg.content})
        if history_settings.get('keep_history', True):
            cl.user_session.set('message_history', message_history)
        is_error = False
    except OpenAIError as e:
        err_msg = _handle_context_size_limit(e)

        cl.logger.error("Error in process_message_and_get_response: %s",
                        err_msg)
        response_msg.content = (
            f"I encountered an error while generating a response: {err_msg}."
        )
    return is_error


async def summarize_content(content: str) -> str:
    """
    Summarize the content using the generative LLM to fit within context limits.
    
    Args:
        content: The content to summarize
        
    Returns:
        Summarized content
    """
    
    try:
        # Get the approximate number of tokens in the content
        token_count = await get_num_tokens(content)
        cl.logger.info(f"Content to summarize has approximately {token_count} tokens")
        
        # If content is already small enough, return it as is
        safe_context_limit = min(16000, config.generation_llm_max_context // 2)
        if token_count <= safe_context_limit:
            cl.logger.info("Content is within token limits, no summarization needed")
            return content
            
        # Calculate safe token limit for each chunk (considering input + output)
        # Reserve space for the system prompt, user instruction and output tokens
        safe_chunk_limit = min(8000, config.generation_llm_max_context // 4)
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        
        # Limit the number of paragraphs to prevent excessive tokenization calls
        MAX_PARAGRAPHS = 1000
        if len(paragraphs) > MAX_PARAGRAPHS:
            cl.logger.warning(f"Too many paragraphs ({len(paragraphs)}), truncating to {MAX_PARAGRAPHS}")
            paragraphs = paragraphs[:MAX_PARAGRAPHS]
        
        # Create chunks based on character count estimation rather than token count
        # This reduces the number of tokenization API calls
        chunks = []
        current_chunk = []
        current_chunk_chars = 0
        # Approximate ratio of chars to tokens (this varies by language but works as estimate)
        chars_per_token = 4
        safe_chunk_chars = safe_chunk_limit * chars_per_token
        
        for paragraph in paragraphs:
            paragraph_chars = len(paragraph)
            
            # If adding this paragraph would exceed the limit, finalize current chunk
            if current_chunk_chars + paragraph_chars > safe_chunk_chars and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_chunk_chars = 0
            
            # If a single paragraph exceeds the limit, split it into smaller pieces
            if paragraph_chars > safe_chunk_chars:
                cl.logger.warning(f"Found extremely large paragraph ({paragraph_chars} chars), splitting it")
                # Split the paragraph into smaller chunks by character count
                for i in range(0, paragraph_chars, safe_chunk_chars):
                    chunk_end = min(i + safe_chunk_chars, paragraph_chars)
                    # Try to split at sentence boundaries when possible
                    split_point = paragraph[i:chunk_end].rfind('. ')
                    if split_point > safe_chunk_chars // 2:
                        chunks.append(paragraph[i:i+split_point+1])
                        i = i + split_point + 1
                    else:
                        chunks.append(paragraph[i:chunk_end])
            else:
                current_chunk.append(paragraph)
                current_chunk_chars += paragraph_chars
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Safety limit on number of chunks
        MAX_CHUNKS = 20
        if len(chunks) > MAX_CHUNKS:
            cl.logger.warning(f"Too many chunks ({len(chunks)}), limiting to {MAX_CHUNKS}")
            # Take first few and last few chunks to preserve context from beginning and end
            chunks = chunks[:MAX_CHUNKS//2] + chunks[-MAX_CHUNKS//2:]
        
        # Always process as chunks even if there's only one
        cl.logger.info(f"Content split into {len(chunks)} chunks for summarization")
        summaries = []
        for i, chunk in enumerate(chunks):
            cl.logger.info(f"Summarizing chunk {i+1}/{len(chunks)} (approx. {len(chunk)/chars_per_token} tokens)")
            chunk_summary = await _perform_summarization(chunk)
            summaries.append(chunk_summary)
        
        # Combine summaries
        combined = "\n\n".join(summaries)
        combined_chars = len(combined)
        cl.logger.info(f"Combined summaries have approximately {combined_chars/chars_per_token} tokens")
        
        # If the combined summary is still too large, summarize it once more
        if combined_chars/chars_per_token > safe_chunk_limit:
            cl.logger.info("Performing final summarization of combined summaries")
            return await _perform_summarization(combined)
        else:
            return combined
            
    except Exception as e:
        cl.logger.error(f"Error in summarize_content: {str(e)}")
        # If summarization fails, truncate instead
        cl.logger.warning("Summarization failed, truncating content instead")
        return content[:2000]

async def _perform_summarization(content: str) -> str:
    """Helper function to perform the actual summarization API call"""
    try:
        # Safety check - if content is extremely large, truncate it before sending to API
        MAX_CHARS = 32000  # Conservative estimate to prevent API errors
        if len(content) > MAX_CHARS:
            cl.logger.warning(f"Content too large for summarization ({len(content)} chars), truncating")
            content = content[:MAX_CHARS] + "\n\n[Content was truncated due to size limitations]"
            
        summarize_messages = [
            {
                "role": "system", 
                "content": "You are a summarization assistant. Extract and summarize the key " +
                "information from the provided text. Maintain important details while being " +
                "concise."
            },
            {
                "role": "user", 
                "content": f"Please summarize the following content:\n\n{content}"
            }
        ]
        
        response = await gen_llm.chat.completions.create(
            messages=summarize_messages,
            model=config.generative_model,
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=config.default_max_tokens,  # Use default max tokens for summary
        )
        
        summarized_content = response.choices[0].message.content
        cl.logger.info("Content successfully summarized")
        return summarized_content
    except OpenAIError as e:
        cl.logger.error(f"Error summarizing content: {str(e)}")
        # If summarization fails, truncate instead (rough approximation)
        cl.logger.warning("Summarization failed, truncating content instead")
        return content[:2000]

async def preprocess_search_results(
        search_results: list, 
        max_context_size: int = config.generation_llm_max_context
    ) -> list[dict]:
    """
    Preprocess search results to ensure they fit within the context window.
    If the content exceeds the maximum context size, it will be summarized.
    
    Args:
        search_results: List of search results with text and metadata
        max_context_size: Maximum number of tokens allowed in the context
        
    Returns:
        Processed search results list that fits within the context window
    """
    from embeddings import get_num_tokens  # Import here to avoid circular imports
    
    # Combine search results into a single string for token count assessment
    combined_content = "\n\n".join([item.get("text", "") for item in search_results])
    
    # Check if content exceeds token limit
    token_count = await get_num_tokens(combined_content)
    
    if token_count > max_context_size:
        cl.logger.info(f"Search results exceed token limit: {token_count} > {max_context_size}. " +
                       "Summarizing...")
        # Summarize the content but preserve the list structure
        summarized_content = await summarize_content(combined_content)
        
        # Create a new list with a single entry containing the summarized content
        # but preserve other fields from the first result if available
        summarized_results = []
        if search_results:
            template = search_results[0].copy()
            template["text"] = summarized_content
            summarized_results.append(template)
        else:
            summarized_results.append({"text": summarized_content})
        
        return summarized_results
    
    return search_results
