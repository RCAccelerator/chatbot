"""Configuration settings for the RCA chatbot application."""

from dataclasses import dataclass
import os

# Constants for values we don't want to expose to the user
SUGGESTED_MINIMUM_SIMILARITY_THRESHOLD = 0.3


# pylint: disable=too-many-instance-attributes,too-few-public-methods
@dataclass(frozen=True)
class Config:
    """Configuration class for the RCA chatbot."""
    generation_llm_api_url: str
    generation_llm_api_key: str
    embeddings_llm_api_url: str
    embeddings_llm_api_key: str
    generative_model: str
    embeddings_model: str
    default_temperature: float
    default_max_tokens: int
    default_top_p: float
    default_n: int
    auth_database_url: str
    vectordb_url: str
    vectordb_api_key: str
    vectordb_port: int
    vectordb_collection_name: str
    search_instruction: str
    search_similarity_threshold: float
    search_top_n: int
    system_prompt: str
    prompt_header: str
    welcome_message: str

    @classmethod
    def from_env(cls) -> 'Config':
        """Create Config instance from environment variables."""
        return cls(
            generation_llm_api_url=os.environ.get(
                "GENERATION_LLM_API_URL", "http://localhost:8000/v1"),
            generation_llm_api_key=os.environ.get(
                "GENERATION_LLM_API_KEY", ""),
            embeddings_llm_api_url=os.environ.get(
                "EMBEDDINGS_LLM_API_URL", "http://localhost:8000/v1"),
            embeddings_llm_api_key=os.environ.get(
                "EMBEDDINGS_LLM_API_KEY", ""),
            generative_model=os.environ.get(
                "GENERATION_LLM_MODEL_NAME",
                'mistralai/Mistral-7B-Instruct-v0.3'),
            embeddings_model=os.environ.get(
                "EMBEDDINGS_LLM_MODEL_NAME", 'BAAI/bge-m3'),
            default_temperature=float(
                os.environ.get("DEFAULT_MODEL_TEMPERATURE", 0.7)),
            default_max_tokens=int(
                os.environ.get("DEFAULT_MODEL_MAX_TOKENS", 1024)),
            default_top_p=float(
                os.environ.get("DEFAULT_MODEL_TOP_P", 1)),
            default_n=int(os.environ.get("DEFAULT_MODEL_N", 1)),
            auth_database_url=os.environ.get(
                "AUTH_DATABASE_URL",
                "postgresql://<username>:<password>@localhost:5432/users"),
            vectordb_url=os.environ.get(
                "VECTORDB_URL", "http://localhost:6333"),
            vectordb_api_key=os.environ.get("VECTORDB_API_KEY", ""),
            vectordb_port=int(os.environ.get("VECTORDB_PORT", 6333)),
            vectordb_collection_name=os.environ.get(
                "VECTORDB_COLLECTION_NAME", 'rca-knowledge-base'),
            search_instruction=os.environ.get(
                "SEARCH_INSTRUCTION",
                "Represent this sentence for searching relevant passages: "),
            search_similarity_threshold=float(
                os.environ.get("SEARCH_SIMILARITY_THRESHOLD", 0.8)),
            search_top_n=int(os.environ.get("SEARCH_TOP_N", 5)),
            system_prompt=os.environ.get(
                "SYSTEM_PROMPT",
                "You are a CI assistant. You help with CI "
                "failures and help define RCA."),
            welcome_message=os.environ.get(
                "WELCOME_MESSAGE",
                "I am your CI assistant. I will help you with your RCA."),
            prompt_header=os.environ.get(
                "CONTEXT_HEADER",
                "The following text provides information from Jira tickets. " +
                "Use this information to help resolve the issue mentioned " +
                "above.\n\n" +
                "Each ticket follows this structure:\n" +
                "<Ticket summary>\n" +
                "<Ticket description><Ticket comments>\n\n" +
                "Here is the text with the information from the tickets:\n")
        )


# Initialize the configuration
config = Config.from_env()
