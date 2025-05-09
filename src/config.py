"""Configuration settings for the RCA chatbot application."""

from dataclasses import dataclass
import os

from constants import (
    CI_LOGS_SYSTEM_PROMPT,
    DOCS_SYSTEM_PROMPT,
    WELCOME_MESSAGE,
    CONTEXT_HEADER,
    SEARCH_INSTRUCTION,
    JIRA_FORMATTING_SYNTAX,
)


# pylint: disable=too-many-instance-attributes,too-few-public-methods
@dataclass(frozen=True)
class Config:
    """Configuration class for the RCA chatbot."""
    generation_llm_api_url: str
    generation_llm_api_key: str
    enable_rerank: bool
    reranking_model_name: str
    reranking_model_api_key: str
    reranking_model_api_url: str
    reranking_model_max_context: int
    embeddings_llm_api_url: str
    embeddings_llm_api_key: str
    embeddings_llm_max_context: int
    generative_model_max_context: int
    default_temperature: float
    default_temperature_without_search_results: float
    default_max_tokens: int
    default_top_p: float
    default_n: int
    auth_database_url: str
    vectordb_url: str
    vectordb_api_key: str
    vectordb_port: int
    vectordb_collection_name_jira: str
    vectordb_collection_name_errata: str
    vectordb_collection_name_documentation: str
    vectordb_collection_name_ci_logs: str
    vectordb_collection_name_solutions: str
    search_instruction: str
    search_similarity_threshold: float
    search_top_n: int
    rerank_top_n: int
    ci_logs_system_prompt: str
    docs_system_prompt: str
    prompt_header: str
    welcome_message: str
    jira_formatting_syntax_prompt: str

    @classmethod
    def from_env(cls) -> 'Config':
        """Create Config instance from environment variables."""
        return cls(
            generation_llm_api_url=os.environ.get(
                "GENERATION_LLM_API_URL", "http://localhost:8000/v1"),
            generation_llm_api_key=os.environ.get(
                "GENERATION_LLM_API_KEY", ""),
            enable_rerank=os.environ.get(
                "ENABLE_RERANK", "true").lower() == "true",
            reranking_model_name=os.environ.get(
                "RERANKING_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
            ),
            reranking_model_api_url=os.environ.get(
                "RERANKING_MODEL_API_URL", "http://localhost:8001/v1"
            ),
            reranking_model_api_key=os.environ.get(
                "RERANKING_MODEL_API_KEY", ""
            ),
            reranking_model_max_context=int(os.environ.get(
                "RERANKING_MODEL_MAX_CONTEXT",
                8192,
            )),
            embeddings_llm_api_url=os.environ.get(
                "EMBEDDINGS_LLM_API_URL", "http://localhost:8000/v1"),
            embeddings_llm_api_key=os.environ.get(
                "EMBEDDINGS_LLM_API_KEY", ""),
            embeddings_llm_max_context=int(os.environ.get(
                "EMBEDDINGS_LLM_MAX_CONTEXT",
                8192,
            )),
            generative_model_max_context=int(os.environ.get(
                "GENERATIVE_MODEL_MAX_CONTEXT",
                32000,
            )),
            default_temperature=float(
                os.environ.get("DEFAULT_MODEL_TEMPERATURE", 0.3)),
            default_temperature_without_search_results=float(
                os.environ.get("DEFAULT_MODEL_TEMPERATURE_WITHOUT_SEARCH_RESULTS", 0.7)),
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
            vectordb_collection_name_jira=os.environ.get(
                "VECTORDB_COLLECTION_NAME_JIRA", 'rca-knowledge-base'),
            vectordb_collection_name_errata=os.environ.get(
                "VECTORDB_COLLECTION_NAME_ERRATA", 'rca-errata'),
            vectordb_collection_name_documentation=os.environ.get(
                "VECTORDB_COLLECTION_NAME_DOCUMENTATION", 'osp-docs-base'),
            vectordb_collection_name_ci_logs=os.environ.get(
                "VECTORDB_COLLECTION_NAME_CI_LOGS", 'rca-ci'),
            vectordb_collection_name_solutions=os.environ.get(
                "VECTORDB_COLLECTION_NAME_SOLUTIONS", 'rca-solutions'),
            search_instruction=os.environ.get(
                "SEARCH_INSTRUCTION", SEARCH_INSTRUCTION),
            search_similarity_threshold=float(
                os.environ.get("SEARCH_SIMILARITY_THRESHOLD", 0.8)),
            ci_logs_system_prompt=os.environ.get("CI_LOGS_SYSTEM_PROMPT", CI_LOGS_SYSTEM_PROMPT),
            docs_system_prompt=os.environ.get("DOCS_SYSTEM_PROMPT", DOCS_SYSTEM_PROMPT),
            welcome_message=os.environ.get("WELCOME_MESSAGE", WELCOME_MESSAGE),
            prompt_header=os.environ.get("CONTEXT_HEADER", CONTEXT_HEADER),
            jira_formatting_syntax_prompt=os.environ.get(
                "JIRA_FORMATTING_SYNTAX", JIRA_FORMATTING_SYNTAX),

            # The maximum number of points we can retrieve from a single vector
            # database collection.
            search_top_n=int(os.environ.get("SEARCH_TOP_N", 10)),

            # The maximum number of points we pass to the generative model after
            # reranking.
            rerank_top_n=int(os.environ.get("RERANK_TOP_N", 5)),
        )


# Initialize the configuration
config = Config.from_env()
