"""Configuration settings for the RCA chatbot application."""
import os

# LLM settings
GEN_LLM_API_URL = os.environ.get("LLM_API_URL", 'http://<changeme>/v1')
GEN_LLM_API_KEY = os.environ.get("OPENAI_API_KEY")
EMB_LLM_API_URL = os.environ.get("EMBEDDINGS_LLM_API_URL",
                                 'http://<changeme>/v1')
EMB_LLM_API_KEY = os.environ.get("EMBEDDINGS_LLM_API_API_KEY")
GENERATIVE_MODEL = os.environ.get("DEFAULT_MODEL_NAME",
                                  'Mistral-7B-Instruct-v0.2')
EMBEDDINGS_MODEL = os.environ.get("DEFAULT_EMBEDDINGS_MODEL",
                                  'bge-large-en-v1.5')

# Model parameters
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_MODEL_TEMPERATURE", 0.7))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MODEL_MAX_TOKENS", 1024))
DEFAULT_TOP_P = float(os.environ.get("DEFAULT_MODEL_TOP_P", 1))
DEFAULT_N = int(os.environ.get("DEFAULT_MODEL_N", 1))

# Database settings
DB_URL = os.environ.get("DB_URL",
                        'mongodb://mongoadmin:<changeme>@<changeme>:27017/')
DEFAULT_DB_NAME = os.environ.get("DEFAULT_DB_NAME", 'conversations')
DEFAULT_COLLECTION_NAME = os.environ.get("DEFAULT_COLLECTION_NAME",
                                         'debug-collection')

# Vector database settings
VECTORDB_URL = os.environ.get("VECTORDB_URL", '<changeme>')
VECTORDB_PORT = int(os.environ.get("VECTORDB_PORT", 6333))
VECTORDB_COLLECTION_NAME = os.environ.get("VECTORDB_COLLECTION_NAME",
                                          'all-jira-tickets')

# Search settings
SEARCH_INSTRUCTION = os.environ.get(
    "SEARCH_INSTRUCTION",
    "Represent this sentence for searching relevant passages: "
)
SEARCH_SIMILARITY_THRESHOLD = float(os.environ.get(
    "SEARCH_SIMILARITY_THRESHOLD", 0.8)
)
SEARCH_TOP_N = int(os.environ.get("SEARCH_TOP_N", 5))

# System prompts
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a CI assistant. You help with CI failures and help define RCA."
)
