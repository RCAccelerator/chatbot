# Environment Variables


The following env variables can be altered when running the chantbot application locally
in order to create the chatbot Config instance based on them:

```
# LLM Settings
GENERATION_LLM_API_URL=your_generation_api_url                  # To initialize generative LLM client
GENERATION_LLM_API_KEY=your_generation_api_key                  # To initialize generative LLM client
GENERATION_LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3    # To alter chat settings with model selection

EMBEDDINGS_LLM_API_URL=your_embeddings_api_url                  # To initialize embedding LLM client
EMBEDDINGS_LLM_API_KEY=your_embeddings_api_key                  # To initialize embedding LLM client
EMBEDDINGS_LLM_MODEL_NAME=BAAI/bge-m3                           # To generate embeddings for the given text using the specified model

# Model Parameters
DEFAULT_MODEL_TEMPERATURE=0.7                                   # To alter chat settings with default model temperature selection
DEFAULT_MODEL_MAX_TOKENS=1024                                   # To alter chat settings with tokens limit selection

# Database Settings
AUTH_DATABASE_URL=your_auth_postgres_url                        # To alter the endpoint of a Postgres DB used for user authentication

# Vector Database Settings
VECTORDB_URL=your_vectordb_url                                  # To alter QdrantClient parameter for VectorDB endpoint
VECTORDB_API_KEY=your_vectordb_api_key                          # To alter QdrantClient parameter for VectorDB API key
VECTORDB_PORT=6333                                              # To alter QdrantClient parameter for VectorDB port
VECTORDB_COLLECTION_NAME=rca-knowledge-base                     # To alter VectorDB colection name (rca-knowledge-base)
```
