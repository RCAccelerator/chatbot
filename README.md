# RCAccelerator

A Chainlit-based chatbot with RAG (Retrieval-Augmented Generation) capabilities for accelerating Root Cause Analysis (RCA) in Continuous Integration (CI) environments.

## Overview

RCAccelerator is an interactive AI assistant designed to assist engineering teams in identifying, analyzing, and resolving CI failures more efficiently. Built on top of [Chainlit](https://www.chainlit.io/), it leverages LLMs and vector search to combine conversational intelligence with deep contextual recall of past incidents, bugs, and system knowledge.

## Running the App

* Install dependencies (which also takes care of installing PDM if needed):

```bash
make install-deps
```

* Create a `.env` file or export the necessary environment variables.

* Start the Chainlit app:

```bash
pdm run chainlit run src/app.py
```

## Environment Variables

The application can be configured using the following environment variables:

### LLM Settings

* `GENERATION_LLM_API_URL`: URL for the generative LLM API
* `GENERATION_LLM_API_KEY`: API key for the generative LLM
* `GENERATION_LLM_MODEL_NAME`: Name of the generative model to use (default: 'mistralai/Mistral-7B-Instruct-v0.3')
* `EMBEDDINGS_LLM_API_URL`: URL for the embeddings LLM API
* `EMBEDDINGS_LLM_API_KEY`: API key for the embeddings LLM
* `EMBEDDINGS_LLM_MODEL_NAME`: Name of the embeddings model to use (default: 'BAAI/bge-m3')

### Model Parameters

* `DEFAULT_MODEL_TEMPERATURE`: Temperature setting for generation (default: 0.7)
* `DEFAULT_MODEL_MAX_TOKENS`: Maximum tokens for generation (default: 1024)
* `DEFAULT_MODEL_TOP_P`: Top-p sampling parameter (default: 1)
* `DEFAULT_MODEL_N`: Number of completions to generate (default: 1)

### Chatbot Database Settings

* `DATABASE_URL`: PostgreSQL connection URL for the chatbot sessions
* `AUTH_DATABASE_URL`: PostgreSQL connection URL for storing users and passwords

### Vector Database Settings

* `VECTORDB_URL`: URL for the vector database
* `VECTORDB_API_KEY`: API key for the vector database
* `VECTORDB_PORT`: Port for the vector database (default: 6333)
* `VECTORDB_COLLECTION_NAME`: Collection name in the vector database (default: 'rca-knowledge-base')
