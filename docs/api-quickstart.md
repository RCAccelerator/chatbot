# API Reference

## Overview

RCAccelerator provides an AI-assisted chatbot interface for troubleshooting CI failures, analyzing logs, and discovering related issues or tickets. This API wraps around the chatbot logic, performing tasks such as:

- Generating answers using a Large Language Model (LLM).
- Searching for relevant context in a vector database based on user queries.
- Handling adjustable parameters for generative models (model name, temperature, max tokens) and embeddings.

## Endpoints

### `POST /prompt`

#### Description

Processes a single user message along with various model and search parameters, then returns a generated response along with any relevant resource URLs.

#### Request Body

JSON object matching the schema:

| Field                   | Type    | Constraints                                                     | Description                                                                                                                                                 |
|-------------------------|---------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `content`               | string  | Required                                                        | The user’s message or prompt content.                                                                                                                       |
| `similarity_threshold`  | float   | Default: `config.search_similarity_threshold` Range: (0, 1]     | Filter for relevant documents by cosine similarity. A higher threshold yields fewer but more precise documents, while a lower threshold is more inclusive.  |
| `temperature`           | float   | Default: `config.default_temperature` Range: [0.1, 1.0]         | Controls the variability in generated responses. A value closer to 1.0 produces more creative/flexible answers; near 0.1 yields more deterministic results. |
| `max_tokens`            | int     | Default: `config.default_max_tokens` Range: [1, 1024]           | Limits the maximum length of the generated response.                                                                                                        |
| `vectordb_collection`   | string  | Default: `config.vectordb_collection_name`                      | The name of the vector database collection to search for relevant context documents.                                                                        |
| `generative_model_name` | string  | Default: `config.generative_model`                              | The name of the LLM to be used for response generation.                                                                                                     |
| `embeddings_model_name` | string  | Default: `config.embeddings_model`                              | The name of the embeddings model used for vector-based document similarity.                                                                                 |

#### Example Request

```json
POST /prompt
Content-Type: application/json

{
  "content": "Why did my CI job fail?",
  "similarity_threshold": 0.75,
  "temperature": 0.7,
  "max_tokens": 300,
  "vectordb_collection": "rca-knowledge-base",
  "generative_model_name": "gpt3.5",
  "embeddings_model_name": "ada-002"
}
```

#### Response Body

JSON object containing either:

- A successful response with the generated answer and any relevant resource URLs:

  ```json
  {
    "response": "...your AI-generated answer...",
    "urls": ["http://link1", "http://link2"]
  }
  ```

- An error message if invalid parameters are provided:

  ```json
  {
    "error": "Invalid collection name. Available collections are: (...)"
  }
  ```

| Field      | Type            | Description                                                                                     |
|------------|-----------------|-------------------------------------------------------------------------------------------------|
| `response` | string          | AI-generated response text.                                                                     |
| `urls`     | array of string | Zero or more URLs deemed relevant to the query.                                                |
| `error`    | string          | Only present if an error occurred (for example, invalid model name or unavailable collection).  |

#### Example Response

```json
HTTP/1.1 200 OK
Content-Type: application/json

{
  "response": "It looks like the job failed because the test environment couldn't connect to the database...",
  "urls": ["https://jira.my-company.com/browse/CI-1234"]
}
```

### Example Curl Request

```bash
curl -X POST https://my-server.com/prompt \
    -H "Content-Type: application/json" \
    -d '{
      "content": "What caused my test job to fail on environment X?",
      "similarity_threshold": 0.8,
      "temperature": 0.5,
      "max_tokens": 1024,
      "vectordb_collection": "rca-knowledge-base",
      "generative_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
      "embeddings_model_name": "adaBAAI/bge-m3"
    }'
```

Response:

```json
{
  "response": "Based on logs from rca-knowledge-base, it appears that the test environment encountered a missing dependency...",
  "urls": [
    "https://jira.my-company.com/browse/CI-1234"
  ]
}
```
