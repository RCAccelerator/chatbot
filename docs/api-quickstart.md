# API Reference

## Overview

RCAccelerator provides an AI-assisted chatbot interface for troubleshooting CI failures, analyzing logs, and discovering related issues or tickets. This API wraps around the chatbot logic, performing tasks such as:

- Generating answers using a Large Language Model (LLM).
- Searching for relevant context in a vector database based on user queries.
- Handling adjustable parameters for generative models (model name, temperature, max tokens) and embeddings.

## Swagger UI

The Swagger interface (available at `https://myserver.com/docs`) can be used to discover the API endpoints and also try them.

## Endpoints

### `POST /prompt`

#### Description

Processes a single user message along with various model and search parameters, then returns a generated response along with any relevant resource URLs.

**Authentication Required**: Include the token in the Authorization header.

#### Request Body

JSON object matching the schema:

| Field                   | Type    | Constraints                                                     | Description                                                                                                                                                 |
|-------------------------|---------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `content`               | string  | Required                                                        | The user's message or prompt content.                                                                                                                       |
| `similarity_threshold`  | float   | Default: `config.search_similarity_threshold` Range: (0, 1]     | Filter for relevant documents by cosine similarity. A higher threshold yields fewer but more precise documents, while a lower threshold is more inclusive.  |
| `temperature`           | float   | Default: `config.default_temperature` Range: [0.1, 1.0]         | Controls the variability in generated responses. A value closer to 1.0 produces more creative/flexible answers; near 0.1 yields more deterministic results. |
| `max_tokens`            | int     | Default: `config.default_max_tokens` Range: [1, 1024]           | Limits the maximum length of the generated response.                                                                                                        |
| `profile_name`          | string  | Default: `constants.CI_LOGS_PROFILE`                            | The name of the profile to use.                                                                                                                             |
| `generative_model_name` | string  | Default: `config.generative_model`                              | The name of the LLM to be used for response generation.                                                                                                     |
| `embeddings_model_name` | string  | Default: `config.embeddings_model`                              | The name of the embeddings model used for vector-based document similarity.                                                                                 |

#### Example Request

```json
POST /prompt
Content-Type: application/json
Authorization: Bearer 6e63fb8d-93a2-4c55-8694-c0e76a4a7233

{
  "content": "Why did my CI job fail?",
  "similarity_threshold": 0.75,
  "temperature": 0.7,
  "max_tokens": 300,
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
    -H "Authorization: Bearer 6e63fb8d-93a2-4c55-8694-c0e76a4a7233" \
    -d '{
      "content": "What caused my test job to fail on environment X?",
      "similarity_threshold": 0.8,
      "temperature": 0.5,
      "max_tokens": 1024,
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

### `POST /rca-from-tempest`

#### Description

Extracts Root Cause Analyses (RCAs) from a Tempest test report URL. This endpoint fetches the HTML report, parses out failed tests and their tracebacks, and generates an RCA for each unique test failure.

**Authentication Required**: Include the token in the Authorization header.

#### Request Body

JSON object matching the schema:

| Field               | Type   | Constraints | Description                               |
|---------------------|--------|-------------|-------------------------------------------|
| `tempest_report_url`| string | Required    | URL of the Tempest report HTML file.      |

#### Example Request

```json
POST /rca-from-tempest
Content-Type: application/json
Authorization: Bearer 6e63fb8d-93a2-4c55-8694-c0e76a4a7233

{
  "tempest_report_url": "https://storage.example.com/ci-logs/tempest-report.html"
}
```

#### Response Body

JSON array of objects, each containing:

| Field       | Type            | Description                                      |
|-------------|-----------------|--------------------------------------------------|
| `test_name` | string          | The name of the failed test.                     |
| `response`  | string          | AI-generated root cause analysis for the failure.|
| `urls`      | array of string | Zero or more URLs deemed relevant to the failure.|

#### Example Response

```json
HTTP/1.1 200 OK
Content-Type: application/json

[
  {
    "test_name": "test_volume_boot_pattern",
    "response": "The test failed because the instance failed to boot from volume. The traceback shows a timeout waiting for the instance to become active. This is likely due to an issue with the Cinder volume service not properly attaching the volume to the instance.",
    "urls": [
      "https://jira.my-company.com/browse/CI-5678",
      "https://docs.openstack.org/cinder/latest/troubleshooting.html"
    ]
  },
  {
    "test_name": "test_network_basic_ops",
    "response": "The network connectivity test failed due to a timeout waiting for the VM to become accessible via SSH. This suggests either a networking configuration issue or a problem with the security group rules.",
    "urls": [
      "https://jira.my-company.com/browse/NET-1234"
    ]
  }
]
```

### Example Curl Request

```bash
curl -X POST https://my-server.com/rca-from-tempest \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 6e63fb8d-93a2-4c55-8694-c0e76a4a7233" \
    -d '{
      "tempest_report_url": "https://storage.example.com/ci-logs/tempest-report.html"
    }'
```
