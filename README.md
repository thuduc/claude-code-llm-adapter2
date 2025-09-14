# Claude Code → Bedrock (Converse) Proxy

Proxy that accepts Anthropic Claude Messages API requests and forwards them to AWS Bedrock Converse API, translating requests, responses, and streaming events. Built for Claude Code with tool-use support.

Proxy Quick start
- Set `AWS_REGION` and AWS credentials in your environment.
- Optional: set `PROXY_API_KEY` to require an `x-api-key` header.
- Set `MODEL_ID_MAP_JSON` to map Anthropic model names to Bedrock model IDs.
  - Example:
    ```json
    {
      "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
      "claude-sonnet-4-20250514": "anthropic.claude-4-sonnet-20250514-v1:0"
    }
    ```
  - Can also be provided as a file via `MODEL_ID_MAP_JSON=@path/to/map.json`.

Claude Code quick start:
- Set environment variables
  - export ANTHROPIC_AUTH_TOKEN="sk-localkey"
  - export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
 

Install (locally)
```
pip install -r requirements.txt
```

Run
```
uvicorn src.server:app --host 127.0.0.1 --port 8080
```

Endpoint
- POST `/v1/messages` – Accepts Anthropic Messages API shape; supports `stream: true` SSE.
- GET `/healthz` – Liveness check.

Implementation details are in IMPLEMENTATION_PLAN.md.

Testing
- With Makefile (creates venv automatically):
  - `make dev`  # install runtime + dev deps
  - `make test`
- Or manual:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt -r requirements-dev.txt`
  - `pytest -q`
