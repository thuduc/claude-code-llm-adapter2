# Claude Code → Bedrock (Converse) Proxy: Implementation Plan

This document specifies a proxy that accepts Anthropic Claude Messages API requests from Claude Code and translates them to AWS Bedrock Converse API (boto3), including full support for tool use and streaming. It also defines response and streaming event mappings back to Anthropic-compatible formats so Claude Code works unchanged.

---

## Goals & Non‑Goals

- Goals
  - Provide a drop‑in HTTP endpoint compatible with Anthropic Messages API (`/v1/messages`).
  - Convert requests to Bedrock `converse` / `converse_stream` calls with feature parity for:
    - System prompts, multi-turn messages, max tokens, temperature, top_p, stop sequences.
    - Tool definitions, tool call outputs (tool_result), and tool choice.
    - Text and image inputs; assistant text and tool_use output.
    - Streaming via SSE with event-by-event fidelity to Anthropic’s stream format.
  - Map Bedrock usage metrics and stop reasons to Anthropic fields.
  - Configurable model ID mapping from Anthropic model names to Bedrock model IDs.
- Non‑Goals (initial)
  - Anthropic file uploads/attachments API surface. We support inline content only.
  - Audio input/output and vision beyond images. (Design leaves room to add.)
  - Anthropic batch or batch streaming endpoints.

---

## High‑Level Architecture

- Client (Claude Code) → Proxy (this service) → AWS Bedrock (Converse API)
- The proxy exposes Anthropic-compatible endpoints and headers, performs strict request/response transformations, and handles streaming SSE translation.
- Implementation stack
  - Language: Python 3.11+
  - Web: FastAPI + Uvicorn
  - AWS: boto3 Bedrock Runtime client (`bedrock-runtime`) using `converse` and `converse_stream`.

---

## Endpoints & Protocol

- POST `/v1/messages`
  - Content-Type: `application/json`
  - Headers accepted for Anthropic compatibility:
    - `x-api-key`: treated as proxy auth secret (configurable); not forwarded to Anthropic.
    - `anthropic-version`: accepted and logged, not strictly required.
    - `anthropic-beta`: accepted and logged; used to gate optional features when applicable.
  - Request body: Anthropic Messages API schema (subset/superset described below).
  - Behavior:
    - If `stream: true`, respond with `text/event-stream` SSE following Anthropic events.
    - Else, respond with a single Anthropic-compatible JSON message object.

- GET `/healthz` – simple 200 for liveness.

---

## Configuration

- `PROXY_API_KEY` (optional): if set, incoming `x-api-key` must match.
- `AWS_REGION`: required, e.g., `us-east-1`.
- AWS credentials: resolved by default AWS SDK chain (env, profile, IMDS, etc.).
- `MODEL_ID_MAP_JSON` (optional): JSON string or file path mapping Anthropic model names to Bedrock model IDs. Example:
  ```json
  {
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-4-sonnet-latest": "anthropic.claude-4-sonnet-2025-xx-xx-v1:0"
  }
  ```
- `ALLOW_IMAGE_URL_FETCH` (default false): if true, proxy may fetch remote `image_url` inputs; otherwise only base64 images are accepted.
- Timeouts and keepalive: server-level (e.g., 60–120s) and SSE ping interval (e.g., 15s).

---

## Request Conversion (Anthropic → Bedrock Converse)

Incoming Anthropic request (representative):
```json
{
  "model": "claude-3-5-sonnet-20240620",
  "system": "You are a coding assistant.",
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "Write a Python function"},
      {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    ]}
  ],
  "tools": [
    {"name": "fs_read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
  ],
  "tool_choice": "auto",
  "temperature": 0.2,
  "top_p": 0.95,
  "max_tokens": 1024,
  "stop_sequences": ["\n```"],
  "stream": false
}
```

Bedrock `converse` parameters built by proxy:
```python
client.converse(
  modelId=map_model_id(body["model"]),
  messages=[
    # Converted roles and content blocks – see Content Mapping below
  ],
  system=[{"text": "You are a coding assistant."}],
  toolConfig={
    "tools": [
      {"name": "fs_read", "description": "Read a file", "inputSchema": { ... JSON Schema ... }}
    ],
    "toolChoice": map_tool_choice(body.get("tool_choice"))
  } if body.get("tools") else None,
  inferenceConfig={
    "maxTokens": body.get("max_tokens"),
    "temperature": body.get("temperature"),
    "topP": body.get("top_p"),
    "stopSequences": body.get("stop_sequences")
  },
  additionalModelRequestFields=collect_provider_specific_fields(body)
)
```

Notes
- Unknown/advanced Anthropic fields are forwarded (when possible) via `additionalModelRequestFields` for provider-specific support (e.g., reasoning budgets, response formats) without breaking.
- `system` may be string or array; normalize to list of `{"text": ...}` entries.
- `messages` must include prior turns (including assistant tool_use and user tool_result blocks) for multi-turn tool cycles.

---

## Content Mapping (Bidirectional)

Roles
- Anthropic `role`: `user` | `assistant` (rarely `tool` in some clients). Map directly to Bedrock `role`.
- If `role` == `tool`, coerce to `user` with a `toolResult` content block (compat shim).

User/Assistant content blocks
- Text
  - Anthropic: `{ "type": "text", "text": "..." }`
  - Bedrock: `{ "text": "..." }`
- Image (input)
  - Anthropic: `{ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "..." } }`
  - Bedrock: `{ "image": { "format": "png", "source": { "bytes": b"..." } } }`
  - If `image_url` and `ALLOW_IMAGE_URL_FETCH` true: fetch bytes; otherwise reject with 400.
- Tool Use (assistant output)
  - Anthropic: `{ "type": "tool_use", "id": "...", "name": "fs_read", "input": { ... } }`
  - Bedrock: `{ "toolUse": { "toolUseId": "...", "name": "fs_read", "input": { ... } } }`
- Tool Result (user follow-up input)
  - Anthropic: `{ "type": "tool_result", "tool_use_id": "...", "content": "..." | [{"type":"text","text":"..."}], "is_error": true? }`
  - Bedrock: `{ "toolResult": { "toolUseId": "...", "content": [ {"text": "..."} ] | [ {"json": {...}} ], "status": "error" | "success" } }`
  - If Anthropic `content` is plain string, wrap as text block. If structured JSON provided, map to `{"json": ...}`.

System prompt
- Anthropic `system`: string or array of content blocks; Bedrock expects `system: [{"text": ...}, ...]`.

---

## Tool Definitions & Choice Mapping

- Tool definition
  - Anthropic: `{ name, description, input_schema (JSON Schema) }`
  - Bedrock: `{ name, description, inputSchema (same JSON Schema) }`
- Tool choice
  - Anthropic values accepted: `"auto" | "none" | {"type":"tool","name":"..."}` (some clients may send `"any"`).
  - Bedrock:
    - `"auto"` → `"auto"`
    - `"none"` → `{ "none": {} }`
    - `{type:"tool", name}` → `{ "tool": { "name": name } }`
    - `"any"` (if present) → `{ "any": {} }`
- Parallel tool calls
  - Anthropic may signal preferences (e.g., disable parallel). Bedrock exposing parallelism control is model/provider-specific; proxy will best-effort pass related hints via `additionalModelRequestFields` if present, otherwise default to model behavior.

---

## Inference Config Mapping

- `max_tokens` → `inferenceConfig.maxTokens`
- `temperature` → `inferenceConfig.temperature`
- `top_p` → `inferenceConfig.topP`
- `stop_sequences` → `inferenceConfig.stopSequences`
- Additional provider/model features
  - Forward via `additionalModelRequestFields`, e.g., `response_format`, `reasoning`, `tool_use_config`, etc., when present, so future Claude 4 / Bedrock features can be carried through.

---

## Response (Non‑Streaming) Mapping (Bedrock → Anthropic)

Bedrock `converse` response (representative):
```json
{
  "output": {
    "message": {
      "role": "assistant",
      "content": [ {"text": "..."}, {"toolUse": {"toolUseId": "id", "name": "fs_read", "input": {...}}} ]
    }
  },
  "stopReason": "end_turn" | "tool_use" | "max_tokens" | "stop_sequence",
  "usage": {"inputTokens": 123, "outputTokens": 456}
}
```

Anthropic message response to return:
```json
{
  "id": "msg_...",            // generated by proxy (UUID based)
  "type": "message",
  "model": original_request.model,
  "role": "assistant",
  "content": [
    {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "id", "name": "fs_read", "input": {...}}
  ],
  "stop_reason": map_stop_reason(stopReason),
  "stop_sequence": null | "...",
  "usage": { "input_tokens": 123, "output_tokens": 456 }
}
```

Stop reason mapping
- `end_turn` → `end_turn`
- `tool_use` → `tool_use`
- `max_tokens` → `max_tokens`
- `stop_sequence` → `stop_sequence`

Content mapping
- Reverse of Content Mapping section; convert Bedrock content blocks back to Anthropic blocks.

---

## Streaming (SSE) Mapping

We will translate Bedrock `converse_stream` event stream to Anthropic SSE events. Ordering and granularity MUST match Anthropic’s expectations used by Claude Code.

Bedrock events → Anthropic SSE
- `messageStart` → `event: message_start`
  - data: `{ "type": "message", "id": "msg_...", "model": ..., "role": "assistant" }`
- `contentBlockStart` (text) → `event: content_block_start`
  - data: `{ "index": i, "content_block": { "type": "text", "text": "" } }`
- `contentBlockDelta` (text) → `event: content_block_delta`
  - data: `{ "index": i, "delta": { "type": "text_delta", "text": "...chunk..." } }`
- `contentBlockStart` (toolUse) → `event: content_block_start`
  - data: `{ "index": i, "content_block": { "type": "tool_use", "id": "...", "name": "...", "input": {} } }`
- `contentBlockDelta` (toolUse JSON) → `event: content_block_delta`
  - data: `{ "index": i, "delta": { "type": "input_json_delta", "partial_json": "...partial..." } }`
- `contentBlockStop` → `event: content_block_stop`
  - data: `{ "index": i }`
- `messageStop` → before `message_stop`, send `message_delta` if stop_reason/usage available
  - `event: message_delta`
    - data: `{ "stop_reason": map_stop_reason(...), "stop_sequence": null | "..." }` (if provided by Bedrock stream)
  - `event: message_stop`
    - data: `{}`
- `metadata` → may include `usage`. Map to `event: message_delta` with usage fields when emitted.
- Keepalive: send `event: ping` every N seconds when idle.

Error handling
- Bedrock stream exceptions → `event: error` with `{ "error": { "type": "provider_error", "message": "..." } }` and then close.

SSE formatting
- Each event is written as:
  - `event: <name>\n`
  - `data: <single-line JSON>\n\n`
- Ensure JSON is minified (no newlines) per SSE conventions.

---

## Error Mapping (HTTP / Body)

- Auth failures → 401 with Anthropic-like error JSON: `{ "type": "error", "error": { "type": "authentication_error", "message": "..." } }`
- Validation errors → 400 `{ "type": "error", "error": { "type": "invalid_request_error", "message": "..." } }`
- Bedrock rejected request → 502/503 with `{ "type": "error", "error": { "type": "provider_error", "message": "...", "raw": <optional sanitized details> } }`
- Timeouts → 504 gateway timeout with similar shape.

---

## Model ID Mapping

- Provide a mapping layer to convert Anthropic model names used by Claude Code to Bedrock model IDs. Strategy:
  1. If request `model` already looks like a Bedrock model ID (prefix `anthropic.`), pass through.
  2. Else, try `MODEL_ID_MAP_JSON` mapping.
  3. Else, apply best-effort heuristics (e.g., normalize date suffix) or return 400 with helpful message.
- Log chosen IDs; do not log prompts/content by default.

---

## Security & Compliance

- Authenticate client with `x-api-key` if `PROXY_API_KEY` is set; otherwise allow.
- Never log message content or tool inputs unless explicitly enabled (PII risk).
- Use AWS default credential chain; support per-request override of region via config only.
- CORS: disable by default; enable if needed with explicit allowlist.

---

## Implementation Outline (Modules & Responsibilities)

Proposed layout
- `src/server.py` – FastAPI app; routes, request validation, streaming responses.
- `src/config.py` – config/env loading; model ID map.
- `src/bedrock_client.py` – boto3 client factory, `converse`, `converse_stream` wrappers.
- `src/mapping.py` – pure functions to map Anthropic ⇄ Bedrock (requests, messages, content blocks, tools, stop reasons, usage).
- `src/sse.py` – SSE writer utilities (minify JSON, ping keepalive, error events).
- `src/types_compat.py` – Pydantic models for accepted Anthropic payload subset; relaxed to allow forward-compat fields.
- `tests/test_mapping.py` – unit tests for mapping edge cases.
- `tests/test_streaming_map.py` – simulate Bedrock stream → expected SSE sequence.

Key functions (pseudocode)
```python
# mapping.py

def map_tool_choice(anthropic_choice): ...

def map_messages_to_bedrock(anthropic_messages, allow_image_url):
    out = []
    for m in anthropic_messages:
        role = map_role(m["role"])  # user/assistant/(tool→user)
        content = []
        for part in normalize_content_array(m.get("content")):
            if part["type"] == "text":
                content.append({"text": part["text"]})
            elif part["type"] == "image":
                img = to_bedrock_image(part)
                content.append({"image": img})
            elif part["type"] == "tool_use":
                content.append({"toolUse": {
                    "toolUseId": part["id"],
                    "name": part["name"],
                    "input": part.get("input", {})
                }})
            elif part["type"] == "tool_result":
                content.append({"toolResult": {
                    "toolUseId": part["tool_use_id"],
                    "content": to_bedrock_tool_result_content(part.get("content")),
                    "status": "error" if part.get("is_error") else "success"
                }})
            else:
                raise UnsupportedContentError(part["type"]) 
        out.append({"role": role, "content": content})
    return out


def map_bedrock_message_to_anthropic(msg):
    blocks = []
    for b in msg.get("content", []):
        if "text" in b:
            blocks.append({"type": "text", "text": b["text"]})
        elif "toolUse" in b:
            tu = b["toolUse"]
            blocks.append({
                "type": "tool_use",
                "id": tu.get("toolUseId"),
                "name": tu.get("name"),
                "input": tu.get("input", {})
            })
        elif "toolResult" in b:
            tr = b["toolResult"]
            blocks.append({
                "type": "tool_result",
                "tool_use_id": tr.get("toolUseId"),
                "content": from_bedrock_tool_result_content(tr.get("content", [])),
                "is_error": tr.get("status") == "error"
            })
        elif "image" in b:
            # assistant images are not typical for Claude Code; ignore or map if required later
            pass
    return {
        "role": msg.get("role", "assistant"),
        "content": blocks
    }
```

Streaming translation (sketch)
```python
async def stream_to_anthropic_sse(bedrock_stream):
    msg_id = gen_msg_id()
    yield sse("message_start", {"type": "message", "id": msg_id, "model": ctx.model, "role": "assistant"})
    for event in bedrock_stream:
        et = event.event_type
        if et == "contentBlockStart":
            if event.start.type == "text":
                yield sse("content_block_start", {"index": event.index, "content_block": {"type": "text", "text": ""}})
            elif event.start.type == "toolUse":
                yield sse("content_block_start", {"index": event.index, "content_block": {"type": "tool_use", "id": event.tool_use_id, "name": event.name, "input": {}}})
        elif et == "contentBlockDelta":
            if event.delta.type == "text":
                yield sse("content_block_delta", {"index": event.index, "delta": {"type": "text_delta", "text": event.delta.text}})
            elif event.delta.type == "toolUse" :
                yield sse("content_block_delta", {"index": event.index, "delta": {"type": "input_json_delta", "partial_json": event.delta.partial_json}})
        elif et == "contentBlockStop":
            yield sse("content_block_stop", {"index": event.index})
        elif et == "metadata":
            if event.usage:
                yield sse("message_delta", {"usage": {"input_tokens": event.usage.inputTokens, "output_tokens": event.usage.outputTokens}})
        elif et == "messageStop":
            if event.stopReason:
                yield sse("message_delta", {"stop_reason": map_stop_reason(event.stopReason)})
            yield sse("message_stop", {})
        else:
            # ignore or log unknown chunks
            pass
```

---

## Edge Cases & Compatibility Notes

- Multiple tool calls in one assistant message: maintain content block ordering and indices consistently; Claude Code expects stable sequence.
- Tool result payload
  - If Anthropic sends `content` as array of blocks, map to Bedrock `content` array; otherwise text-only.
  - Preserve `is_error` → Bedrock `status: error`.
- Images in assistant output: rarely used by Claude Code; omit unless needed.
- Attachments / file IDs: not supported initially; respond 400 with clear message if provided outside inline content.
- `response_format` (Anthropic JSON mode): forward to Bedrock via `additionalModelRequestFields` (e.g., `{"responseFormat": {"type":"json_object"}}`) if supported by target model.
- Token accounting: map Bedrock `usage.inputTokens`/`outputTokens` 1:1. If missing in stream, emit at end if available.
- Stop sequence echo: if Bedrock returns `stop_sequence`, include in Anthropic response/delta when available.
- Version headers: accept any `anthropic-version` and proceed; log for analytics only.

---

## Testing Plan

- Unit tests (no network)
  - Content mapping: text, image (base64), tool_use, tool_result (text and json), role coercion.
  - Tool choice mapping across all variants.
  - Inference config mapping, system prompt normalization.
  - Model ID mapping heuristics and errors.
- Stream mapping tests
  - Simulate Bedrock stream event sequence: text-only, tool_use with JSON deltas, mixed blocks, multiple blocks.
  - Assert produced SSE sequence matches Anthropic expectations exactly (event names, payload shapes, ordering).
- Integration (with Bedrock mocked/stubbed)
  - Non-streaming: end-to-end request → mapped call → mapped response.
  - Streaming: end-to-end SSE.

---

## Logging & Observability

- Request ID per call; include in logs and SSE (via comment lines if helpful, optional).
- Log: model mapping decision, stop reason, token usage; exclude content by default.
- Error logging includes sanitized provider errors.

---

## Delivery Steps

1. Scaffold FastAPI project and config loader.
2. Implement mapping functions (requests, content blocks, tools, stop reasons).
3. Implement Bedrock client wrapper (`converse`, `converse_stream`).
4. Implement `/v1/messages` non-streaming path.
5. Implement SSE streaming path with event translation and keepalive pings.
6. Add authentication and error handling.
7. Add unit and stream mapping tests.
8. Manual validation against actual Bedrock model (optional, environment permitting).
9. Document environment variables and run instructions.

---

## Runbook (later)

- Start: `uvicorn src.server:app --host 0.0.0.0 --port 8000`
- Configure model mapping via `MODEL_ID_MAP_JSON`.
- Point Claude Code at proxy base URL, using its Anthropic client pointed to `http://localhost:8000` and providing any required proxy API key.

---

## Known Limitations / Future Work

- Audio input/output and advanced modalities not yet mapped.
- Attachments via Anthropic Files API are not supported; could be added by fetching file bytes from storage.
- Parallel tool call control may vary by model/provider.
- Bedrock guardrails configuration optional; can be added via config.
- Reasoning-specific controls (e.g., budget tokens) can be forwarded but need model-specific validation.

