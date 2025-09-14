import asyncio

import pytest

from src.server import _streaming_response


async def _collect_stream_text(resp) -> str:
    chunks = []
    async for c in resp.body_iterator:  # type: ignore[attr-defined]
        if isinstance(c, bytes):
            chunks.append(c.decode())
        else:
            chunks.append(str(c))
    return "".join(chunks)


class FakeStream:
    def __init__(self, events):
        self._events = list(events)
        self._iter = iter(self._events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, events):
        self.events = events

    def converse_stream(self, **kwargs):
        return {"stream": FakeStream(self.events)}


@pytest.mark.asyncio
async def test_stream_text_flow():
    # Simulate a simple text generation sequence
    events = [
        {"messageStart": {"model": "anthropic.claude"}},
        {"contentBlockStart": {"index": 0, "start": {"text": ""}}},
        {"contentBlockDelta": {"index": 0, "delta": {"text": "Hello"}}},
        {"contentBlockStop": {"index": 0}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 5}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    client = FakeClient(events)
    body = {"model": "claude-3-5-sonnet-20240620", "stream": True}
    args = {"modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0", "messages": []}

    resp = _streaming_response(client, body, args)

    payload = await _collect_stream_text(resp)
    assert "event: message_start" in payload
    assert "event: content_block_start" in payload
    assert "text\":\"Hello\"" in payload
    assert "event: content_block_stop" in payload
    assert "event: message_delta" in payload
    assert "\"stop_reason\":\"end_turn\"" in payload
    assert "\"usage\":{\"input_tokens\":3,\"output_tokens\":5}" in payload
    assert "event: message_stop" in payload


@pytest.mark.asyncio
async def test_stream_tool_use_flow():
    # Simulate a tool_use block with partial JSON input
    events = [
        {"contentBlockStart": {"index": 0, "start": {"toolUse": {"toolUseId": "tu1", "name": "fs"}}}},
        {"contentBlockDelta": {"index": 0, "delta": {"toolUse": {"input": {"partialJson": "{\"path\": \"/tmp\"}"}}}}},
        {"contentBlockStop": {"index": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]

    client = FakeClient(events)
    body = {"model": "claude-3-5-sonnet-20240620", "stream": True}
    args = {"modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0", "messages": []}

    resp = _streaming_response(client, body, args)

    payload = await _collect_stream_text(resp)
    assert "event: content_block_start" in payload
    assert "\"type\":\"tool_use\"" in payload
    assert "event: content_block_delta" in payload
    assert "input_json_delta" in payload
    assert "event: content_block_stop" in payload
    assert "event: message_stop" in payload


@pytest.mark.asyncio
async def test_stream_tool_result_flow():
    # Simulate an unusual provider emission of toolResult in assistant stream
    events = [
        {"contentBlockStart": {"index": 0, "start": {"toolResult": {"toolUseId": "tu1", "status": "success"}}}},
        {"contentBlockDelta": {"index": 0, "delta": {"toolResult": {"content": {"partialJson": "{\"ok\":true}"}}}}},
        {"contentBlockStop": {"index": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    client = FakeClient(events)
    body = {"model": "claude-3-5-sonnet-20240620", "stream": True}
    args = {"modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0", "messages": []}

    resp = _streaming_response(client, body, args)

    payload = await _collect_stream_text(resp)
    assert "event: content_block_start" in payload
    assert "\"type\":\"tool_result\"" in payload
    assert "\"tool_use_id\":\"tu1\"" in payload
    assert "event: content_block_delta" in payload
    # Verify we used output_json_delta to represent tool_result JSON partials
    assert "output_json_delta" in payload
    # partial_json is a JSON string containing JSON; thus escaped quotes
    assert "\\\"ok\\\":true" in payload
    assert "event: content_block_stop" in payload
    assert "event: message_stop" in payload


@pytest.mark.asyncio
async def test_stream_mixed_blocks_flow():
    # Mixed response: text → tool_use → tool_result with usage and end_turn
    events = [
        {"contentBlockStart": {"index": 0, "start": {"text": ""}}},
        {"contentBlockDelta": {"index": 0, "delta": {"text": "Step 1"}}},
        {"contentBlockStop": {"index": 0}},

        {"contentBlockStart": {"index": 1, "start": {"toolUse": {"toolUseId": "tu1", "name": "fs"}}}},
        {"contentBlockDelta": {"index": 1, "delta": {"toolUse": {"input": {"partialJson": "{\"path\":\"/tmp/test.txt\"}"}}}}},
        {"contentBlockStop": {"index": 1}},

        {"contentBlockStart": {"index": 2, "start": {"toolResult": {"toolUseId": "tu1", "status": "success"}}}},
        {"contentBlockDelta": {"index": 2, "delta": {"toolResult": {"content": {"partialJson": "{\"data\":\"ok\"}"}}}}},
        {"contentBlockStop": {"index": 2}},

        {"metadata": {"usage": {"inputTokens": 7, "outputTokens": 11}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    client = FakeClient(events)
    body = {"model": "claude-3-5-sonnet-20240620", "stream": True}
    args = {"modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0", "messages": []}

    resp = _streaming_response(client, body, args)

    payload = await _collect_stream_text(resp)

    # Three content block starts and stops
    assert payload.count("event: content_block_start") == 3
    assert payload.count("event: content_block_stop") == 3

    # Text delta present
    assert "\"type\":\"text\"" in payload
    assert "\"type\":\"text_delta\"" in payload
    assert "Step 1" in payload

    # Tool use and tool result present with their deltas
    assert "\"type\":\"tool_use\"" in payload
    assert "input_json_delta" in payload
    assert "/tmp/test.txt" in payload

    assert "\"type\":\"tool_result\"" in payload
    assert "output_json_delta" in payload
    # escaped raw JSON inside JSON string
    assert "\\\"data\\\":\\\"ok\\\"" in payload

    # Usage and stop reason emitted in message_delta
    assert "\"usage\":{\"input_tokens\":7,\"output_tokens\":11}" in payload
    assert "\"stop_reason\":\"end_turn\"" in payload
