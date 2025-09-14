import base64

import pytest

from src.mapping import (
    collect_additional_fields,
    from_bedrock_tool_result_content,
    map_bedrock_message_to_anthropic,
    map_inference_config,
    map_messages_to_bedrock,
    map_stop_reason,
    map_system_to_bedrock,
    map_tool_choice,
    map_tools_to_bedrock,
)


def test_map_messages_text_only():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=False)
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["text"] == "Hi"
    assert out[1]["role"] == "assistant"
    assert out[1]["content"][0]["text"] == "Hello"


def test_map_messages_image_base64():
    data = base64.b64encode(b"123").decode()
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": data},
                }
            ],
        }
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=False)
    img = out[0]["content"][0]["image"]
    assert img["format"] == "png"
    assert img["source"]["bytes"] == b"123"


def test_map_messages_document_base64():
    data = base64.b64encode(b"%PDF-1.7\n...").decode()
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                }
            ],
        }
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=False)
    doc = out[0]["content"][0]["document"]
    assert doc["format"] in ("pdf", "octet-stream")
    assert isinstance(doc["source"]["bytes"], (bytes, bytearray))


def test_map_messages_tool_use_and_result():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "fs_read", "input": {"path": "/tmp/x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "file content"},
            ],
        },
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=False)
    tu = out[0]["content"][0]["toolUse"]
    assert tu["toolUseId"] == "tu1"
    assert tu["name"] == "fs_read"
    tr = out[1]["content"][0]["toolResult"]
    assert tr["toolUseId"] == "tu1"
    assert tr["content"][0]["text"] == "file content"


def test_map_tools_and_choice():
    tools = [
        {
            "name": "fs_read",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        }
    ]
    bed = map_tools_to_bedrock(tools)
    assert bed and "toolSpec" in bed["tools"][0]
    spec = bed["tools"][0]["toolSpec"]
    assert spec["name"] == "fs_read"
    assert "inputSchema" in spec

    assert map_tool_choice("auto") == {"auto": {}}
    assert map_tool_choice("none") == {"none": {}}
    assert map_tool_choice({"type": "tool", "name": "fs_read"}) == {"tool": {"name": "fs_read"}}


def test_map_system_and_inference():
    system = map_system_to_bedrock(["You are helpful."])
    assert system == [{"text": "You are helpful."}]
    cfg = map_inference_config({"max_tokens": 10, "temperature": 0.1, "top_p": 0.9, "stop_sequences": ["END"]})
    assert cfg == {"maxTokens": 10, "temperature": 0.1, "topP": 0.9, "stopSequences": ["END"]}


def test_map_bedrock_message_to_anthropic():
    msg = {
        "role": "assistant",
        "content": [
            {"text": "Hello"},
            {"toolUse": {"toolUseId": "tu1", "name": "fs", "input": {"x": 1}}},
            {"toolResult": {"toolUseId": "tu1", "content": [{"text": "ok"}], "status": "success"}},
        ],
    }
    out = map_bedrock_message_to_anthropic(msg)
    assert out["role"] == "assistant"
    assert out["content"][0] == {"type": "text", "text": "Hello"}
    assert out["content"][1]["type"] == "tool_use"
    assert out["content"][2]["type"] == "tool_result"


def test_tool_result_content_roundtrip():
    br = [{"text": "x"}, {"json": {"a": 1}}]
    ant = from_bedrock_tool_result_content(br)
    assert isinstance(ant, list)
    assert ant[0] == {"type": "text", "text": "x"}


def test_stop_reason_mapping():
    assert map_stop_reason("end_turn") == "end_turn"
    assert map_stop_reason("max_tokens") == "max_tokens"


def test_collect_additional_fields():
    body = {"response_format": {"type": "json_object"}, "bedrock": {"guardrail": {"id": "g1"}}}
    add = collect_additional_fields(body)
    assert add and "responseFormat" in add and "guardrail" in add
