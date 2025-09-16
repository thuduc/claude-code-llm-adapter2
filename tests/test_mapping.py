import base64

import pytest

import src.mapping as mapping_module

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
    MappingError,
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
    assert img["name"] == "image PNG"
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
    assert out[0]["content"][0]["text"] == "Document attached."
    doc = out[0]["content"][1]["document"]
    assert doc["format"] in ("pdf", "octet-stream")
    assert doc["name"] == "document PDF" or doc["name"] == "document OCTET-STREAM"
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


def test_map_messages_missing_tool_result_raises():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "fs_read", "input": {"path": "/tmp/x"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "no result"}],
        },
    ]

    with pytest.raises(MappingError):
        map_messages_to_bedrock(msgs, allow_image_url=False)


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
    assert map_tool_choice("any") == {"any": {}}
    assert map_tool_choice({"type": "tool", "name": "fs_read"}) == {"tool": {"name": "fs_read"}}
    assert map_tool_choice({"name": "fs_read"}) == {"tool": {"name": "fs_read"}}
    assert map_tool_choice({"type": "any"}) == {"any": {}}
    assert map_tool_choice({"type": "auto"}) == {"auto": {}}

    with pytest.raises(MappingError):
        map_tool_choice("disabled")
    with pytest.raises(MappingError):
        map_tool_choice({"type": "tool"})
    with pytest.raises(MappingError):
        map_tool_choice(123)


def test_map_system_and_inference():
    system = map_system_to_bedrock(["You are helpful."])
    assert system == [{"text": "You are helpful."}]
    cfg = map_inference_config(
        {
            "max_tokens": 10,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 5,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.1,
            "max_output_tokens": 128,
            "stop_sequences": ["END"],
        }
    )
    assert cfg == {
        "maxTokens": 10,
        "temperature": 0.1,
        "topP": 0.9,
        "topK": 5,
        "presencePenalty": 0.2,
        "frequencyPenalty": 0.1,
        "maxOutputTokens": 128,
        "stopSequences": ["END"],
    }


def test_map_messages_image_url_fetch(monkeypatch):
    monkeypatch.setattr(
        mapping_module,
        "_REMOTE_FETCHER",
        lambda url: (b"img-bytes", "image/png"),
        raising=False,
    )
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "image_url",
                        "url": "https://example.com/cat.png",
                        "media_type": "image/png",
                    },
                }
            ],
        }
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=True)
    image = out[0]["content"][0]["image"]
    assert image["format"] == "png"
    assert image["name"] == "cat PNG"
    assert image["source"]["bytes"] == b"img-bytes"


def test_map_messages_document_url_fetch(monkeypatch):
    monkeypatch.setattr(
        mapping_module,
        "_REMOTE_FETCHER",
        lambda url: (b"%PDF-1.7", "application/pdf"),
        raising=False,
    )
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "document_url",
                        "url": "https://example.com/file.pdf",
                        "media_type": "application/pdf",
                    },
                }
            ],
        }
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=True)
    assert out[0]["content"][0]["text"] == "Document attached."
    doc = out[0]["content"][1]["document"]
    assert doc["format"] == "pdf"
    assert doc["name"] == "file PDF"
    assert doc["source"]["bytes"] == b"%PDF-1.7"


def test_map_document_with_text_no_padding():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see attachment"},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64.b64encode(b"%PDF-1.7\n...").decode(),
                    },
                },
            ],
        }
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=False)
    assert out[0]["content"][0] == {"text": "see attachment"}
    assert out[0]["content"][1]["document"]["name"].startswith("document")


def test_map_document_with_tool_result_no_padding():
    data = base64.b64encode(b"%PDF-1.7\n...").decode()
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-123",
                    "content": "ok",
                },
            ],
        }
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=False)
    assert out[0]["content"][0]["toolResult"]["toolUseId"] == "tool-123"
    assert out[0]["content"][1] == {"text": "Document attached."}
    assert out[0]["content"][2]["document"]["name"].startswith("document")


def test_tool_result_reordered_before_other_blocks():
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the result",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-abc",
                    "content": "done",
                },
            ],
        }
    ]
    out = map_messages_to_bedrock(msgs, allow_image_url=False)
    assert out[0]["content"][0]["toolResult"]["toolUseId"] == "tool-abc"
    assert out[0]["content"][1]["text"] == "Here is the result"


def test_tool_result_sorted_by_pending_order():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "id1", "name": "fs", "input": {}},
                {"type": "tool_use", "id": "id2", "name": "fs", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "id2", "content": "second"},
                {"type": "tool_result", "tool_use_id": "id1", "content": "first"},
            ],
        },
    ]

    out = map_messages_to_bedrock(msgs, allow_image_url=False)
    ids = [block["toolResult"]["toolUseId"] for block in out[1]["content"] if "toolResult" in block]
    assert ids == ["id1", "id2"]


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
