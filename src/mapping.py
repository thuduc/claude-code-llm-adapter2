import base64
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


class MappingError(Exception):
    pass


class UnsupportedContentError(MappingError):
    pass


def normalize_content_array(content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, list):
        return content
    # Anthropic allows string shorthand in some contexts; normalize to text block
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    raise MappingError("content must be a list or string")


def map_role(role: str) -> str:
    if role == "tool":
        # No direct 'tool' role in Bedrock; tool results are content blocks under user
        return "user"
    if role in {"user", "assistant"}:
        return role
    # Default to user for unknown roles
    return "user"


def _decode_base64_to_bytes(data_b64: str) -> bytes:
    return base64.b64decode(data_b64, validate=False)


def to_bedrock_image(part: Dict[str, Any], allow_url: bool = False) -> Dict[str, Any]:
    src = part.get("source") or {}
    src_type = src.get("type")
    if src_type == "base64":
        media = src.get("media_type") or "image/png"
        fmt = media.split("/")[-1].lower()
        data = src.get("data") or ""
        return {"format": fmt, "source": {"bytes": _decode_base64_to_bytes(data)}}
    if src_type == "url" or src_type == "image_url":
        if not allow_url:
            raise MappingError("remote image URLs not allowed; enable ALLOW_IMAGE_URL_FETCH")
        # We do not fetch here – server should prefetch earlier if enabled.
        raise MappingError("image_url fetch not implemented in mapper; prefetch required")
    raise MappingError(f"unsupported image source type: {src_type}")


def to_bedrock_document(part: Dict[str, Any], allow_url: bool = False) -> Dict[str, Any]:
    src = part.get("source") or {}
    src_type = src.get("type")
    if src_type == "base64":
        media = src.get("media_type") or src.get("mediaType") or "application/pdf"
        fmt = (media.split("/")[-1] or "pdf").lower()
        data = src.get("data") or ""
        return {"format": fmt, "source": {"bytes": _decode_base64_to_bytes(data)}}
    if src_type in {"url", "document_url"}:
        if not allow_url:
            raise MappingError("remote document URLs not allowed; enable ALLOW_IMAGE_URL_FETCH")
        raise MappingError("document_url fetch not implemented in mapper; prefetch required")
    raise MappingError(f"unsupported document source type: {src_type}")


def to_bedrock_tool_result_content(content: Any) -> List[Dict[str, Any]]:
    # Anthropic tool_result.content can be a string or an array of blocks
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, dict):
        # Treat as structured JSON
        return [{"json": content}]
    if isinstance(content, list):
        out: List[Dict[str, Any]] = []
        for it in content:
            if isinstance(it, dict):
                t = it.get("type")
                if t == "text":
                    out.append({"text": it.get("text", "")})
                elif t == "json":
                    out.append({"json": it.get("json", {})})
                else:
                    # Best-effort: if it looks like JSON payload
                    if "json" in it:
                        out.append({"json": it.get("json")})
                    elif "text" in it:
                        out.append({"text": it.get("text", "")})
        return out
    # Fallback: dump to JSON string
    try:
        return [{"text": json.dumps(content)}]
    except Exception:
        return [{"text": str(content)}]


def from_bedrock_tool_result_content(content: List[Dict[str, Any]]) -> Any:
    # Return a string if single text, else array of blocks for richer data
    if not content:
        return ""
    if len(content) == 1 and "text" in content[0]:
        return content[0].get("text", "")
    blocks: List[Dict[str, Any]] = []
    for it in content:
        if "text" in it:
            blocks.append({"type": "text", "text": it.get("text", "")})
        elif "json" in it:
            blocks.append({"type": "json", "json": it.get("json", {})})
    return blocks


def map_messages_to_bedrock(anthropic_messages: List[Dict[str, Any]], allow_image_url: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in anthropic_messages or []:
        role = map_role(m.get("role", "user"))
        content_blocks: List[Dict[str, Any]] = []
        for part in normalize_content_array(m.get("content")):
            t = part.get("type")
            if t == "text":
                content_blocks.append({"text": part.get("text", "")})
            elif t == "image":
                content_blocks.append({"image": to_bedrock_image(part, allow_image_url)})
            elif t == "document":
                content_blocks.append({"document": to_bedrock_document(part, allow_image_url)})
            elif t == "tool_use":
                content_blocks.append({
                    "toolUse": {
                        "toolUseId": part.get("id"),
                        "name": part.get("name"),
                        "input": part.get("input", {}),
                    }
                })
            elif t == "tool_result":
                content_blocks.append({
                    "toolResult": {
                        "toolUseId": part.get("tool_use_id"),
                        "content": to_bedrock_tool_result_content(part.get("content")),
                        "status": "error" if part.get("is_error") else "success",
                    }
                })
            else:
                raise UnsupportedContentError(f"unsupported content type: {t}")
        out.append({"role": role, "content": content_blocks})
    return out


def map_system_to_bedrock(system: Any) -> Optional[List[Dict[str, str]]]:
    if not system:
        return None
    if isinstance(system, str):
        return [{"text": system}]
    # Could be array of content blocks; accept strings inside
    if isinstance(system, list):
        out: List[Dict[str, str]] = []
        for s in system:
            if isinstance(s, str):
                out.append({"text": s})
            elif isinstance(s, dict) and s.get("type") == "text":
                out.append({"text": s.get("text", "")})
        return out or None
    return None


def map_tool_choice(choice: Any) -> Any:
    if not choice or choice == "auto":
        return {"auto": {}}
    if choice == "none":
        return {"none": {}}
    if choice == "any":
        return {"any": {}}
    if isinstance(choice, dict):
        if choice.get("type") == "tool" and choice.get("name"):
            return {"tool": {"name": choice["name"]}}
    return {"auto": {}}


def map_tools_to_bedrock(tools: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not tools:
        return None
    bed_tools: List[Dict[str, Any]] = []
    for t in tools:
        name = t.get("name")
        desc = t.get("description")
        schema = t.get("input_schema") or {}
        # Bedrock expects JSON Schema under 'inputSchema'. Some models require {'json': schema} wrapper.
        input_schema = schema if "json" in schema else {"json": schema}
        bed_tools.append({
            "toolSpec": {
                "name": name,
                "description": desc,
                "inputSchema": input_schema,
            }
        })
    return {"tools": bed_tools}


def map_inference_config(body: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if body.get("max_tokens") is not None:
        cfg["maxTokens"] = body.get("max_tokens")
    if body.get("temperature") is not None:
        cfg["temperature"] = body.get("temperature")
    if body.get("top_p") is not None:
        cfg["topP"] = body.get("top_p")
    if body.get("stop_sequences") is not None:
        cfg["stopSequences"] = body.get("stop_sequences")
    return cfg


def collect_additional_fields(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Best-effort forward compatibility for features like response_format
    add: Dict[str, Any] = {}
    rf = body.get("response_format")
    if rf:
        # Anthropic style: {"type": "json_object"}
        add["responseFormat"] = rf
    # Pass through any provider-specific fields under a namespaced key
    provider = body.get("bedrock") or body.get("provider")
    if isinstance(provider, dict):
        add.update(provider)
    return add or None


def map_stop_reason(sr: Optional[str]) -> Optional[str]:
    if not sr:
        return None
    # Bedrock already uses Anthropic-like stop reasons in Converse
    # Map directly with a small alias table
    alias = {
        "end_turn": "end_turn",
        "tool_use": "tool_use",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
    }
    return alias.get(sr, sr)


def map_bedrock_message_to_anthropic(msg: Dict[str, Any]) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    for b in msg.get("content", []) or []:
        if "text" in b:
            blocks.append({"type": "text", "text": b.get("text", "")})
        elif "toolUse" in b:
            tu = b["toolUse"] or {}
            blocks.append({
                "type": "tool_use",
                "id": tu.get("toolUseId"),
                "name": tu.get("name"),
                "input": tu.get("input", {}),
            })
        elif "toolResult" in b:
            tr = b["toolResult"] or {}
            blocks.append({
                "type": "tool_result",
                "tool_use_id": tr.get("toolUseId"),
                "content": from_bedrock_tool_result_content(tr.get("content", [])),
                "is_error": tr.get("status") == "error",
            })
        # Images in assistant output are typically not used by Claude Code; skip for now
    return {"role": msg.get("role", "assistant"), "content": blocks}
