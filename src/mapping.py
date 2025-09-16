import base64
import json
import logging
import socket
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class MappingError(Exception):
    pass


class UnsupportedContentError(MappingError):
    pass


_REMOTE_FETCH_TIMEOUT_SECONDS = 10
_MAX_REMOTE_MEDIA_BYTES = 10 * 1024 * 1024
_REMOTE_FETCHER: Optional[Callable[[str], Tuple[bytes, Optional[str]]]] = None


def _normalize_media_type(media: Optional[str]) -> Optional[str]:
    if not media:
        return None
    return media.split(";", 1)[0].strip().lower()


def _fetch_remote_media(url: str) -> Tuple[bytes, Optional[str]]:
    """Fetch remote media bytes with basic validation and size limits."""

    if _REMOTE_FETCHER is not None:
        return _REMOTE_FETCHER(url)

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise MappingError("remote media URLs must use http or https")

    req = Request(url, headers={"User-Agent": "claude-code-proxy/1.0"})
    try:
        with urlopen(req, timeout=_REMOTE_FETCH_TIMEOUT_SECONDS) as resp:  # nosec B310 - controlled URL
            content_type = resp.headers.get("Content-Type")
            content_length = resp.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > _MAX_REMOTE_MEDIA_BYTES:
                        raise MappingError("remote media exceeds 10MB limit")
                except ValueError:
                    pass

            buf = bytearray()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > _MAX_REMOTE_MEDIA_BYTES:
                    raise MappingError("remote media exceeds 10MB limit")

            return bytes(buf), content_type
    except (HTTPError, URLError, socket.timeout) as exc:  # pragma: no cover - network errors
        raise MappingError(f"failed to fetch remote media: {exc}") from exc


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


def _sanitize_media_name(value: str, fallback: str) -> str:
    def _clean(raw: str) -> str:
        cleaned_chars: List[str] = []
        prev_space = False
        for ch in raw:
            if ch.isalnum():
                cleaned_chars.append(ch)
                prev_space = False
            elif ch in "-()[]":
                cleaned_chars.append(ch)
                prev_space = False
            elif ch.isspace():
                if not prev_space:
                    cleaned_chars.append(" ")
                    prev_space = True
            else:
                if not prev_space:
                    cleaned_chars.append(" ")
                    prev_space = True
        cleaned = "".join(cleaned_chars).strip()
        return cleaned

    for candidate in (value, fallback):
        if candidate:
            cleaned = _clean(candidate)
            if cleaned:
                return cleaned
    return fallback or "document"


def _resolve_media_name(
    part: Dict[str, Any], src: Dict[str, Any], fmt: str, url: Optional[str], fallback_prefix: str
) -> str:
    name = part.get("name") or src.get("name") or src.get("filename")
    if not name and url:
        parsed = urlparse(url)
        path = parsed.path or ""
        if path and path != "/":
            candidate = path.rsplit("/", 1)[-1]
            if candidate:
                name = candidate
    if name and "." in name:
        head, tail = name.rsplit(".", 1)
        if head and tail:
            name = f"{head} {tail.upper()}"
    fallback = f"{fallback_prefix} {fmt.upper()}" if fmt else fallback_prefix
    return _sanitize_media_name(name or fallback, fallback)


def to_bedrock_image(part: Dict[str, Any], allow_url: bool = False) -> Dict[str, Any]:
    src = part.get("source") or {}
    src_type = src.get("type")
    if src_type == "base64":
        media = src.get("media_type") or "image/png"
        fmt = media.split("/")[-1].lower()
        data = src.get("data") or ""
        name = _resolve_media_name(part, src, fmt, None, "image")
        return {"format": fmt, "name": name, "source": {"bytes": _decode_base64_to_bytes(data)}}
    if src_type == "url" or src_type == "image_url":
        if not allow_url:
            raise MappingError("remote image URLs not allowed; enable ALLOW_IMAGE_URL_FETCH")
        url = src.get("url")
        if not url:
            raise MappingError("image_url source missing 'url'")
        data, content_type = _fetch_remote_media(url)
        media = _normalize_media_type(src.get("media_type") or content_type)
        if not media:
            media = "image/png"
        if not media.startswith("image/"):
            raise MappingError(f"remote image must have image/* content-type, got '{media}'")
        fmt = media.split("/")[-1] or "png"
        name = _resolve_media_name(part, src, fmt, url, "image")
        return {"format": fmt.lower(), "name": name, "source": {"bytes": data}}
    raise MappingError(f"unsupported image source type: {src_type}")


def to_bedrock_document(part: Dict[str, Any], allow_url: bool = False) -> Dict[str, Any]:
    src = part.get("source") or {}
    src_type = src.get("type")
    if src_type == "base64":
        media = src.get("media_type") or src.get("mediaType") or "application/pdf"
        fmt = (media.split("/")[-1] or "pdf").lower()
        data = src.get("data") or ""
        name = _resolve_media_name(part, src, fmt, None, "document")
        return {"format": fmt, "name": name, "source": {"bytes": _decode_base64_to_bytes(data)}}
    if src_type in {"url", "document_url"}:
        if not allow_url:
            raise MappingError("remote document URLs not allowed; enable ALLOW_IMAGE_URL_FETCH")
        url = src.get("url")
        if not url:
            raise MappingError("document_url source missing 'url'")
        data, content_type = _fetch_remote_media(url)
        media = _normalize_media_type(src.get("media_type") or src.get("mediaType") or content_type)
        if media and not (media.startswith("application/") or media.startswith("text/")):
            raise MappingError(f"remote document content-type '{media}' is not supported")
        fmt = (media.split("/")[-1] if media else "octet-stream") or "octet-stream"
        name = _resolve_media_name(part, src, fmt, url, "document")
        return {"format": fmt.lower(), "name": name, "source": {"bytes": data}}
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
    logger = logging.getLogger("proxy")
    logger.debug(f"map_messages_to_bedrock called with {len(anthropic_messages or [])} messages")

    # Log input message structure for debugging
    for i, msg in enumerate(anthropic_messages or []):
        role = msg.get("role", "unknown")
        content = msg.get("content", [])
        if isinstance(content, list):
            content_types = [c.get("type") if isinstance(c, dict) else str(type(c)) for c in content]
            tool_use_ids = [c.get("id") for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
            tool_result_ids = [c.get("tool_use_id") for c in content if isinstance(c, dict) and c.get("type") == "tool_result"]
            logger.debug(f"  INPUT Message {i} ({role}): content_types={content_types}, tool_use_ids={tool_use_ids}, tool_result_ids={tool_result_ids}")
        else:
            logger.debug(f"  INPUT Message {i} ({role}): content={type(content)} (non-list)")

    out: List[Dict[str, Any]] = []
    pending_tool_use_ids: List[str] = []

    for message in anthropic_messages or []:
        role = map_role(message.get("role", "user"))
        content = message.get("content", [])

        # Track tool_use_ids from current message BEFORE checking for consecutive assistant logic
        current_message_tool_use_ids: List[str] = []
        if role == "assistant":
            # Pre-scan for tool_use blocks to get the IDs before we process the message
            for part in content or []:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    tool_use_id = part.get("id")
                    if tool_use_id:
                        current_message_tool_use_ids.append(tool_use_id)

        # Check for consecutive assistant messages, but only log - don't fix here
        # The final check at the end will handle any unresolved tool_use_ids
        if pending_tool_use_ids and role == "assistant":
            logger.debug(f"Consecutive assistant message detected with pending tool_use_ids: {pending_tool_use_ids}")
            logger.debug("Will handle this at the end of message processing")

        has_text = False
        has_document = False
        tool_use_ids_in_message: List[str] = []
        tool_result_entries: List[Tuple[int, Dict[str, Any], Optional[str]]] = []
        other_entries: List[Tuple[int, Dict[str, Any]]] = []

        content_items = normalize_content_array(message.get("content"))
        for idx, part in enumerate(content_items):
            t = part.get("type")
            if t == "text":
                has_text = True
                block = {"text": part.get("text", "")}
                other_entries.append((idx, block))
            elif t == "image":
                block = {"image": to_bedrock_image(part, allow_image_url)}
                other_entries.append((idx, block))
            elif t == "document":
                document_payload = to_bedrock_document(part, allow_image_url)
                if tool_result_entries and tool_result_entries[-1][0] <= idx:
                    _, tr_block, _ = tool_result_entries[-1]
                    tr_content = tr_block["toolResult"].setdefault("content", [])
                    tr_content.append({"json": {"document": document_payload}})
                else:
                    has_document = True
                    other_entries.append((idx, {"document": document_payload}))
            elif t == "tool_use":
                tool_use_id = part.get("id")
                if tool_use_id:
                    tool_use_ids_in_message.append(tool_use_id)
                block = {
                    "toolUse": {
                        "toolUseId": tool_use_id,
                        "name": part.get("name"),
                        "input": part.get("input", {}),
                    }
                }
                other_entries.append((idx, block))
            elif t == "tool_result":
                tool_use_id = part.get("tool_use_id")
                block = {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": to_bedrock_tool_result_content(part.get("content")),
                        "status": "error" if part.get("is_error") else "success",
                    }
                }
                tool_result_entries.append((idx, block, tool_use_id))
            else:
                raise UnsupportedContentError(f"unsupported content type: {t}")

        ordered_tool_results: List[Dict[str, Any]] = []
        if tool_result_entries:
            if role != "assistant" and pending_tool_use_ids:
                def _sort_key(item: Tuple[int, Dict[str, Any], Optional[str]]) -> Tuple[int, int]:
                    original_index, block, tool_use_id = item
                    if tool_use_id in pending_tool_use_ids:
                        return (pending_tool_use_ids.index(tool_use_id), original_index)
                    return (len(pending_tool_use_ids) + original_index, original_index)

                ordered_tool_results = [block for _, block, _ in sorted(tool_result_entries, key=_sort_key)]
            else:
                ordered_tool_results = [block for _, block, _ in tool_result_entries]

        ordered_non_tool = [block for _, block in sorted(other_entries, key=lambda item: item[0])]
        content_blocks: List[Dict[str, Any]] = ordered_tool_results + ordered_non_tool

        if has_document and not has_text:
            placeholder = {"text": "Document attached."}
            insert_index = len(ordered_tool_results)
            for offset, block in enumerate(ordered_non_tool):
                if "document" in block:
                    insert_index = len(ordered_tool_results) + offset
                    break
            content_blocks.insert(insert_index, placeholder)

        emitted_messages: List[Dict[str, Any]] = []
        if role == "user" and ordered_tool_results and pending_tool_use_ids:
            for block in ordered_tool_results:
                emitted_messages.append({"role": role, "content": [block]})
            remaining = content_blocks[len(ordered_tool_results):]
            if remaining:
                emitted_messages.append({"role": role, "content": remaining})
        else:
            emitted_messages.append({"role": role, "content": content_blocks})

        out.extend(emitted_messages)

        if role == "assistant":
            pending_tool_use_ids.extend(tool_use_ids_in_message)
        else:
            for emitted in emitted_messages:
                tool_result_ids = [
                    (block.get("toolResult") or {}).get("toolUseId")
                    for block in emitted.get("content", [])
                    if "toolResult" in block
                ]

                if not tool_result_ids:
                    continue

                for tool_result_id in tool_result_ids:
                    if not tool_result_id:
                        continue
                    if pending_tool_use_ids and tool_result_id == pending_tool_use_ids[0]:
                        pending_tool_use_ids.pop(0)
                    elif tool_result_id in pending_tool_use_ids:
                        pending_tool_use_ids.remove(tool_result_id)

                if not pending_tool_use_ids:
                    continue

            if pending_tool_use_ids:
                raise MappingError(
                    "tool_result required for tool_use ids: {ids}".format(
                        ids=", ".join(pending_tool_use_ids)
                    )
                )

    # CRITICAL FIX: Pre-process all messages to split mixed content BEFORE any validation
    # BUT NEVER split user messages (always creates consecutive users - Bedrock rejects this)
    logger.debug("PRE-PROCESSING: Splitting mixed content messages to ensure Bedrock compatibility")

    i = 0
    while i < len(out):
        message = out[i]
        content = message.get("content", [])

        if not isinstance(content, list) or len(content) <= 1:
            i += 1
            continue

        # Check if this message has mixed content (tool_result + other content types)
        tool_result_blocks = []
        other_blocks = []

        for block in content:
            if isinstance(block, dict) and "toolResult" in block:
                tool_result_blocks.append(block)
            else:
                other_blocks.append(block)

        # If we have both tool_result blocks AND other blocks, check if we should split them
        if tool_result_blocks and other_blocks:
            # CRITICAL: Never split user messages as this ALWAYS creates consecutive user messages
            # Bedrock requires strict alternating user/assistant pattern
            if message.get("role") == "user":
                logger.debug(f"KEEPING mixed content at index {i} - cannot split user messages (creates consecutive users): {len(tool_result_blocks)} tool_results + {len(other_blocks)} other blocks")
                # Don't split - but ensure tool_result blocks come first for better Bedrock parsing
                message["content"] = tool_result_blocks + other_blocks
                i += 1
            else:
                logger.debug(f"SPLITTING mixed content message at index {i}: {len(tool_result_blocks)} tool_results + {len(other_blocks)} other blocks")

                # First message: only tool_results
                message["content"] = tool_result_blocks

                # Second message: other content (documents, text, etc.)
                other_message = {
                    "role": message["role"],
                    "content": other_blocks
                }
                out.insert(i + 1, other_message)

                # Skip both messages since we've processed them
                i += 2
        else:
            i += 1

    # CRITICAL FIX 2: Merge consecutive user messages to avoid user-user sequences
    # BUT only merge if neither message contains tool_result blocks (to preserve mixed content splitting)
    logger.debug("POST-PROCESSING: Merging consecutive user messages to ensure proper conversation flow")

    def has_tool_result(content):
        """Check if message content contains any tool_result blocks"""
        if not isinstance(content, list):
            return False
        return any(isinstance(block, dict) and "toolResult" in block for block in content)

    i = 0
    while i < len(out) - 1:
        current_msg = out[i]
        next_msg = out[i + 1]

        # If both current and next messages are from user, check if we should merge them
        if (current_msg.get("role") == "user" and next_msg.get("role") == "user"):
            current_content = current_msg.get("content", [])
            next_content = next_msg.get("content", [])

            # NEVER merge if either message contains tool_result blocks
            # This preserves our mixed content splitting
            if has_tool_result(current_content) or has_tool_result(next_content):
                logger.debug(f"SKIPPING merge of user messages at indices {i} and {i+1} - contains tool_result blocks")
                i += 1
                continue

            logger.debug(f"MERGING consecutive user messages at indices {i} and {i+1}")

            # Ensure both are lists
            if not isinstance(current_content, list):
                current_content = [current_content] if current_content else []
            if not isinstance(next_content, list):
                next_content = [next_content] if next_content else []

            # Merge the content
            current_msg["content"] = current_content + next_content

            # Remove the next message
            out.pop(i + 1)

            # Don't increment i, check the same position again in case of multiple consecutive user messages
        else:
            i += 1

    # Enhanced final check: verify all tool_use blocks have corresponding tool_results
    # This is needed because mixed content splitting can disrupt the tool_use tracking
    all_tool_use_ids: Set[str] = set()
    all_tool_result_ids: Set[str] = set()

    # Collect all tool_use and tool_result IDs from the final message list
    for msg in out:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "toolUse" in block:
                        tool_use_id = block.get("toolUse", {}).get("toolUseId")
                        if tool_use_id:
                            all_tool_use_ids.add(tool_use_id)
                    elif "toolResult" in block:
                        tool_result_id = block.get("toolResult", {}).get("toolUseId")
                        if tool_result_id:
                            all_tool_result_ids.add(tool_result_id)

    # Find unresolved tool_use_ids
    unresolved_tool_use_ids = all_tool_use_ids - all_tool_result_ids

    logger.debug(f"Enhanced final check: all_tool_use_ids={sorted(all_tool_use_ids)}, all_tool_result_ids={sorted(all_tool_result_ids)}, unresolved={sorted(unresolved_tool_use_ids)}")

    if unresolved_tool_use_ids:
        logger.debug(f"Final check: unresolved_tool_use_ids = {sorted(unresolved_tool_use_ids)}")

        # Create synthetic tool_result blocks for any unresolved tool_use blocks
        synthetic_content: List[Dict[str, Any]] = []
        for tool_use_id in sorted(unresolved_tool_use_ids):
            logger.debug(f"Creating synthetic tool_result for unresolved {tool_use_id}")
            synthetic_content.append({
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": "Tool execution pending..."}],
                    "status": "success",
                }
            })

        if synthetic_content:
            # Add synthetic user message with placeholder tool results
            synthetic_message = {
                "role": "user",
                "content": synthetic_content
            }
            out.append(synthetic_message)
            logger.debug(f"Added final synthetic user message with {len(synthetic_content)} tool_results")

    # Final verification - check if there are still any unresolved tool_use_ids
    final_all_tool_use_ids: Set[str] = set()
    final_all_tool_result_ids: Set[str] = set()

    for msg in out:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "toolUse" in block:
                        tool_use_id = block.get("toolUse", {}).get("toolUseId")
                        if tool_use_id:
                            final_all_tool_use_ids.add(tool_use_id)
                    elif "toolResult" in block:
                        tool_result_id = block.get("toolResult", {}).get("toolUseId")
                        if tool_result_id:
                            final_all_tool_result_ids.add(tool_result_id)

    final_unresolved = final_all_tool_use_ids - final_all_tool_result_ids
    if final_unresolved:
        raise MappingError(
            "tool_result required for tool_use ids: {ids}".format(
                ids=", ".join(sorted(final_unresolved))
            )
        )

    # Log output message structure for debugging
    logger.debug(f"map_messages_to_bedrock returning {len(out)} messages")
    for i, msg in enumerate(out):
        role = msg.get("role", "unknown")
        content = msg.get("content", [])
        if isinstance(content, list):
            content_types = [list(c.keys()) if isinstance(c, dict) else str(type(c)) for c in content]
            tool_use_ids = [c.get("toolUse", {}).get("toolUseId") for c in content if isinstance(c, dict) and "toolUse" in c]
            tool_result_ids = [c.get("toolResult", {}).get("toolUseId") for c in content if isinstance(c, dict) and "toolResult" in c]
            logger.debug(f"  OUTPUT Message {i} ({role}): content_types={content_types}, tool_use_ids={tool_use_ids}, tool_result_ids={tool_result_ids}")
        else:
            logger.debug(f"  OUTPUT Message {i} ({role}): content={type(content)} (non-list)")

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
    if choice is None:
        return {"auto": {}}

    if isinstance(choice, str):
        lowered = choice.lower()
        if lowered == "auto":
            return {"auto": {}}
        if lowered == "none":
            return {"none": {}}
        if lowered == "any":
            return {"any": {}}
        raise MappingError(f"unsupported tool_choice value: {choice}")

    if isinstance(choice, dict):
        choice_type_raw = choice.get("type")
        choice_type = choice_type_raw.lower() if isinstance(choice_type_raw, str) else ""
        if not choice_type and choice.get("name"):
            choice_type = "tool"

        if choice_type == "auto":
            return {"auto": {}}
        if choice_type == "none":
            return {"none": {}}
        if choice_type == "any":
            return {"any": {}}
        if choice_type == "tool":
            name = choice.get("name")
            if not name:
                raise MappingError("tool_choice of type 'tool' requires a name")
            return {"tool": {"name": name}}
        raise MappingError(f"unsupported tool_choice type: {choice_type_raw}")

    raise MappingError("tool_choice must be a string or dict when provided")


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
    if body.get("top_k") is not None:
        cfg["topK"] = body.get("top_k")
    if body.get("presence_penalty") is not None:
        cfg["presencePenalty"] = body.get("presence_penalty")
    if body.get("frequency_penalty") is not None:
        cfg["frequencyPenalty"] = body.get("frequency_penalty")
    if body.get("max_output_tokens") is not None:
        cfg["maxOutputTokens"] = body.get("max_output_tokens")
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
