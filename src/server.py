import asyncio
import uuid
import logging
import json
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from .bedrock_client import BedrockUnavailable, converse, converse_stream, get_client
from .config import get_settings, map_model_id
from .mapping import (
    collect_additional_fields,
    map_bedrock_message_to_anthropic,
    map_inference_config,
    map_messages_to_bedrock,
    map_stop_reason,
    map_system_to_bedrock,
    map_tool_choice,
    map_tools_to_bedrock,
    from_bedrock_tool_result_content,
    MappingError,
)
from .sse import sse


app = FastAPI(title="Claude Code → Bedrock Proxy", version="0.1.0")


# ---- Logging setup ----
logger = logging.getLogger("proxy")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler("proxy_log.txt", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.debug("Done setting up console and file logging")


def _sanitize_for_log(value: Any, max_len: int = 200) -> Any:
    # Recursively sanitize values for logging to avoid huge blobs and binary dumps
    try:
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, bytes):
            return f"<bytes len={len(value)}>"
        if isinstance(value, str):
            if len(value) > max_len:
                return value[:max_len] + f"...<truncated len={len(value)}>"
            return value
        if isinstance(value, list):
            return [_sanitize_for_log(v, max_len) for v in value]
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                # Special-case: image sources and content
                if k == "source" and isinstance(v, dict):
                    src_type = v.get("type")
                    if src_type == "base64":
                        data = v.get("data")
                        if isinstance(data, str):
                            out[k] = {
                                **{kk: vv for kk, vv in v.items() if kk != "data"},
                                "data": f"<base64 len={len(data)}>",
                            }
                            continue
                out[k] = _sanitize_for_log(v, max_len)
            return out
        # Fallback to string
        s = str(value)
        return s[:max_len] + (f"...<truncated len={len(s)}>" if len(s) > max_len else "")
    except Exception:
        return "<unloggable>"


def _require_auth(x_api_key: Optional[str] = Header(default=None, alias="x-api-key")) -> None:
    settings = get_settings()
    if settings.proxy_api_key and x_api_key != settings.proxy_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key",
                },
            },
        )


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


def _anthropic_error(status_code: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


def _gen_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _build_bedrock_args(body: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    try:
        messages = map_messages_to_bedrock(body.get("messages", []), settings.allow_image_url_fetch)
        system = map_system_to_bedrock(body.get("system"))
        tool_cfg = map_tools_to_bedrock(body.get("tools"))
        if tool_cfg is not None:
            tc = map_tool_choice(body.get("tool_choice"))
            # Only include toolChoice when tools exist
            if tc is not None:
                tool_cfg["toolChoice"] = tc
        inference_cfg = map_inference_config(body)
        additional = collect_additional_fields(body)
    except MappingError as e:
        logger.error(f"MappingError error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    model_id = map_model_id(body.get("model"))
    args = {
        "modelId": model_id,
        "messages": messages,
        "system": system,
        "toolConfig": tool_cfg,
        "inferenceConfig": inference_cfg or None,
        "additionalModelRequestFields": additional,
    }
    logger.debug(
        "Built Bedrock args: %s",
        json.dumps(_sanitize_for_log(args), ensure_ascii=False),
    )
    return args


@app.post("/v1/messages")
def create_message(
    body: Dict[str, Any],
    response: Response,
    x_api_key: Optional[str] = Depends(_require_auth),
):
    # Non-streaming and streaming supported based on body.stream
    stream = bool(body.get("stream"))
    try:
        logger.debug(
            "Incoming /v1/messages stream=%s body=%s",
            stream,
            json.dumps(_sanitize_for_log(body), ensure_ascii=False),
        )
    except Exception:
        logger.error("Failed to log incoming request body")
    try:
        args = _build_bedrock_args(body)
    except HTTPException as e:
        logger.error(f"HTTPException error: {e.status_code} - {e.detail}")
        return _anthropic_error(e.status_code, str(e.detail))

    # Prepare Bedrock client
    try:
        client = get_client(get_settings().aws_region)
    except BedrockUnavailable as e:
        logger.error(f"BedrockUnavailable error: {str(e)}")
        return _anthropic_error(500, str(e), err_type="provider_error")

    if stream:
        return _streaming_response(client, body, args)
    else:
        return _non_streaming_response(client, body, args)


def _non_streaming_response(client, body: Dict[str, Any], args: Dict[str, Any]) -> JSONResponse:
    try:
        logger.info("Calling Bedrock converse: modelId=%s", args.get("modelId"))
        resp = converse(client, **args)
    except Exception as e:  # boto errors
        logger.error(f"Bedrock Converse error: {e}")
        return _anthropic_error(502, f"Bedrock error: {e}", err_type="provider_error")

    out_msg = resp.get("output", {}).get("message", {})
    stop_reason = map_stop_reason(resp.get("stopReason"))
    usage = resp.get("usage", {})
    mapped = map_bedrock_message_to_anthropic(out_msg)

    result = {
        "id": _gen_msg_id(),
        "type": "message",
        "model": body.get("model"),
        "role": mapped.get("role", "assistant"),
        "content": mapped.get("content", []),
        "stop_reason": stop_reason,
        "stop_sequence": None,  # Filled if provider supplies later
        "usage": {
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
        },
    }
    logger.debug(
        "Bedrock response (non-stream): %s",
        json.dumps(_sanitize_for_log(resp), ensure_ascii=False),
    )
    logger.debug(
        "Responding Anthropic message: %s",
        json.dumps(_sanitize_for_log(result), ensure_ascii=False),
    )
    return JSONResponse(status_code=200, content=result)


def _streaming_response(client, body: Dict[str, Any], args: Dict[str, Any]) -> StreamingResponse:
    async def gen():
        # message_start
        msg_id = _gen_msg_id()
        line = sse(
            "message_start",
            {"type": "message", "id": msg_id, "model": body.get("model"), "role": "assistant"},
        )
        logger.debug("SSE> %s", line.strip())
        yield line
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        stop_reason: Optional[str] = None

        # Start Bedrock stream call in a thread to avoid blocking event loop
        loop = asyncio.get_running_loop()

        def _start_stream():
            try:
                return converse_stream(client, **args)
            except Exception as exc:  # pragma: no cover - provider path
                logger.error(f"Bedrock Converse stream error: {exc}")
                return exc

        logger.info("Calling Bedrock converse_stream: modelId=%s", args.get("modelId"))
        resp = await loop.run_in_executor(None, _start_stream)
        if isinstance(resp, Exception):
            line = sse(
                "error",
                {"error": {"type": "provider_error", "message": f"Bedrock error: {resp}"}},
            )
            logger.debug("SSE> %s", line.strip())
            yield line
            return

        stream = resp.get("stream")
        try:
            # Iterate provider event stream in thread
            def _iter_stream():
                try:
                    for evt in stream:
                        yield evt
                finally:
                    # Ensure provider stream is closed
                    try:
                        stream.close()
                    except Exception:
                        pass

            it = _iter_stream()
            while True:
                evt = await loop.run_in_executor(None, lambda: next(it, None))
                if evt is None:
                    break
                # AWS Bedrock streams send a dict with exactly one key indicating type
                try:
                    if "messageStart" in evt:
                        # Optionally available; no SSE needed besides message_start we already sent
                        pass
                    elif "contentBlockStart" in evt:
                        cbs = evt["contentBlockStart"]
                        index = cbs.get("index", 0)
                        start = cbs.get("start") or cbs.get("contentBlock") or {}
                        if "text" in start:
                            text_value = start.get("text")
                            if isinstance(text_value, dict):
                                text_value = text_value.get("text", "")
                            payload = {"type": "text", "text": text_value or ""}
                            line = sse(
                                "content_block_start",
                                {"index": index, "content_block": payload},
                            )
                            logger.debug("SSE> %s", line.strip())
                            yield line
                        elif "toolUse" in start:
                            tu = start.get("toolUse", {})
                            payload = {
                                "type": "tool_use",
                                "id": tu.get("toolUseId"),
                                "name": tu.get("name"),
                                "input": tu.get("input") or {},
                            }
                            line = sse(
                                "content_block_start",
                                {"index": index, "content_block": payload},
                            )
                            logger.debug("SSE> %s", line.strip())
                            yield line
                        elif "toolResult" in start:
                            tr = start.get("toolResult", {})
                            content_field = tr.get("content")
                            if isinstance(content_field, list):
                                mapped_content = from_bedrock_tool_result_content(content_field)
                            elif content_field is None:
                                mapped_content = ""
                            else:
                                mapped_content = content_field
                            payload = {
                                "type": "tool_result",
                                "tool_use_id": tr.get("toolUseId"),
                                "is_error": (tr.get("status") == "error"),
                                "content": mapped_content,
                            }
                            line = sse(
                                "content_block_start",
                                {"index": index, "content_block": payload},
                            )
                            logger.debug("SSE> %s", line.strip())
                            yield line
                    elif "contentBlockDelta" in evt:
                        cbd = evt["contentBlockDelta"]
                        index = cbd.get("index", 0)
                        delta = cbd.get("delta", {})
                        if "text" in delta:
                            line = sse(
                                "content_block_delta",
                                {"index": index, "delta": {"type": "text_delta", "text": delta.get("text", "")}},
                            )
                            logger.debug("SSE> %s", line.strip())
                            yield line
                        elif "toolUse" in delta:
                            tu = delta.get("toolUse", {})
                            # Bedrock may send partial JSON for the tool input
                            # Try a few potential field names
                            partial = (
                                tu.get("input", {}).get("partialJson")
                                or tu.get("input", {}).get("json")
                                or tu.get("partialJson")
                                or ""
                            )
                            line = sse(
                                "content_block_delta",
                                {"index": index, "delta": {"type": "input_json_delta", "partial_json": partial}},
                            )
                            logger.debug("SSE> %s", line.strip())
                            yield line
                        elif "toolResult" in delta:
                            tr = delta.get("toolResult", {})
                            partial = None
                            content_field = tr.get("content")
                            if isinstance(content_field, dict):
                                partial = content_field.get("partialJson")
                            if partial is None:
                                partial = tr.get("partialJson")
                            if partial is not None:
                                line = sse(
                                    "content_block_delta",
                                    {"index": index, "delta": {"type": "output_json_delta", "partial_json": partial}},
                                )
                                logger.debug("SSE> %s", line.strip())
                                yield line
                            else:
                                # Try text content list format
                                if isinstance(content_field, list) and content_field:
                                    first = content_field[0]
                                    if isinstance(first, dict) and "text" in first:
                                        line = sse(
                                            "content_block_delta",
                                            {"index": index, "delta": {"type": "text_delta", "text": first.get("text", "")}},
                                        )
                                        logger.debug("SSE> %s", line.strip())
                                        yield line
                    elif "contentBlockStop" in evt:
                        cbstop = evt["contentBlockStop"]
                        index = cbstop.get("index", 0)
                        line = sse("content_block_stop", {"index": index})
                        logger.debug("SSE> %s", line.strip())
                        yield line
                    elif "messageStop" in evt:
                        ms = evt["messageStop"]
                        stop_reason = map_stop_reason(ms.get("stopReason"))
                        if stop_reason is not None or (input_tokens is not None or output_tokens is not None):
                            delta_payload: Dict[str, Any] = {}
                            if stop_reason is not None:
                                delta_payload["stop_reason"] = stop_reason
                                delta_payload["stop_sequence"] = None
                            if input_tokens is not None or output_tokens is not None:
                                delta_payload["usage"] = {
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                }
                            if delta_payload:
                                line = sse("message_delta", delta_payload)
                                logger.debug("SSE> %s", line.strip())
                                yield line
                        line = sse("message_stop", {})
                        logger.debug("SSE> %s", line.strip())
                        yield line
                    elif "metadata" in evt:
                        md = evt["metadata"] or {}
                        usage = md.get("usage") or {}
                        input_tokens = usage.get("inputTokens", input_tokens)
                        output_tokens = usage.get("outputTokens", output_tokens)
                    else:
                        # Unknown chunk – ignore
                        pass
                except Exception as parse_exc:
                    logger.error(f"Stream response parse chunk error: {parse_exc}")
                    # If we cannot parse a chunk, emit error and stop
                    line = sse(
                        "error",
                        {"error": {"type": "stream_parse_error", "message": str(parse_exc)}},
                    )
                    logger.debug("SSE> %s", line.strip())
                    yield line
                    break
        except Exception as e:
            logger.error(f"Stream response parse error: {e}")
            line = sse("error", {"error": {"type": "provider_error", "message": str(e)}})
            logger.debug("SSE> %s", line.strip())
            yield line
        finally:
            # Ensure final message_stop if not already sent? Anthropic requires termination
            pass

    return StreamingResponse(gen(), media_type="text/event-stream")
