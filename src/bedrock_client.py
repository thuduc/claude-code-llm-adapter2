from typing import Any, Dict, Optional, List


class BedrockUnavailable(RuntimeError):
    pass


def get_client(region: str):
    try:
        import boto3  # type: ignore
    except Exception as e:  # pragma: no cover - import-time guard
        raise BedrockUnavailable("boto3 is required to use Bedrock") from e
    return boto3.client("bedrock-runtime", region_name=region)


def converse(client, **kwargs) -> Dict[str, Any]:
    """Thin wrapper over client.converse.

    kwargs should include keys like modelId, messages, system, toolConfig, inferenceConfig,
    additionalModelRequestFields, guardrailConfig, etc.
    """
    return client.converse(**{k: v for k, v in kwargs.items() if v is not None})


def converse_stream(client, **kwargs) -> Dict[str, Any]:
    """Thin wrapper over client.converse_stream.

    Returns provider response containing 'stream'. Caller is responsible for iterating it.
    """
    return client.converse_stream(**{k: v for k, v in kwargs.items() if v is not None})


def list_models(region: str) -> List[str]:
    try:
        import boto3  # type: ignore
    except Exception as e:  # pragma: no cover
        raise BedrockUnavailable("boto3 is required to use Bedrock") from e
    ctl = boto3.client("bedrock", region_name=region)
    try:
        resp = ctl.list_foundation_models()
    except Exception:
        return []
    models = resp.get("modelSummaries") or []
    ids: List[str] = []
    for m in models:
        mid = m.get("modelId")
        if isinstance(mid, str):
            ids.append(mid)
    return ids
