import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional


@dataclass
class Settings:
    aws_region: str
    proxy_api_key: Optional[str]
    allow_image_url_fetch: bool
    model_id_map: Dict[str, str]


def _load_model_id_map() -> Dict[str, str]:
    raw = os.getenv("MODEL_ID_MAP_JSON", "").strip()
    if not raw:
        return {}
    # Allow @path to load from file
    if raw.startswith("@"):
        path = raw[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Otherwise parse as JSON string
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try env var that is a single mapping like A=B;C=D
        mapping: Dict[str, str] = {}
        for pair in raw.split(";"):
            if not pair:
                continue
            if "=" in pair:
                k, v = pair.split("=", 1)
                mapping[k.strip()] = v.strip()
        return mapping


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        # Keep flexible; server may start without and fail per-request if needed.
        region = "us-east-1"
    proxy_api_key = os.getenv("PROXY_API_KEY")
    allow_image_url_fetch = os.getenv("ALLOW_IMAGE_URL_FETCH", "false").lower() in {"1", "true", "yes"}
    model_map = _load_model_id_map()
    return Settings(
        aws_region=region,
        proxy_api_key=proxy_api_key,
        allow_image_url_fetch=allow_image_url_fetch,
        model_id_map=model_map,
    )


def map_model_id(requested_model: str) -> str:
    """Map Anthropic model name to Bedrock modelId.
    """
    if not requested_model:
        raise ValueError("model is required")
    m = get_settings().model_id_map.get(requested_model)
    if m:
        return m

    normalized = requested_model.lower()
    if "sonnet" in normalized:
        return "us.anthropic.claude-sonnet-4-20250514-v1:0"
    if "haiku" in normalized:
        # Default to Claude 3 Haiku unless operator overrides via MODEL_ID_MAP_JSON
        return "anthropic.claude-3-haiku-20240307-v1:0"
    if "opus" in normalized:
        return "us.anthropic.claude-opus-4-20250514-v1:0"

    raise ValueError(
        "Unknown model name '{model}'. Configure MODEL_ID_MAP_JSON to map it to a Bedrock modelId (e.g., 'anthropic.claude-3-5-sonnet-20240620-v1:0').".format(
            model=requested_model
        )
    )
