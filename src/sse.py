import json
from typing import Any, AsyncGenerator, Dict


def _json_dumps_min(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\n" f"data: {_json_dumps_min(data)}\n\n"


async def ping_sse(interval_seconds: int = 15) -> AsyncGenerator[str, None]:
    import asyncio

    while True:
        yield "event: ping\n\n"
        await asyncio.sleep(interval_seconds)

