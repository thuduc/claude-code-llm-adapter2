import pytest

from src.config import get_settings, map_model_id


def _reset_settings_cache() -> None:
    try:
        get_settings.cache_clear()
    except AttributeError:  # pragma: no cover - defensive
        pass


def test_map_model_id_haiku_fallback(monkeypatch):
    monkeypatch.delenv("MODEL_ID_MAP_JSON", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("MODEL_ID_MAP_JSON", "")
    _reset_settings_cache()
    try:
        mapped = map_model_id("claude-3-haiku-20240307")
        assert mapped == "anthropic.claude-3-haiku-20240307-v1:0"
    finally:
        _reset_settings_cache()


def test_map_model_id_unknown_model(monkeypatch):
    monkeypatch.setenv("MODEL_ID_MAP_JSON", "")
    _reset_settings_cache()
    try:
        with pytest.raises(ValueError):
            map_model_id("claude-foo-bar")
    finally:
        _reset_settings_cache()
