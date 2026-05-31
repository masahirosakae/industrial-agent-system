import json
import urllib.error

import pytest

from src.llm.base import LLMProvider, LLMResponse
from src.llm.factory import create_llm_provider
from src.llm.fugu_provider import FuguProvider, FuguProviderError
from src.llm.ollama_provider import OllamaProvider


class FakeResponse:
    def __init__(self, data: dict):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.data).encode("utf-8")

    def close(self) -> None:
        pass


def test_ollama_provider_reproduces_generate_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse({"response": "generated text"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = OllamaProvider().generate("hello")

    payload = json.loads(captured["request"].data.decode("utf-8"))
    assert payload == {
        "model": "qwen2.5:1.5b",
        "prompt": "hello",
        "stream": False,
    }
    assert response == LLMResponse(
        text="generated text",
        model="qwen2.5:1.5b",
        provider="ollama",
    )


def test_factory_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    provider = create_llm_provider()

    assert isinstance(provider, LLMProvider)
    assert isinstance(provider, OllamaProvider)


def test_factory_selects_fugu_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fugu")
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")

    provider = create_llm_provider(ollama_model="local-only")

    assert isinstance(provider, FuguProvider)
    assert provider.model == "test-fugu"


def test_fugu_provider_uses_environment_variables(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "fugu text"}}]})

    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = FuguProvider().generate("hello")

    request = captured["request"]
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert response == LLMResponse(
        text="fugu text",
        model="test-fugu",
        provider="fugu",
    )


def test_fugu_provider_requires_environment_variables(monkeypatch):
    monkeypatch.delenv("FUGU_API_KEY", raising=False)
    monkeypatch.delenv("FUGU_BASE_URL", raising=False)
    monkeypatch.delenv("FUGU_MODEL", raising=False)

    with pytest.raises(ValueError, match="FUGU_API_KEY"):
        FuguProvider()


def test_fugu_provider_requires_model(monkeypatch):
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.delenv("FUGU_MODEL", raising=False)

    with pytest.raises(ValueError, match="FUGU_MODEL"):
        FuguProvider()


def test_fugu_provider_reports_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            FakeResponse({"error": {"message": "invalid API key"}}),
        )

    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(FuguProviderError, match="HTTP 401: invalid API key"):
        FuguProvider().generate("hello")


def test_fugu_provider_reports_api_error_response(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse({"error": {"message": "rate limit exceeded"}})

    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(FuguProviderError, match="rate limit exceeded"):
        FuguProvider().generate("hello")


def test_fugu_provider_rejects_invalid_response_shape(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse({"choices": []})

    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(FuguProviderError, match=r"choices\[0\]"):
        FuguProvider().generate("hello")
