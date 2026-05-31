import os

from src.llm.base import LLMProvider
from src.llm.fugu_provider import FuguProvider
from src.llm.ollama_provider import OllamaProvider


def create_llm_provider(
    provider_name: str | None = None,
    ollama_model: str | None = None,
    **kwargs,
) -> LLMProvider:
    name = (provider_name or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()

    if name == "ollama":
        if ollama_model is not None:
            kwargs.setdefault("model", ollama_model)
        return OllamaProvider(**kwargs)
    if name == "fugu":
        return FuguProvider(**kwargs)

    raise ValueError(f"Unsupported LLM provider: {name}")
