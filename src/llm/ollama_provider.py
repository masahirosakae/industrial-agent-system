import json
import urllib.request

from src.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: int | None = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        return LLMResponse(
            text=data["response"],
            model=self.model,
            provider="ollama",
        )
