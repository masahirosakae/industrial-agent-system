import json
import os
import urllib.error
import urllib.request

from src.llm.base import LLMProvider, LLMResponse


_DEFAULT_TIMEOUT = object()


class FuguProviderError(RuntimeError):
    """Raised when the Fugu API request or response is invalid."""


class FuguProvider(LLMProvider):
    def __init__(
        self,
        timeout: int | None | object = _DEFAULT_TIMEOUT,
    ):
        self.api_key = os.getenv("FUGU_API_KEY")
        self.base_url = os.getenv("FUGU_BASE_URL")
        self.model = os.getenv("FUGU_MODEL")
        self.timeout = (
            int(os.getenv("FUGU_TIMEOUT_SECONDS", "180"))
            if timeout is _DEFAULT_TIMEOUT
            else timeout
        )

        if not self.api_key:
            raise ValueError("FUGU_API_KEY is not set")
        if not self.base_url:
            raise ValueError("FUGU_BASE_URL is not set")
        if not self.model:
            raise ValueError("FUGU_MODEL is not set")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = self._read_http_error_detail(error)
            raise FuguProviderError(
                f"Fugu API request failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise FuguProviderError(
                f"Fugu API request failed: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise FuguProviderError("Fugu API request timed out") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FuguProviderError("Fugu API returned invalid JSON") from error

        if not isinstance(data, dict):
            raise FuguProviderError("Fugu API response must be a JSON object")

        if "error" in data:
            raise FuguProviderError(
                f"Fugu API returned an error: {self._format_api_error(data['error'])}"
            )

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise FuguProviderError(
                "Fugu API response is missing choices[0].message.content"
            ) from error

        if not isinstance(text, str):
            raise FuguProviderError(
                "Fugu API response choices[0].message.content must be a string"
            )

        return LLMResponse(
            text=text,
            model=self.model,
            provider="fugu",
        )

    @staticmethod
    def _read_http_error_detail(error: urllib.error.HTTPError) -> str:
        try:
            data = json.loads(error.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return error.reason

        if isinstance(data, dict):
            data = data.get("error", data)

        return FuguProvider._format_api_error(data)

    @staticmethod
    def _format_api_error(error) -> str:
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error)
