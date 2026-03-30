"""Provider adapters for OpenAI-compatible chat completions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key_env: str
    base_url: str | None = None


PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(name="openai", api_key_env="OPENAI_API_KEY"),
    "openrouter": ProviderConfig(
        name="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
}


def provider_env_var(provider: str) -> str:
    """Return the environment variable required by a provider."""

    return _provider_config(provider).api_key_env


def has_api_key(provider: str) -> bool:
    """Report whether a provider API key is available."""

    return bool(os.getenv(provider_env_var(provider)))


def make_model_caller(provider: str, model: str) -> Callable[[str, str], str]:
    """Create a callable that submits prompts to the selected provider."""

    return _OpenAICompatibleCaller(provider=provider, model=model)


def _provider_config(provider: str) -> ProviderConfig:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {available}") from exc


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                chunks.append(str(item.text))
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    return str(content).strip()


class _OpenAICompatibleCaller:
    """Thin adapter around the OpenAI client for OpenAI-compatible providers."""

    def __init__(self, provider: str, model: str) -> None:
        config = _provider_config(provider)
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"{config.api_key_env} is not set. "
                f"Set it before running OpenXRD with provider '{provider}'."
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
        if config.base_url:
            base_url = os.getenv("OPENROUTER_BASE_URL", config.base_url)
            client_kwargs["base_url"] = base_url

        self.provider = provider
        self.model = model
        self.client = OpenAI(**client_kwargs)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def __call__(self, prompt: str, system_prompt: str) -> str:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        if self.provider == "openrouter":
            params["extra_headers"] = {
                "HTTP-Referer": os.getenv(
                    "OPENXRD_REFERER_URL",
                    "https://github.com/niaz60/OpenXRD",
                ),
                "X-Title": "OpenXRD",
            }

        response = self.client.chat.completions.create(**params)
        return _content_to_text(response.choices[0].message.content)
