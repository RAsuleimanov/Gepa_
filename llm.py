"""LLM adapters and config for GigaChat and OpenAI-compatible providers."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        payload = asdict(self)
        payload["total_tokens"] = self.total_tokens
        return payload

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

AGENT_ROLE = "agent"
ASSISTANT_ROLE = "assistant"


class ProviderType(str, Enum):
    OPENAI_COMPATIBLE = "openai_compatible"
    GIGACHAT = "gigachat"
    ANTHROPIC = "anthropic"


class RetryConfig(BaseModel):
    # Defaults are tuned for unstable networks:
    # waits between retries: 5, 10, 20, 40, 80 seconds.
    max_attempts: int = Field(default=6, ge=1)
    backoff_seconds: float = Field(default=5.0, ge=0.0)


class ModelProviderConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(min_length=1)
    type: ProviderType
    model: str = Field(min_length=1)
    base_url: HttpUrl
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    api_key_env: str | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)
    verify_ssl: bool = True
    oauth_scope: str | None = None
    client_id_env: str | None = None
    auth_key_env: str | None = None
    cert_file: Path | None = None
    key_file: Path | None = None
    extra_params: dict[str, object] = Field(default_factory=dict)

    @field_validator("cert_file", "key_file")
    @classmethod
    def validate_optional_existing_path(cls, value: Path | None) -> Path | None:
        if value is not None and not value.exists():
            raise ValueError(f"path does not exist: {value}")
        return value

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> ModelProviderConfig:
        if self.type == ProviderType.OPENAI_COMPATIBLE and not self.api_key_env:
            raise ValueError("openai_compatible provider requires api_key_env")
        if self.type == ProviderType.GIGACHAT:
            has_credentials = bool(self.auth_key_env)
            has_mtls = bool(self.cert_file and self.key_file)
            if not (has_credentials or has_mtls):
                raise ValueError(
                    "gigachat provider requires either auth_key_env "
                    "or cert_file + key_file"
                )
        return self


# ---------------------------------------------------------------------------
# Adapter interface and request/response
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GenerationRequest:
    system_prompt: str
    user_message: str | None = None
    history: list[dict[str, str]] | None = None
    temperature: float = 0.0
    metadata: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelResponse:
    provider_name: str
    model: str
    output_text: str
    raw_response: dict[str, Any]
    usage: TokenUsage | None = None


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> ModelResponse:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources. Override if the adapter holds connections."""


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter
# ---------------------------------------------------------------------------

class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(self, provider_config: ModelProviderConfig, top_p: float | None = None) -> None:
        from openai import OpenAI

        self.provider_config = provider_config
        self._top_p = top_p
        api_key = os.getenv(provider_config.api_key_env or "", "") or "no-key"
        # Keep timeout config-driven; no forced 5-minute lower bound.
        effective_timeout_seconds = provider_config.timeout_seconds
        self.client = OpenAI(
            base_url=str(provider_config.base_url),
            api_key=api_key,
            timeout=effective_timeout_seconds,
            # Keep SDK retries disabled: retry policy is controlled explicitly
            # by ModelProviderConfig.retry in generate().
            max_retries=0,
        )

    @staticmethod
    def _build_messages(request: GenerationRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": request.system_prompt}]
        if request.history:
            for item in request.history:
                role = ASSISTANT_ROLE if item["role"] == AGENT_ROLE else item["role"]
                messages.append({"role": role, "content": item["content"]})
        elif request.user_message is not None:
            messages.append({"role": "user", "content": request.user_message})
        return messages

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        # OpenAI SDK errors may expose either .status_code or .response.status_code.
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True

        msg = str(error).lower()
        return any(
            marker in msg
            for marker in (
                "429",
                "rate limit",
                "timed out",
                "timeout",
                "connection",
                "temporarily unavailable",
                "503",
                "502",
                "500",
                "504",
            )
        )

    def generate(self, request: GenerationRequest) -> ModelResponse:
        kwargs: dict[str, object] = {
            "model": self.provider_config.model,
            "temperature": request.temperature,
            "messages": self._build_messages(request),
        }
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self.provider_config.extra_params:
            extra = dict(self.provider_config.extra_params)
            # max_tokens is a top-level OpenAI parameter, not extra_body.
            if "max_tokens" in extra:
                kwargs["max_tokens"] = extra.pop("max_tokens")
            if extra:
                kwargs["extra_body"] = extra
        if request.response_format:
            kwargs["response_format"] = request.response_format

        if not hasattr(self, '_logged_first_call'):
            self._logged_first_call = True
            print(f"[OpenAI] first call: model={self.provider_config.model}, "
                  f"temperature={request.temperature}, "
                  f"top_p={self._top_p}, "
                  f"max_tokens={kwargs.get('max_tokens', 'N/A')}, "
                  f"extra_body={kwargs.get('extra_body', 'N/A')}")

        max_retries = self.provider_config.retry.max_attempts
        base_backoff = self.provider_config.retry.backoff_seconds
        completion = None
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
                break
            except Exception as e:
                last_error = e
                if self._is_retryable_error(e) and attempt < max_retries - 1:
                    wait = base_backoff * (2 ** attempt)
                    print(f"Rate limited, waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    break

        if completion is None:
            print(f"[ERROR] {self.provider_config.name}/{self.provider_config.model}: {last_error}")
            return ModelResponse(
                provider_name=self.provider_config.name,
                model=self.provider_config.model,
                output_text="",
                raw_response={
                    "error": str(last_error) if last_error else "openai_compatible request failed",
                    "provider": self.provider_config.name,
                    "model": self.provider_config.model,
                },
                usage=None,
            )

        usage = None
        if completion.usage:
            usage = TokenUsage(
                input_tokens=completion.usage.prompt_tokens or 0,
                output_tokens=completion.usage.completion_tokens or 0,
            )
        if not completion.choices:
            return ModelResponse(
                provider_name=self.provider_config.name,
                model=self.provider_config.model,
                output_text="",
                raw_response=completion.model_dump() if completion else {},
                usage=usage,
            )
        return ModelResponse(
            provider_name=self.provider_config.name,
            model=self.provider_config.model,
            output_text=completion.choices[0].message.content or "",
            raw_response=completion.model_dump(),
            usage=usage,
        )

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------------------
# GigaChat adapter
# ---------------------------------------------------------------------------

class GigaChatAdapter(ModelAdapter):
    def __init__(self, provider_config: ModelProviderConfig, top_p: float | None = None, temperature: float | None = None) -> None:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_gigachat.chat_models import GigaChat

        self.provider_config = provider_config
        self._top_p = top_p
        self._AIMessage = AIMessage
        self._HumanMessage = HumanMessage
        self._SystemMessage = SystemMessage

        kwargs: dict[str, object] = {
            "base_url": str(provider_config.base_url),
            "model": provider_config.model,
            "verify_ssl_certs": provider_config.verify_ssl,
            "timeout": provider_config.timeout_seconds,
            "streaming": False,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if temperature is not None:
            kwargs["temperature"] = temperature
        if provider_config.auth_key_env:
            kwargs["credentials"] = os.getenv(provider_config.auth_key_env, "")
        if provider_config.oauth_scope:
            kwargs["scope"] = provider_config.oauth_scope
        if provider_config.cert_file:
            kwargs["cert_file"] = str(provider_config.cert_file)
        if provider_config.key_file:
            kwargs["key_file"] = str(provider_config.key_file)
        kwargs.update(provider_config.extra_params)
        self.client = GigaChat(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code in {408, 429, 500, 502, 503, 504}:
            return True

        msg = str(error).lower()
        return any(
            marker in msg
            for marker in (
                "timeout",
                "timed out",
                "read operation timed out",
                "connection",
                "temporarily unavailable",
                "503",
                "502",
                "500",
                "504",
                "429",
                "rate limit",
            )
        )

    def generate(self, request: GenerationRequest) -> ModelResponse:
        messages = [self._SystemMessage(content=request.system_prompt)]
        if request.history:
            for item in request.history:
                if item["role"] in (AGENT_ROLE, ASSISTANT_ROLE):
                    messages.append(self._AIMessage(content=item["content"]))
                else:
                    messages.append(self._HumanMessage(content=item["content"]))
        elif request.user_message is not None:
            messages.append(self._HumanMessage(content=request.user_message))

        # Set temperature/top_p per-request (langchain GigaChat reads them from the client object).
        self.client.temperature = request.temperature
        if self._top_p is not None:
            self.client.top_p = self._top_p

        if not hasattr(self, '_logged_first_call'):
            self._logged_first_call = True
            print(f"[GigaChat] first call: model={self.client.model}, "
                  f"temperature={self.client.temperature} (type={type(self.client.temperature).__name__}), "
                  f"top_p={getattr(self.client, 'top_p', 'N/A')}, "
                  f"repetition_penalty={getattr(self.client, 'repetition_penalty', 'N/A')}")

        max_retries = self.provider_config.retry.max_attempts
        base_backoff = self.provider_config.retry.backoff_seconds
        total_usage = TokenUsage()
        response = None
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.client.invoke(messages)
            except Exception as exc:
                last_error = exc
                if self._is_retryable_error(exc) and attempt < max_retries - 1:
                    wait = base_backoff * (2 ** attempt)
                    print(
                        f"GigaChat request failed ({attempt + 1}/{max_retries}): {exc}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                break

            token_usage = response.response_metadata.get("token_usage", {})
            if token_usage:
                total_usage.add(TokenUsage(
                    input_tokens=token_usage.get("prompt_tokens", 0),
                    output_tokens=token_usage.get("completion_tokens", 0),
                ))
            finish_reason = response.response_metadata.get("finish_reason", "")
            if finish_reason == "blacklist" and attempt < max_retries - 1:
                print(f"GigaChat blacklist hit, retrying ({attempt + 1}/{max_retries})...")
                continue
            break

        if response is None:
            print(f"[ERROR] {self.provider_config.name}/{self.provider_config.model}: {last_error}")
            return ModelResponse(
                provider_name=self.provider_config.name,
                model=self.provider_config.model,
                output_text="",
                raw_response={
                    "error": str(last_error) if last_error else "gigachat request failed",
                    "provider": self.provider_config.name,
                    "model": self.provider_config.model,
                },
                usage=None,
            )

        return ModelResponse(
            provider_name=self.provider_config.name,
            model=self.provider_config.model,
            output_text=response.content if isinstance(response.content, str) else str(response.content),
            raw_response=response.response_metadata,
            usage=total_usage if total_usage.total_tokens > 0 else None,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_adapter(provider_config: ModelProviderConfig, top_p: float | None = None, temperature: float | None = None) -> ModelAdapter:
    if provider_config.type == ProviderType.GIGACHAT:
        return GigaChatAdapter(provider_config, top_p=top_p, temperature=temperature)
    if provider_config.type == ProviderType.OPENAI_COMPATIBLE:
        return OpenAICompatibleAdapter(provider_config)
    raise ValueError(f"Unknown provider type: {provider_config.type!r}")


# ---------------------------------------------------------------------------
# GigaChatLanguageModel — gepa reflection_lm protocol wrapper
# ---------------------------------------------------------------------------

class GigaChatLanguageModel:
    """Callable (str | list[dict]) -> str for use as gepa reflection_lm."""

    def __init__(self, provider_config: ModelProviderConfig, temperature: float = 0.7) -> None:
        self._adapter = create_adapter(provider_config, temperature=temperature)
        self._temperature = temperature
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            request = GenerationRequest(
                system_prompt="",
                user_message=prompt,
                temperature=self._temperature,
            )
        else:
            system_prompt = ""
            history: list[dict[str, str]] = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                else:
                    history.append({"role": role, "content": content})
            request = GenerationRequest(
                system_prompt=system_prompt,
                history=history or None,
                temperature=self._temperature,
            )

        response = self._adapter.generate(request)
        if response.usage:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
        return response.output_text
