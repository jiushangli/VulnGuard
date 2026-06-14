"""
VulnGuard LLM Provider Implementations.

Concrete providers for OpenAI, Anthropic, and Ollama backends.
Each provider implements the LLMProvider abstract interface from utils.llm.

Graceful import handling:
- If openai or anthropic is not installed, the class is still importable
  but will raise ImportError only when actually instantiated or used.
- OllamaProvider only requires httpx (a VulnGuard dependency).
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator, Optional

from .llm import CompletionResult, LLMProvider, StreamingChunk, estimate_tokens

logger = logging.getLogger(__name__)

# Lazy imports — caches so we don't repeat import attempts
_openai_module: Any = None
_openai_checked: bool = False
_anthropic_module: Any = None
_anthropic_checked: bool = False


def _get_openai():
    """Lazy import of the openai package."""
    global _openai_module, _openai_checked
    if not _openai_checked:
        try:
            import openai as _oi
            _openai_module = _oi
        except ImportError:
            _openai_module = None
        _openai_checked = True
    if _openai_module is None:
        raise ImportError(
            "The 'openai' package is required for OpenAIProvider. "
            "Install it with: pip install openai"
        )
    return _openai_module


def _get_anthropic():
    """Lazy import of the anthropic package."""
    global _anthropic_module, _anthropic_checked
    if not _anthropic_checked:
        try:
            import anthropic as _ai
            _anthropic_module = _ai
        except ImportError:
            _anthropic_module = None
        _anthropic_checked = True
    if _anthropic_module is None:
        raise ImportError(
            "The 'anthropic' package is required for AnthropicProvider. "
            "Install it with: pip install anthropic"
        )
    return _anthropic_module


# ──────────────────────── OpenAI Provider ────────────────────────


class OpenAIProvider(LLMProvider):
    """
    LLM provider backed by OpenAI-compatible APIs (OpenAI, Azure OpenAI, etc.).

    Uses the openai async SDK (AsyncOpenAI). Supports prompt caching via
    ephemeral system-message markers when prompt_caching=True.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        organization: str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = extra or {}
        self._client: Any = None  # lazily created AsyncOpenAI
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization

    def _ensure_client(self):
        """Create the AsyncOpenAI client on first use (raises ImportError if missing)."""
        if self._client is not None:
            return
        openai = _get_openai()
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._organization:
            kwargs["organization"] = self._organization
        self._client = openai.AsyncOpenAI(**kwargs)

    @property
    def name(self) -> str:
        return f"openai-{self._model}"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        prompt_caching: bool = True,
    ) -> CompletionResult:
        self._ensure_client()
        openai = _get_openai()

        # Convert VulnGuard cache_control markers to OpenAI-compatible format.
        # OpenAI doesn't use cache_control in messages; it uses automatic prefix
        # matching. However, some OpenAI-compatible endpoints (e.g. Anthropic's
        # OpenAI bridge) may respect ephemeral markers. We strip our internal
        # markers for strict OpenAI.
        api_messages = self._strip_cache_markers(messages) if prompt_caching else messages

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature or self._temperature,
        }
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
            # OpenAI may return prompt_tokens_details with cached_tokens
            if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
                cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
                usage["cached_tokens"] = cached

        # Detect if cached (OpenAI doesn't always report this)
        cached = usage.get("cached_tokens", 0) > 0

        return CompletionResult(
            content=content,
            model=response.model or self._model,
            usage=usage,
            cached=cached,
            cache_creation_tokens=0,
            cache_read_tokens=usage.get("cached_tokens", 0),
            finish_reason=choice.finish_reason or "",
        )

    async def complete_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        prompt_caching: bool = True,
    ) -> AsyncIterator[StreamingChunk]:
        self._ensure_client()

        api_messages = self._strip_cache_markers(messages) if prompt_caching else messages

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature or self._temperature,
            "stream": True,
        }
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        stream = await self._client.chat.completions.create(**kwargs)

        model_name = self._model
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                finish = chunk.choices[0].finish_reason
                if content or finish:
                    yield StreamingChunk(
                        content=content,
                        model=chunk.model or model_name,
                        finish_reason=finish,
                    )

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count. Uses tiktoken if available, otherwise falls back
        to rough estimation.
        """
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self._model)
            return len(enc.encode(text))
        except Exception:
            return estimate_tokens(text)

    # ──── Helpers ────

    @staticmethod
    def _strip_cache_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove cache_control markers that VulnGateway adds (OpenAI doesn't use them)."""
        cleaned = []
        for msg in messages:
            m = {k: v for k, v in msg.items() if k != "cache_control"}
            cleaned.append(m)
        return cleaned

    @staticmethod
    def _format_tools(tools: list[dict]) -> list[dict]:
        """Convert VulnGuard tool defs to OpenAI function-calling format if needed."""
        formatted = []
        for tool in tools:
            # If already in OpenAI format, pass through
            if "type" in tool:
                formatted.append(tool)
            else:
                # Assume {name, description, parameters} format
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                })
        return formatted


# ──────────────────────── Anthropic Provider ────────────────────────


class AnthropicProvider(LLMProvider):
    """
    LLM provider backed by Anthropic's Claude API.

    Uses the anthropic async SDK (AsyncAnthropic). Supports prompt caching
    via Anthropic's cache_control markers on messages.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        extra: dict[str, Any] | None = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = extra or {}
        self._client: Any = None
        self._api_key = api_key
        self._base_url = base_url

    def _ensure_client(self):
        """Create the AsyncAnthropic client on first use."""
        if self._client is not None:
            return
        anthropic = _get_anthropic()
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    @property
    def name(self) -> str:
        return f"anthropic-{self._model}"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        prompt_caching: bool = True,
    ) -> CompletionResult:
        self._ensure_client()

        # Anthropic requires separating system message from conversation messages
        system_content, conv_messages = self._split_system(messages)

        # If prompt_caching is enabled, ensure cache_control markers are
        # applied to system content blocks (Anthropic requires cache breakpoints)
        if prompt_caching and system_content:
            if isinstance(system_content, str):
                # Convert to content blocks with caching on the last block
                system_content = [
                    {"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(system_content, list):
                # Ensure the last block has cache_control
                if system_content:
                    last = dict(system_content[-1])
                    last["cache_control"] = {"type": "ephemeral"}
                    system_content = system_content[:-1] + [last]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": conv_messages,
            "max_tokens": max_tokens or self._max_tokens,
        }

        if system_content:
            kwargs["system"] = system_content

        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        # Anthropic SDK requires prompt caching to be opted-in via extra_headers
        extra_headers = {}
        if prompt_caching:
            # Enable prompt caching via beta header if needed
            # The anthropic SDK >=0.30.0 supports cache_control natively
            pass

        response = await self._client.messages.create(**kwargs)

        # Extract content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "output_tokens", 0) or 0,
                "total_tokens": (
                    (getattr(response.usage, "input_tokens", 0) or 0)
                    + (getattr(response.usage, "output_tokens", 0) or 0)
                ),
            }
            # Cache metrics from Anthropic
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            usage["cache_creation_tokens"] = cache_creation
            usage["cache_read_tokens"] = cache_read

        cache_creation = usage.get("cache_creation_tokens", 0)
        cache_read = usage.get("cache_read_tokens", 0)

        return CompletionResult(
            content=content,
            model=response.model or self._model,
            usage=usage,
            cached=cache_read > 0,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            finish_reason=response.stop_reason or "",
        )

    async def complete_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        prompt_caching: bool = True,
    ) -> AsyncIterator[StreamingChunk]:
        self._ensure_client()

        system_content, conv_messages = self._split_system(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": conv_messages,
            "max_tokens": max_tokens or self._max_tokens,
        }

        if system_content:
            kwargs["system"] = system_content
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        async with self._client.messages.stream(**kwargs) as stream:
            model_name = self._model
            async for text in stream.text_stream:
                yield StreamingChunk(
                    content=text,
                    model=model_name,
                    finish_reason=None,
                )
            # Yield final chunk with finish reason
            yield StreamingChunk(
                content="",
                model=model_name,
                finish_reason=stream.stop_reason if hasattr(stream, "stop_reason") else "end_turn",
            )

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count. Uses Anthropic's tokenizer if available,
        otherwise falls back to rough estimation.
        """
        # Anthropic uses roughly 3.5 chars/token for English text
        return max(1, len(text) // 4)

    # ──── Helpers ────

    @staticmethod
    def _split_system(
        messages: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        """
        Separate system messages from conversation messages.

        Anthropic requires the system prompt as a separate parameter.
        Returns (system_content, conversation_messages).
        system_content can be a string, a list of content blocks, or None.
        """
        system_parts: list[dict[str, Any]] = []
        conv_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                # Preserve cache_control markers from the gateway
                cache_control = msg.get("cache_control")
                content = msg.get("content", "")
                block = {"type": "text", "text": content}
                if cache_control:
                    block["cache_control"] = cache_control
                system_parts.append(block)
            else:
                m = {k: v for k, v in msg.items() if k != "cache_control"}
                # Anthropic expects content as string or content blocks
                conv_messages.append(m)

        system_content = system_parts if system_parts else None
        return system_content, conv_messages

    @staticmethod
    def _format_tools(tools: list[dict]) -> list[dict]:
        """Convert VulnGuard tool defs to Anthropic tool format."""
        formatted = []
        for tool in tools:
            if "type" in tool and tool["type"] == "custom":
                # Already in Anthropic format
                formatted.append(tool)
            elif "name" in tool:
                formatted.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", tool.get("input_schema", {})),
                })
            else:
                formatted.append(tool)
        return formatted


# ──────────────────────── Ollama Provider ────────────────────────


class OllamaProvider(LLMProvider):
    """
    LLM provider backed by a local Ollama instance.

    Uses httpx to call the Ollama REST API at localhost:11434.
    This is ideal for local/offline usage with models like Llama 3, Mistral, etc.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        *,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        extra: dict[str, Any] | None = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = extra or {}
        self._client: Any = None  # lazily created httpx.AsyncClient

    def _ensure_client(self):
        """Create the httpx client on first use."""
        if self._client is not None:
            return
        import httpx
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))

    @property
    def name(self) -> str:
        return f"ollama-{self._model}"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        prompt_caching: bool = True,
    ) -> CompletionResult:
        self._ensure_client()

        # Strip cache_control markers (Ollama doesn't support them)
        api_messages = self._strip_cache_markers(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "options": {
                "temperature": temperature or self._temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
            "stream": False,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0) or 0
        prompt_eval_count = data.get("prompt_eval_count", 0) or 0
        total = prompt_eval_count + eval_count

        return CompletionResult(
            content=content,
            model=data.get("model", self._model),
            usage={
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_tokens": total,
            },
            finish_reason=data.get("done_reason", "stop") if data.get("done") else "",
        )

    async def complete_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        prompt_caching: bool = True,
    ) -> AsyncIterator[StreamingChunk]:
        self._ensure_client()

        api_messages = self._strip_cache_markers(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "options": {
                "temperature": temperature or self._temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
            "stream": True,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()

            model_name = self._model
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                import json
                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = chunk_data.get("message", {}).get("content", "")
                done = chunk_data.get("done", False)

                if content:
                    yield StreamingChunk(
                        content=content,
                        model=chunk_data.get("model", model_name),
                        finish_reason=None,
                    )
                if done:
                    yield StreamingChunk(
                        content="",
                        model=chunk_data.get("model", model_name),
                        finish_reason=chunk_data.get("done_reason", "stop"),
                    )

    def count_tokens(self, text: str) -> int:
        """Estimate token count using 4 chars ≈ 1 token."""
        return max(1, len(text) // 4)

    @staticmethod
    def _strip_cache_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove cache_control markers (Ollama doesn't support them)."""
        cleaned = []
        for msg in messages:
            m = {k: v for k, v in msg.items() if k != "cache_control"}
            cleaned.append(m)
        return cleaned

    async def close(self):
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ──────────────────────── Factory Function ────────────────────────


# Map of provider type names to their classes
_PROVIDER_REGISTRY: dict[str, type] = {}

# Register known providers (safe even if SDKs aren't installed)
# We register by name string; the actual import check happens at instantiation time.
_BUILTIN_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}

# Aliases for convenience
_PROVIDER_ALIASES = {
    "gpt": "openai",
    "gpt4": "openai",
    "claude": "anthropic",
    "local": "ollama",
}


def create_provider_from_config(config: Any) -> LLMProvider | None:
    """
    Factory function: create an LLM provider from an LLMProviderConfig.

    The config object is expected to have these fields:
    - name: identifier and provider type hint (e.g. "openai", "anthropic", "ollama")
    - model: model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514", "llama3")
    - api_base / base_url: API endpoint URL
    - api_key: API key
    - temperature: sampling temperature
    - max_tokens: max tokens per response
    - extra: dict of additional provider-specific options

    The `name` field or the `extra.type` field is used to determine which
    provider class to instantiate. Supported values:
      "openai", "anthropic", "ollama" (plus aliases "gpt", "claude", "local")

    Returns None if the provider type cannot be determined or the required
    SDK is not installed.
    """
    # Determine provider type from config
    provider_type = _determine_provider_type(config)
    if provider_type is None:
        logger.warning(f"Cannot determine provider type from config: name={getattr(config, 'name', '?')}")
        return None

    provider_cls = _BUILTIN_PROVIDERS.get(provider_type)
    if provider_cls is None:
        logger.warning(f"Unknown provider type: {provider_type}")
        return None

    # Extract common config values, supporting both dataclass and dict
    model = getattr(config, "model", "") or ""
    api_key = getattr(config, "api_key", "") or ""
    base_url = getattr(config, "api_base", "") or getattr(config, "base_url", "") or ""
    temperature = getattr(config, "temperature", 0.3) or 0.3
    max_tokens = getattr(config, "max_tokens", 4096) or 4096
    extra = getattr(config, "extra", {}) or {}

    try:
        if provider_cls is OpenAIProvider:
            return OpenAIProvider(
                model=model or "gpt-4o",
                api_key=api_key,
                base_url=base_url or None,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )
        elif provider_cls is AnthropicProvider:
            return AnthropicProvider(
                model=model or "claude-sonnet-4-20250514",
                api_key=api_key,
                base_url=base_url or None,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )
        elif provider_cls is OllamaProvider:
            return OllamaProvider(
                model=model or "llama3",
                base_url=base_url or OllamaProvider.DEFAULT_BASE_URL,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )
        else:
            logger.error(f"Provider class not recognized: {provider_cls}")
            return None
    except ImportError as e:
        logger.warning(f"Cannot create {provider_type} provider (missing SDK): {e}")
        return None


def _determine_provider_type(config: Any) -> str | None:
    """
    Determine the provider type from the config object.

    Checks in order:
    1. extra.type field
    2. name field (matched against known providers and aliases)
    3. model field (heuristics: "gpt-*" → openai, "claude-*" → anthropic)
    """
    # Check extra.type first
    extra = getattr(config, "extra", None) or {}
    if isinstance(extra, dict) and "type" in extra:
        t = extra["type"].lower()
        if t in _BUILTIN_PROVIDERS:
            return t
        if t in _PROVIDER_ALIASES:
            return _PROVIDER_ALIASES[t]

    # Check name field
    name = getattr(config, "name", "") or ""
    if name:
        name_lower = name.lower()
        # Direct match
        if name_lower in _BUILTIN_PROVIDERS:
            return name_lower
        if name_lower in _PROVIDER_ALIASES:
            return _PROVIDER_ALIASES[name_lower]
        # Prefix match
        for key in _BUILTIN_PROVIDERS:
            if name_lower.startswith(key):
                return key

    # Check model field heuristics
    model = getattr(config, "model", "") or ""
    if model:
        model_lower = model.lower()
        if model_lower.startswith("gpt-") or model_lower.startswith("o"):
            return "openai"
        if model_lower.startswith("claude-") or "claude" in model_lower:
            return "anthropic"

    return None