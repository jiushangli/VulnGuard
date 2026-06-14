"""
VulnGuard LLM Gateway — Multi-provider abstraction with prompt caching support.

Design principles (from Hermes three-layer caching):
- Providers are selected by agent role (miner/verifier/observer) to ensure independence
- Prompt caching uses stable/context/volatile layering:
  - STABLE layer: byte-identical across the entire audit session → cache hit guaranteed
  - CONTEXT layer: changes per phase/intent transition → periodic cache hits
  - VOLATILE layer: changes every OODA cycle → rarely cached, but small
- Token counting respects the layer structure for accurate budget estimation

Architecture:
- LLMProvider: abstract interface for LLM backends (OpenAI, Anthropic, local, etc.)
- LLMGateway: routing layer that selects provider by role and manages caching hints
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


# ──────────────────────── Role-Based Provider Selection ────────────────────────


class AgentRole(Enum):
    """Agent roles that map to distinct LLM providers for independence."""
    MINER = "miner"          # Primary analysis/exploration agents
    VERIFIER = "verifier"    # Independent verification agents
    OBSERVER = "observer"    # Strategic oversight agents


# ──────────────────────── Completion Types ────────────────────────


@dataclass
class CompletionResult:
    """Result from a completion call."""
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    cached: bool = False           # Whether this was served from prompt cache
    cache_creation_tokens: int = 0 # Tokens written to cache (Anthropic-style)
    cache_read_tokens: int = 0     # Tokens read from cache (Anthropic-style)
    finish_reason: str = ""


@dataclass
class StreamingChunk:
    """A single chunk from a streaming completion."""
    content: str
    model: str = ""
    finish_reason: Optional[str] = None


# ──────────────────────── LLMProvider Abstract Base ────────────────────────


class LLMProvider(abc.ABC):
    """
    Abstract interface for LLM backends.

    Each provider must implement:
    - complete(): Standard completion with full result metadata
    - complete_stream(): Streaming completion via async generator
    - count_tokens(): Token counting for budget estimation

    Providers are expected to respect prompt_caching hints in the messages
    (Anthropic-style cache_control markers, OpenAI-style ephemeral markers, etc.).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of this provider (e.g. 'openai-gpt4', 'anthropic-claude')."""
        ...

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Model identifier used in API calls."""
        ...

    @abc.abstractmethod
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
        """
        Generate a completion for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                      May include cache_control markers for prompt caching.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Optional stop sequences.
            tools: Optional tool definitions for function calling.
            prompt_caching: Whether to use prompt caching (stable/context layer hints).

        Returns:
            CompletionResult with content, usage, and cache statistics.
        """
        ...

    @abc.abstractmethod
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
        """
        Generate a streaming completion for the given messages.

        Yields StreamingChunk objects as they arrive.
        """
        ...

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.

        Used for budget estimation before sending to the LLM.
        Implementations should use provider-specific tokenizers when available,
        or fall back to a reasonable approximation (4 chars ≈ 1 token).
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} model={self.model_id!r}>"


# ──────────────────────── Fallback Token Counter ────────────────────────


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: ~4 characters per token.
    Used when the provider doesn't supply a tokenizer.
    """
    return max(1, len(text) // 4)


# ──────────────────────── LLM Gateway ────────────────────────


class LLMGateway:
    """
    Central routing layer for LLM calls.

    Manages multiple providers and routes requests based on agent role.
    This ensures:
    - Miner and Verifier use different providers (independence guarantee)
    - Observer can use either (configurable)
    - Prompt caching hints are applied per the three-layer structure

    Usage:
        gateway = LLMGateway()
        gateway.register_provider(AgentRole.MINER, OpenAIProvider(...))
        gateway.register_provider(AgentRole.VERIFIER, AnthropicProvider(...))
        result = await gateway.complete(AgentRole.MINER, messages)
    """

    def __init__(self) -> None:
        self._providers: dict[AgentRole, LLMProvider] = {}
        self._fallback_provider: Optional[LLMProvider] = None

    def register_provider(
        self,
        role: AgentRole,
        provider: LLMProvider,
        fallback: bool = False,
    ) -> None:
        """
        Register an LLM provider for a specific agent role.

        Args:
            role: The agent role this provider serves.
            provider: The LLM provider instance.
            fallback: If True, use this provider as fallback for unregistered roles.
        """
        self._providers[role] = provider
        if fallback:
            self._fallback_provider = provider
        logger.info(f"Registered LLM provider '{provider.name}' for role '{role.value}'")

    def get_provider(self, role: AgentRole) -> LLMProvider:
        """
        Get the provider for a given role.

        Falls back to the fallback provider if no direct mapping exists.
        Raises RuntimeError if no provider is available.
        """
        provider = self._providers.get(role)
        if provider is not None:
            return provider

        if self._fallback_provider is not None:
            logger.debug(f"No provider for role '{role.value}', using fallback")
            return self._fallback_provider

        raise RuntimeError(
            f"No LLM provider registered for role '{role.value}' "
            f"and no fallback configured. Available roles: "
            f"{list(self._providers.keys())}"
        )

    async def complete(
        self,
        role: AgentRole,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        prompt_caching: bool = True,
    ) -> CompletionResult:
        """
        Generate a completion using the provider for the given role.

        This is the primary entry point for all LLM calls in VulnGuard.
        The role determines which provider to use, ensuring independence.
        """
        provider = self.get_provider(role)
        result = await provider.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=tools,
            prompt_caching=prompt_caching,
        )
        logger.debug(
            f"LLM complete via {provider.name}: "
            f"{result.usage.get('total_tokens', '?')} tokens, "
            f"cached={result.cached}"
        )
        return result

    async def complete_stream(
        self,
        role: AgentRole,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        prompt_caching: bool = True,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Stream a completion using the provider for the given role.
        """
        provider = self.get_provider(role)
        async for chunk in provider.complete_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=tools,
            prompt_caching=prompt_caching,
        ):
            yield chunk

    def count_tokens(self, role: AgentRole, text: str) -> int:
        """
        Estimate token count using the provider for the given role.
        """
        provider = self.get_provider(role)
        return provider.count_tokens(text)

    def build_cached_messages(
        self,
        stable_messages: list[dict[str, Any]],
        context_messages: list[dict[str, Any]],
        volatile_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Assemble messages with prompt caching markers per the three-layer structure.

        Anthropic-style cache_control usage:
        - Stable layer: cache_control={"type": "ephemeral"} on the last stable message
          → this prefix stays byte-identical for the entire session, maximizing cache hits
        - Context layer: cache_control={"type": "ephemeral"} on the last context message
          → changes per phase/intent transition, but stable within a cycle
        - Volatile layer: no caching markers → fresh every time

        For OpenAI-style providers, the cache_control markers are simply ignored
        (OpenAI uses automatic prefix matching).

        Returns:
            Combined message list with cache_control markers applied.
        """
        messages: list[dict[str, Any]] = []

        # Stable layer — always byte-identical within a session
        if stable_messages:
            for i, msg in enumerate(stable_messages):
                m = dict(msg)
                # Mark the last stable message as a cache breakpoint
                if i == len(stable_messages) - 1:
                    m["cache_control"] = {"type": "ephemeral"}
                messages.append(m)

        # Context layer — changes per phase/intent transition
        if context_messages:
            for i, msg in enumerate(context_messages):
                m = dict(msg)
                # Mark the last context message as a cache breakpoint
                if i == len(context_messages) - 1:
                    m["cache_control"] = {"type": "ephemeral"}
                messages.append(m)

        # Volatile layer — changes every cycle, no caching markers
        if volatile_messages:
            for msg in volatile_messages:
                messages.append(dict(msg))

        return messages

    @property
    def available_roles(self) -> list[AgentRole]:
        """List roles that have a registered provider."""
        return list(self._providers.keys())

    def __repr__(self) -> str:
        role_info = ", ".join(
            f"{r.value}={p.name}" for r, p in self._providers.items()
        )
        return f"LLMGateway({role_info})"