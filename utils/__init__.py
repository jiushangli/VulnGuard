"""
VulnGuard Utilities Package.

Public interfaces:
- LLMGateway, LLMProvider, AgentRole, CompletionResult, StreamingChunk
- PromptManager, PromptLayer
- estimate_tokens (fallback token counter)
- Provider implementations: OpenAIProvider, AnthropicProvider, OllamaProvider
- create_provider_from_config (factory function)
"""

from .llm import (
    AgentRole,
    CompletionResult,
    LLMGateway,
    LLMProvider,
    StreamingChunk,
    estimate_tokens,
)

from .prompt import (
    PromptLayer,
    PromptManager,
)

from .providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    create_provider_from_config,
)

__all__ = [
    # LLM
    "AgentRole",
    "CompletionResult",
    "LLMGateway",
    "LLMProvider",
    "StreamingChunk",
    "estimate_tokens",
    # Prompt
    "PromptLayer",
    "PromptManager",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "create_provider_from_config",
]