"""
VulnGuard Utilities Package.

Public interfaces:
- LLMGateway, LLMProvider, AgentRole, CompletionResult, StreamingChunk
- PromptManager, PromptLayer
- estimate_tokens (fallback token counter)
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
]