"""
VulnGuard Configuration Package.

Exports:
    - VulnGuardConfig and all sub-config dataclasses
    - default_config() factory function
"""

from .settings import (
    TargetConfig,
    IntelligenceConfig,
    AgentPoolConfig,
    AgentBudgetConfig,
    ObserverConfig,
    VulnKBConfig,
    LLMProviderConfig,
    LLMConfig,
    SecurityConfig,
    RulesConfig,
    VulnGuardConfig,
    default_config,
)

__all__ = [
    "TargetConfig",
    "IntelligenceConfig",
    "AgentPoolConfig",
    "AgentBudgetConfig",
    "ObserverConfig",
    "VulnKBConfig",
    "LLMProviderConfig",
    "LLMConfig",
    "SecurityConfig",
    "RulesConfig",
    "VulnGuardConfig",
    "default_config",
]