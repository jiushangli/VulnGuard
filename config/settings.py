"""
VulnGuard Configuration System.

Complete configuration definition using dataclasses, with YAML
serialization/deserialization support.

Configuration hierarchy:
- VulnGuardConfig (top-level)
  - TargetConfig        — audit target (repo path, language, entry points)
  - IntelligenceConfig  — code intelligence engine settings
  - AgentPoolConfig     — agent pool sizing (miners, verifiers, observer)
  - AgentBudgetConfig  — per-agent resource limits (OODA cycles, tokens, heartbeat)
  - ObserverConfig     — observer review policy
  - VulnKBConfig       — knowledge base storage and admission settings
  - LLMConfig          — LLM providers and role routing
  - SecurityConfig     — phase policies, sandbox, PoC directory
  - RulesConfig        — vulnerability rule sources
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional


# ──────────────────────── Configuration Data Classes ────────────────────────


@dataclass
class TargetConfig:
    """Configuration for the audit target repository."""
    repo_path: str = ""
    language: str = "python"  # python, java, go, javascript, typescript
    entry_points: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "*.test.*", "*_test.*", "test_*"
    ])


@dataclass
class IntelligenceConfig:
    """Configuration for the CodeIntelligenceEngine."""
    max_tokens_per_module: int = 36000
    max_cluster_depth: int = 3
    extract_api_routes: bool = True
    extract_dataflow: bool = True
    extract_state_machines: bool = True
    parse_languages: list[str] = field(default_factory=lambda: [
        "python", "java", "javascript", "typescript", "go"
    ])


@dataclass
class AgentPoolConfig:
    """Configuration for the agent pool."""
    miner_count: int = 4
    miner_specializations: list[str] = field(default_factory=lambda: [
        "api_sequence", "dataflow_taint", "business_logic", "attack_surface"
    ])
    verifier_count: int = 2
    observer_enabled: bool = True


@dataclass
class AgentBudgetConfig:
    """Per-agent resource limits and timing configuration."""
    max_ooda_cycles: int = 20
    max_consecutive_failures: int = 3
    budget_per_agent: int = 50000  # tokens
    heartbeat_interval: int = 30  # seconds
    claim_lease_seconds: int = 300


@dataclass
class ObserverConfig:
    """Configuration for the ObserverAgent review policy."""
    review_every_n_facts: int = 5
    reminder_min_interval: int = 3
    default_stance: str = "no_change"  # no_change > adjust > add


@dataclass
class VulnKBConfig:
    """Configuration for the VulnKB knowledge base."""
    storage: str = "sqlite"
    db_path: str = ":memory:"
    gist_max_tokens: int = 220
    summary_max_tokens: int = 2000
    verification_enabled: bool = True
    heartbeat_interval_seconds: int = 30
    intent_lease_seconds: int = 300


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""
    name: str = ""
    model: str = ""
    api_base: str = ""         # Alias: also accessible as base_url
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        """Alias for api_base, for convenience."""
        return self.api_base

    @base_url.setter
    def base_url(self, value: str) -> None:
        self.api_base = value


@dataclass
class LLMConfig:
    """Configuration for LLM providers and role routing."""
    providers: dict[str, LLMProviderConfig] = field(default_factory=dict)
    # roles: miner → provider name, verifier → provider name, observer → provider name
    role_mapping: dict[str, str] = field(default_factory=lambda: {
        "miner": "primary",
        "verifier": "verifier",
        "observer": "primary"
    })
    prompt_caching: bool = True


@dataclass
class SecurityConfig:
    """Security and sandbox configuration."""
    phase_policies: dict[str, dict] = field(default_factory=dict)
    sandbox_enabled: bool = True
    poc_directory: str = "/tmp/vulnguard-poc"
    allowed_network_targets: list[str] = field(default_factory=list)


@dataclass
class RulesConfig:
    """Vulnerability rule sources configuration."""
    sources: list[str] = field(default_factory=lambda: [
        "owasp_api_top10", "cwe_top_25"
    ])
    custom_rules_path: str = ""


@dataclass
class VulnGuardConfig:
    """Top-level configuration for the VulnGuard framework."""
    target: TargetConfig = field(default_factory=TargetConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    agents: AgentPoolConfig = field(default_factory=AgentPoolConfig)
    budget: AgentBudgetConfig = field(default_factory=AgentBudgetConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    vulnkb: VulnKBConfig = field(default_factory=VulnKBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)

    # ──────────────────────── YAML I/O ────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> 'VulnGuardConfig':
        """
        Load configuration from a YAML file.

        Supports nested configuration sections and environment variable
        interpolation for sensitive values (API keys).

        Environment variable interpolation:
          ${ENV_VAR}            → os.environ['ENV_VAR']
          ${ENV_VAR:-default}   → os.environ.get('ENV_VAR', 'default')
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML configuration. "
                "Install it with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Interpolate environment variables
        raw = cls._interpolate_env(raw)

        return cls._from_dict(raw)

    def to_yaml(self, path: str) -> None:
        """
        Save configuration to a YAML file.

        Creates parent directories if they don't exist.
        API keys are masked in the output for security.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML configuration. "
                "Install it with: pip install pyyaml"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict_masked()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ──────────────────────── Dict Conversion ────────────────────────

    @classmethod
    def _from_dict(cls, data: dict) -> 'VulnGuardConfig':
        """Recursively construct VulnGuardConfig from a nested dict."""
        config = cls()

        # TargetConfig
        if "target" in data:
            config.target = _update_dataclass(config.target, data["target"])

        # IntelligenceConfig
        if "intelligence" in data:
            config.intelligence = _update_dataclass(
                config.intelligence, data["intelligence"]
            )

        # AgentPoolConfig
        if "agents" in data:
            config.agents = _update_dataclass(config.agents, data["agents"])

        # AgentBudgetConfig
        if "budget" in data:
            config.budget = _update_dataclass(config.budget, data["budget"])

        # ObserverConfig
        if "observer" in data:
            config.observer = _update_dataclass(
                config.observer, data["observer"]
            )

        # VulnKBConfig
        if "vulnkb" in data:
            config.vulnkb = _update_dataclass(config.vulnkb, data["vulnkb"])

        # LLMConfig — special handling for nested providers dict
        if "llm" in data:
            llm_data = data["llm"]
            if "providers" in llm_data:
                providers = {}
                for name, prov_data in llm_data["providers"].items():
                    providers[name] = _update_dataclass(
                        LLMProviderConfig(), prov_data
                    )
                config.llm.providers = providers
            if "role_mapping" in llm_data:
                config.llm.role_mapping = llm_data["role_mapping"]
            if "prompt_caching" in llm_data:
                config.llm.prompt_caching = llm_data["prompt_caching"]

        # SecurityConfig
        if "security" in data:
            config.security = _update_dataclass(
                config.security, data["security"]
            )

        # RulesConfig
        if "rules" in data:
            config.rules = _update_dataclass(config.rules, data["rules"])

        return config

    def _to_dict_masked(self) -> dict:
        """
        Convert config to dict with API keys masked for safe output.
        """
        data = asdict(self)
        # Mask API keys
        if "llm" in data and "providers" in data["llm"]:
            for name, prov in data["llm"]["providers"].items():
                if prov.get("api_key"):
                    key = prov["api_key"]
                    prov["api_key"] = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
        return data

    @staticmethod
    def _interpolate_env(data: Any) -> Any:
        """
        Recursively interpolate environment variables in config values.

        Supports:
          ${ENV_VAR}            → os.environ['ENV_VAR']
          ${ENV_VAR:-default}   → os.environ.get('ENV_VAR', 'default')
        """
        import re

        env_pattern = re.compile(r'\$\{([^}:]+)(?::-([^}]*))?\}')

        def replace_env_vars(value: str) -> str:
            def replacer(match):
                var_name = match.group(1)
                default_val = match.group(2)
                if default_val is not None:
                    return os.environ.get(var_name, default_val)
                return os.environ[var_name]
            return env_pattern.sub(replacer, value)

        if isinstance(data, dict):
            return {k: VulnGuardConfig._interpolate_env(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [VulnGuardConfig._interpolate_env(v) for v in data]
        elif isinstance(data, str):
            try:
                return replace_env_vars(data)
            except KeyError:
                # If env var not found and no default, leave as-is
                return data
        return data


# ──────────────────────── Helper Functions ────────────────────────


def _update_dataclass(dc, data: dict):
    """
    Update a dataclass instance from a dict, only setting fields that
    are present in the dict. Preserves default values for missing fields.
    """
    if data is None:
        return dc

    field_names = {f.name for f in fields(dc)}
    for key, value in data.items():
        if key in field_names:
            setattr(dc, key, value)
    return dc


# ──────────────────────── Default Config Instance ────────────────────────


def default_config() -> VulnGuardConfig:
    """Create and return a default VulnGuardConfig."""
    return VulnGuardConfig()