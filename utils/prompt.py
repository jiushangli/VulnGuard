"""
VulnGuard Three-Layer Prompt Caching Manager.

Adapted from Hermes Architect's prompt caching strategy:
- STABLE layer: Byte-identical across the entire audit session.
  Contains agent role definition, audit framework, and security constraints.
  Must NOT change once set — this ensures LLM prompt cache hits.
- CONTEXT layer: Changes per phase/intent transition.
  Contains current audit phase, VulnKB gist summary, and current intent.
  Relatively stable within an OODA cycle.
- VOLATILE layer: Changes every OODA iteration.
  Contains latest tool results, course-correction reminders, and scratch content.

The build_prompt() method assembles layers in order, and the LLMGateway
adds appropriate cache_control markers for providers that support them.

Design invariant:
  STABLE messages must be byte-for-byte identical for the entire session.
  If you need to change stable content, start a new session.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────── Prompt Layer Enum ────────────────────────


class PromptLayer(Enum):
    """
    The three layers of prompt construction, ordered from most stable to most volatile.

    The ordering matters:
    1. STABLE (system prefix) — cached across the entire session
    2. CONTEXT (session state) — cached within a phase/intent window
    3. VOLATILE (per-cycle) — fresh content each OODA cycle
    """
    STABLE = "stable"
    CONTEXT = "context"
    VOLATILE = "volatile"


# ──────────────────────── Prompt Manager ────────────────────────


class PromptManager:
    """
    Manages the three-layer prompt structure for VulnGuard agents.

    Each layer serves a distinct caching purpose:

    STABLE (set once per session):
      - Agent role definition (who you are, what you do)
      - Audit framework description (VulnKB structure, fact types, intent lifecycle)
      - Security constraints (never exfiltrate, never modify VulnKB without admission)
      - Output format specifications

    CONTEXT (updates per phase/intent transition):
      - Current audit phase
      - VulnKB gist summary (compressed snapshot)
      - Current intent being worked on
      - Relevant hints filtered by severity

    VOLATILE (updates every OODA cycle):
      - Latest tool execution results
      - Course-correction reminders from Observer
      - Intermediate reasoning / scratch content
      - Recent facts added since last cycle

    Usage:
        pm = PromptManager()
        pm.set_stable(role_definition="...", audit_framework="...", security_constraints="...")
        pm.update_context(phase="vuln_mining", kb_gist="...", current_intent="...")
        pm.update_volatile(tool_results="...", corrections="...")
        messages = pm.build_messages()
    """

    def __init__(self, system_role: str = "system") -> None:
        """
        Args:
            system_role: The role string used for the system message (default: "system").
        """
        self._system_role = system_role

        # STABLE layer — set once, must remain byte-identical for the session
        self._role_definition: str = ""
        self._audit_framework: str = ""
        self._security_constraints: str = ""
        self._output_format: str = ""
        self._stable_custom: list[str] = []  # Additional stable sections
        self._stable_frozen: bool = False  # Lock after initial set

        # CONTEXT layer — updated per phase/intent transition
        self._current_phase: str = ""
        self._kb_gist: str = ""
        self._current_intent: str = ""
        self._relevant_hints: str = ""
        self._context_custom: list[str] = []

        # VOLATILE layer — updated every OODA cycle
        self._tool_results: str = ""
        self._corrections: str = ""
        self._recent_facts: str = ""
        self._scratch: str = ""
        self._volatile_custom: list[str] = []

        # Version tracking for cache invalidation diagnostics
        self._stable_version: int = 0
        self._context_version: int = 0
        self._volatile_version: int = 0

    # ──────────────────────── Stable Layer ────────────────────────

    def set_stable(
        self,
        *,
        role_definition: str = "",
        audit_framework: str = "",
        security_constraints: str = "",
        output_format: str = "",
        custom_sections: list[str] | None = None,
    ) -> None:
        """
        Set the STABLE layer content.

        This should be called exactly once per audit session. Once set,
        the stable layer is frozen to prevent accidental changes that would
        break LLM prompt cache hits.

        Args:
            role_definition: Agent's role, specialization, and responsibilities.
            audit_framework: Description of VulnKB, fact types, intent lifecycle.
            security_constraints: Safety rules (no exfiltration, verified admission, etc.).
            output_format: Expected output format specification.
            custom_sections: Additional stable sections.
        """
        if self._stable_frozen:
            raise RuntimeError(
                "Stable layer is frozen — cannot modify after initial set. "
                "Start a new PromptManager if you need different stable content."
            )

        self._role_definition = role_definition
        self._audit_framework = audit_framework
        self._security_constraints = security_constraints
        self._output_format = output_format
        self._stable_custom = list(custom_sections) if custom_sections else []
        self._stable_frozen = True
        self._stable_version = 1
        logger.info("Stable layer set and frozen (ensures prompt cache hits)")

    def _build_stable(self) -> str:
        """
        Build the stable layer string.

        Key invariant: This must produce byte-identical output every time
        for the entire session, otherwise prompt caching breaks.
        """
        parts: list[str] = []

        if self._role_definition:
            parts.append(f"# Role Definition\n{self._role_definition}")

        if self._audit_framework:
            parts.append(f"# Audit Framework\n{self._audit_framework}")

        if self._security_constraints:
            parts.append(f"# Security Constraints\n{self._security_constraints}")

        if self._output_format:
            parts.append(f"# Output Format\n{self._output_format}")

        for section in self._stable_custom:
            if section:
                parts.append(section)

        return "\n\n".join(parts)

    # ──────────────────────── Context Layer ────────────────────────

    def update_context(
        self,
        *,
        phase: str = "",
        kb_gist: str = "",
        current_intent: str = "",
        relevant_hints: str = "",
        custom_sections: list[str] | None = None,
    ) -> None:
        """
        Update the CONTEXT layer.

        Called when the agent transitions to a new phase, claims a new intent,
        or when the VulnKB gist changes significantly.

        Args:
            phase: Current audit phase (e.g., "code_intelligence", "vuln_mining", "verification").
            kb_gist: Compressed knowledge graph snapshot from kb.build_context(level="gist").
            current_intent: The intent currently being worked on.
            relevant_hints: Filtered hints relevant to the current task.
            custom_sections: Additional context sections.
        """
        self._current_phase = phase
        self._kb_gist = kb_gist
        self._current_intent = current_intent
        self._relevant_hints = relevant_hints
        if custom_sections is not None:
            self._context_custom = list(custom_sections)
        self._context_version += 1
        logger.debug(f"Context layer updated (version={self._context_version})")

    def _build_context(self) -> str:
        """Build the context layer string."""
        parts: list[str] = []

        if self._current_phase:
            parts.append(f"## Current Phase: {self._current_phase}")

        if self._kb_gist:
            parts.append(f"## Knowledge Graph Summary\n{self._kb_gist}")

        if self._current_intent:
            parts.append(f"## Current Intent\n{self._current_intent}")

        if self._relevant_hints:
            parts.append(f"## Relevant Hints\n{self._relevant_hints}")

        for section in self._context_custom:
            if section:
                parts.append(section)

        return "\n\n".join(parts)

    # ──────────────────────── Volatile Layer ────────────────────────

    def update_volatile(
        self,
        *,
        tool_results: str = "",
        corrections: str = "",
        recent_facts: str = "",
        scratch: str = "",
        custom_sections: list[str] | None = None,
    ) -> None:
        """
        Update the VOLATILE layer.

        Called at the beginning of each OODA cycle with fresh tool results,
        correction reminders, and other per-cycle content.

        Args:
            tool_results: Results from the most recent tool executions.
            corrections: Course-correction reminders from Observer or self-assessment.
            recent_facts: Facts added to VulnKB since last cycle.
            scratch: Intermediate reasoning or scratch content.
            custom_sections: Additional volatile sections.
        """
        self._tool_results = tool_results
        self._corrections = corrections
        self._recent_facts = recent_facts
        self._scratch = scratch
        if custom_sections is not None:
            self._volatile_custom = list(custom_sections)
        self._volatile_version += 1
        logger.debug(f"Volatile layer updated (version={self._volatile_version})")

    def _build_volatile(self) -> str:
        """Build the volatile layer string."""
        parts: list[str] = []

        if self._tool_results:
            parts.append(f"### Tool Results\n{self._tool_results}")

        if self._corrections:
            parts.append(f"### Course Corrections\n{self._corrections}")

        if self._recent_facts:
            parts.append(f"### Recent Discoveries\n{self._recent_facts}")

        if self._scratch:
            parts.append(f"### Working Notes\n{self._scratch}")

        for section in self._volatile_custom:
            if section:
                parts.append(section)

        return "\n\n".join(parts)

    # ──────────────────────── Build Methods ────────────────────────

    def build_prompt(self) -> str:
        """
        Build the complete prompt by concatenating all three layers.

        Layer ordering: STABLE → CONTEXT → VOLATILE
        This ordering maximizes prefix overlap with cached prompts.

        Returns:
            The assembled prompt string.
        """
        layers: list[str] = []

        stable = self._build_stable()
        if stable:
            layers.append(stable)

        context = self._build_context()
        if context:
            layers.append(context)

        volatile = self._build_volatile()
        if volatile:
            layers.append(volatile)

        return "\n\n---\n\n".join(layers)

    def build_messages(self) -> list[dict[str, Any]]:
        """
        Build messages in the standard chat format with layer separation.

        Returns a list of message dicts suitable for LLM Gateway consumption.
        The stable and context layers are combined into a system message,
        and the volatile layer becomes a user message.

        This format supports:
        - Anthropic-style cache_control markers (added by LLMGateway)
        - OpenAI-style message format
        - Easy integration with tool calling

        Returns:
            List of message dicts: [{"role": "system", "content": "..."}, ...]
        """
        messages: list[dict[str, Any]] = []

        # Stable layer as system message
        stable = self._build_stable()
        if stable:
            messages.append({
                "role": self._system_role,
                "content": stable,
            })

        # Context layer as a continuation system message
        context = self._build_context()
        if context:
            messages.append({
                "role": self._system_role,
                "content": context,
            })

        # Volatile layer as a user message
        volatile = self._build_volatile()
        if volatile:
            messages.append({
                "role": "user",
                "content": volatile,
            })

        return messages

    def build_layered_messages(self) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Return messages separated by layer for LLMGateway.build_cached_messages().

        This enables the gateway to apply proper cache_control markers
        to each layer boundary.

        Returns:
            Tuple of (stable_messages, context_messages, volatile_messages)
        """
        stable_msgs: list[dict[str, Any]] = []
        context_msgs: list[dict[str, Any]] = []
        volatile_msgs: list[dict[str, Any]] = []

        stable = self._build_stable()
        if stable:
            stable_msgs.append({
                "role": self._system_role,
                "content": stable,
            })

        context = self._build_context()
        if context:
            context_msgs.append({
                "role": self._system_role,
                "content": context,
            })

        volatile = self._build_volatile()
        if volatile:
            volatile_msgs.append({
                "role": "user",
                "content": volatile,
            })

        return stable_msgs, context_msgs, volatile_msgs

    # ──────────────────────── Metadata ────────────────────────

    @property
    def stable_frozen(self) -> bool:
        """Whether the stable layer has been set and frozen."""
        return self._stable_frozen

    @property
    def versions(self) -> dict[str, int]:
        """Current version numbers for each layer (for cache diagnostics)."""
        return {
            "stable": self._stable_version,
            "context": self._context_version,
            "volatile": self._volatile_version,
        }

    def layer_content(self, layer: PromptLayer) -> str:
        """
        Get the current content of a specific layer.

        Useful for diagnostics and token counting.
        """
        if layer == PromptLayer.STABLE:
            return self._build_stable()
        elif layer == PromptLayer.CONTEXT:
            return self._build_context()
        elif layer == PromptLayer.VOLATILE:
            return self._build_volatile()
        else:
            raise ValueError(f"Unknown layer: {layer}")

    def __repr__(self) -> str:
        frozen = "FROZEN" if self._stable_frozen else "UNSET"
        return (
            f"PromptManager(stable={frozen}, "
            f"versions=({self._stable_version}/{self._context_version}/{self._volatile_version}))"
        )