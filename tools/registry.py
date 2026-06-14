"""
VulnGuard Audit Tool Registry.

Central registry for tool discovery, phase-based filtering, and tool plan generation.

Design principles:
- Registry is the single source of truth for what tools exist
- Phase-based filtering enforces audit security boundaries
- Tool plans include diagnostic info for hidden tools (observability)
- Fact-type dependency resolution before plan construction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..vulnkb.models import RiskLevel, AuditPhase, FactType

from .descriptor import (
    AuditTool,
    ToolDescriptor,
    ToolContext,
    ToolAvailabilityExpression,
    _risk_level_order,
)


# ──────────────────── Phase Risk Policy ────────────────────

# Maps each audit phase to the maximum risk level permitted.
# This is the core security boundary for VulnGuard.
PHASE_RISK_CAP: dict[AuditPhase, RiskLevel] = {
    AuditPhase.CODE_INTELLIGENCE: RiskLevel.ANALYSIS,
    # Phase 0: AST, dependency graphs — only safe reads and static analysis
    AuditPhase.VULN_MINING: RiskLevel.ANALYSIS,
    # Phase 1: Parallel exploration — analysis + read-only, no writes yet
    AuditPhase.VERIFICATION: RiskLevel.DANGEROUS,
    # Phase 2: PoC construction — all tools including dangerous ones
}


# ──────────────────── Tool Plan ────────────────────


@dataclass(frozen=True)
class HiddenToolDiagnostic:
    """Diagnostic info explaining why a tool is hidden."""
    tool_name: str
    reason: str


@dataclass(frozen=True)
class ToolPlan:
    """
    Result of building a tool plan for a specific phase and context.

    Inspired by OpenClaw's tool planning: the registry determines which
    tools are visible and which are hidden (with diagnostic reasons).
    """
    phase: AuditPhase
    visible: tuple[ToolDescriptor, ...]
    hidden: tuple[HiddenToolDiagnostic, ...]

    @property
    def visible_names(self) -> list[str]:
        return [t.name for t in self.visible]

    @property
    def hidden_names(self) -> list[str]:
        return [d.tool_name for d in self.hidden]

    def summary(self) -> str:
        lines = [
            f"Tool Plan for phase={self.phase.value}:",
            f"  Visible ({len(self.visible)}): {', '.join(self.visible_names) or '(none)'}",
            f"  Hidden ({len(self.hidden)}):",
        ]
        for diag in self.hidden:
            lines.append(f"    - {diag.tool_name}: {diag.reason}")
        return "\n".join(lines)


# ──────────────────── Audit Tool Registry ────────────────────


class AuditToolRegistry:
    """
    Central registry for VulnGuard audit tools.

    Responsibilities:
    1. Registration: tools register with descriptor + implementation
    2. Discovery: find tools by phase, agent type, or name
    3. Filtering: build tool plans based on phase security policies
    4. Dependency resolution: check fact-type dependencies from VulnKB
    """

    def __init__(self) -> None:
        self._tools: dict[str, AuditTool] = {}
        self._descriptors: dict[str, ToolDescriptor] = {}

    def register(self, tool: AuditTool) -> None:
        """
        Register an audit tool.

        The tool's descriptor provides metadata; the tool instance provides
        runtime behavior. Both are stored in the registry.
        """
        desc = tool.descriptor
        if desc.name in self._tools:
            raise ValueError(
                f"Tool '{desc.name}' is already registered. "
                f"Existing: {self._descriptors[desc.name]}, "
                f"New: {desc}"
            )
        self._tools[desc.name] = tool
        self._descriptors[desc.name] = desc

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)
        self._descriptors.pop(name, None)

    def get(self, name: str) -> Optional[AuditTool]:
        """Get a tool implementation by name."""
        return self._tools.get(name)

    def get_descriptor(self, name: str) -> Optional[ToolDescriptor]:
        """Get a tool descriptor by name."""
        return self._descriptors.get(name)

    @property
    def all_tools(self) -> list[AuditTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    @property
    def all_descriptors(self) -> list[ToolDescriptor]:
        """Get all registered descriptors."""
        return list(self._descriptors.values())

    def get_tools_for_phase(
        self, phase: AuditPhase, available_fact_types: set[FactType] | None = None,
        agent_type: str | None = None, permissions: set[str] | None = None,
    ) -> list[AuditTool]:
        """
        Get tools available in the specified audit phase.

        Applies security boundary: each phase has a maximum risk level.
        Also evaluates availability expressions against the runtime context.
        """
        if available_fact_types is None:
            available_fact_types = set()
        if permissions is None:
            permissions = set()

        context = self._build_context(phase, agent_type, available_fact_types, permissions)
        return [tool for tool in self._tools.values() if tool.descriptor.is_available(context)]

    def get_tools_for_agent(
        self, agent_type: str, phase: AuditPhase,
        available_fact_types: set[FactType] | None = None,
        permissions: set[str] | None = None,
    ) -> list[AuditTool]:
        """
        Get tools available to a specific agent type in the given phase.

        Filters by:
        1. Phase risk cap
        2. Agent type availability expression
        3. Fact-type dependencies
        4. Permission requirements
        """
        if available_fact_types is None:
            available_fact_types = set()
        if permissions is None:
            permissions = set()

        context = self._build_context(phase, agent_type, available_fact_types, permissions)
        tools = []
        for tool in self._tools.values():
            desc = tool.descriptor
            # Phase membership check
            if phase not in desc.audit_phases:
                continue
            # Availability expression check
            if not desc.is_available(context):
                continue
            # Permission check
            granted, _ = tool.check_permissions(permissions)
            if not granted:
                continue
            tools.append(tool)
        return tools

    def build_tool_plan(
        self, phase: AuditPhase,
        available_fact_types: set[FactType] | None = None,
        agent_type: str | None = None,
        permissions: set[str] | None = None,
    ) -> ToolPlan:
        """
        Build a complete tool plan for the given phase.

        Returns a ToolPlan with:
        - visible: tools that are available in this context
        - hidden: tools that are NOT available, with diagnostic reasons

        This provides full observability into *why* a tool is or isn't available,
        which is critical for debugging audit strategy and understanding agent capability boundaries.
        """
        if available_fact_types is None:
            available_fact_types = set()
        if permissions is None:
            permissions = set()

        context = self._build_context(phase, agent_type, available_fact_types, permissions)
        visible: list[ToolDescriptor] = []
        hidden: list[HiddenToolDiagnostic] = []

        for name, desc in self._descriptors.items():
            if desc.is_available(context):
                # Additional permission check for tool implementation
                tool = self._tools[name]
                perm_granted, perm_reason = tool.check_permissions(permissions)
                if perm_granted:
                    visible.append(desc)
                else:
                    hidden.append(HiddenToolDiagnostic(
                        tool_name=name,
                        reason=perm_reason,
                    ))
            else:
                # Collect the reason from descriptor's diagnosis
                diagnosis = desc.availability_diagnosis(context)
                hidden.append(HiddenToolDiagnostic(
                    tool_name=name,
                    reason=diagnosis,
                ))

        return ToolPlan(
            phase=phase,
            visible=tuple(visible),
            hidden=tuple(hidden),
        )

    def _build_context(
        self,
        phase: AuditPhase,
        agent_type: str | None,
        available_fact_types: set[FactType],
        permissions: set[str],
    ) -> ToolContext:
        """Build a ToolContext for the given runtime parameters."""
        # Determine the risk cap for this phase
        risk_cap = PHASE_RISK_CAP.get(phase, RiskLevel.SAFE)

        return ToolContext(
            phase=phase,
            agent_type=agent_type,
            available_fact_types=available_fact_types,
            permissions=permissions,
            current_risk_cap=risk_cap,
        )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        names = ", ".join(self._tools.keys())
        return f"AuditToolRegistry([{names}])"