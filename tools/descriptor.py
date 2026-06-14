"""
VulnGuard Declarative Tool Descriptor System.

Adapted from OpenClaw's ToolDescriptor + ToolAvailabilityExpression design
for vulnerability audit scenarios. Adds risk-level classification, audit-phase
filtering, and fact-type dependency declarations.

Design principles:
- Tools describe *what they need*, not *when they run*
- The registry evaluates availability expressions against runtime context
- Risk levels map to audit phases: lower phases restrict dangerous tools
- Concurrency safety is a property of the tool, not the caller
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from ..vulnkb.models import RiskLevel, AuditPhase, FactType


# ──────────────────── Tool Availability Expression ────────────────────


class _AvailabilityExpr(abc.ABC):
    """
    Abstract base for composable availability expressions.

    Inspired by OpenClaw's ToolAvailabilityExpression, but extended with
    audit-domain primitives (phase, risk_level, fact dependencies).
    """

    @abc.abstractmethod
    def evaluate(self, context: ToolContext) -> bool:
        """Evaluate this expression against the given runtime context."""
        ...

    @abc.abstractmethod
    def describe(self) -> str:
        """Human-readable description of this expression (for diagnostics)."""
        ...

    def __and__(self, other: "_AvailabilityExpr") -> "_AvailabilityExpr":
        return AllOfExpr([self, other])

    def __or__(self, other: "_AvailabilityExpr") -> "_AvailabilityExpr":
        return AnyOfExpr([self, other])

    def __invert__(self) -> "_AvailabilityExpr":
        return NotExpr(self)


@dataclass(frozen=True)
class ToolContext:
    """
    Runtime context against which availability expressions are evaluated.

    Provides the 'world state' that determines if a tool is available.
    """
    phase: AuditPhase
    agent_type: Optional[str] = None
    available_fact_types: set[FactType] = field(default_factory=set)
    permissions: set[str] = field(default_factory=set)
    current_risk_cap: RiskLevel = RiskLevel.DANGEROUS
    """Maximum risk level allowed in the current context."""


# -- Leaf expressions --


@dataclass(frozen=True)
class PhaseExpr(_AvailabilityExpr):
    """Tool is available only during the specified audit phase(s)."""
    phases: tuple[AuditPhase, ...]

    def evaluate(self, context: ToolContext) -> bool:
        return context.phase in self.phases

    def describe(self) -> str:
        names = ", ".join(p.value for p in self.phases)
        return f"phase({names})"


@dataclass(frozen=True)
class RiskCapExpr(_AvailabilityExpr):
    """
    Tool is available when the current risk cap permits its risk level.
    risk_level(<=X) means the tool's risk level must be <= X.
    """
    max_risk: RiskLevel

    def evaluate(self, context: ToolContext) -> bool:
        # Phase-based risk caps override explicit caps
        effective_cap = context.current_risk_cap
        return _risk_level_order(effective_cap) >= _risk_level_order(self.max_risk)

    def describe(self) -> str:
        return f"risk_level(<={self.max_risk.value})"


@dataclass(frozen=True)
class AgentTypeExpr(_AvailabilityExpr):
    """Tool is available only for agents of the specified type(s)."""
    agent_types: tuple[str, ...]

    def evaluate(self, context: ToolContext) -> bool:
        if context.agent_type is None:
            return True  # No agent restriction in context means available to all
        return context.agent_type in self.agent_types

    def describe(self) -> str:
        return f"agent_type({', '.join(self.agent_types)})"


@dataclass(frozen=True)
class RequiresExpr(_AvailabilityExpr):
    """
    Tool requires that the knowledge graph contains facts of the specified type(s).
    If multiple fact types are given, ALL must be present (use anyOf for OR).
    """
    fact_types: tuple[FactType, ...]

    def evaluate(self, context: ToolContext) -> bool:
        if not self.fact_types:
            return True
        return all(ft in context.available_fact_types for ft in self.fact_types)

    def describe(self) -> str:
        names = ", ".join(ft.value for ft in self.fact_types)
        return f"requires({names})"


@dataclass(frozen=True)
class HasPermissionExpr(_AvailabilityExpr):
    """Tool requires a specific permission to be granted."""
    permission: str

    def evaluate(self, context: ToolContext) -> bool:
        return self.permission in context.permissions

    def describe(self) -> str:
        return f"has_permission({self.permission})"


# -- Composite expressions --


@dataclass(frozen=True)
class AllOfExpr(_AvailabilityExpr):
    """All sub-expressions must be True (logical AND)."""
    expressions: tuple[_AvailabilityExpr, ...]

    def evaluate(self, context: ToolContext) -> bool:
        return all(expr.evaluate(context) for expr in self.expressions)

    def describe(self) -> str:
        inner = " AND ".join(expr.describe() for expr in self.expressions)
        return f"allOf({inner})"


@dataclass(frozen=True)
class AnyOfExpr(_AvailabilityExpr):
    """At least one sub-expression must be True (logical OR)."""
    expressions: tuple[_AvailabilityExpr, ...]

    def evaluate(self, context: ToolContext) -> bool:
        return any(expr.evaluate(context) for expr in self.expressions)

    def describe(self) -> str:
        inner = " OR ".join(expr.describe() for expr in self.expressions)
        return f"anyOf({inner})"


@dataclass(frozen=True)
class NotExpr(_AvailabilityExpr):
    """Negate a sub-expression."""
    expression: _AvailabilityExpr

    def evaluate(self, context: ToolContext) -> bool:
        return not self.expression.evaluate(context)

    def describe(self) -> str:
        return f"NOT({self.expression.describe()})"


# Convenience constructor functions (fluent API, matching the design spec)


def phase(*phases: AuditPhase) -> PhaseExpr:
    """Create a phase availability expression."""
    return PhaseExpr(phases=tuple(phases))


def risk_level(max_risk: RiskLevel) -> RiskCapExpr:
    """Create a risk level cap expression."""
    return RiskCapExpr(max_risk=max_risk)


def agent_type(*types: str) -> AgentTypeExpr:
    """Create an agent type availability expression."""
    return AgentTypeExpr(agent_types=tuple(types))


def requires(*fact_types: FactType) -> RequiresExpr:
    """Create a fact-type dependency expression."""
    return RequiresExpr(fact_types=tuple(fact_types))


def all_of(*expressions: _AvailabilityExpr) -> AllOfExpr:
    """Create an allOf (AND) composite expression."""
    return AllOfExpr(expressions=tuple(expressions))


def any_of(*expressions: _AvailabilityExpr) -> AnyOfExpr:
    """Create an anyOf (OR) composite expression."""
    return AnyOfExpr(expressions=tuple(expressions))


# Public alias matching the design spec name
ToolAvailabilityExpression = _AvailabilityExpr


# ──────────────────── Risk Level Ordering ────────────────────


_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.SAFE: 0,
    RiskLevel.READ_ONLY: 1,
    RiskLevel.ANALYSIS: 2,
    RiskLevel.WRITE: 3,
    RiskLevel.DANGEROUS: 4,
}


def _risk_level_order(level: RiskLevel) -> int:
    return _RISK_ORDER.get(level, 99)


# ──────────────────── Tool Descriptor ────────────────────


@dataclass(frozen=True)
class ToolDescriptor:
    """
    Declarative description of a VulnGuard audit tool.

    Inspired by OpenClaw's ToolDescriptor, but augmented with:
    - risk_level: vulnerability audit risk classification
    - availability: composable expression governing when the tool is visible
    - audit_phases: which audit phases this tool participates in

    The descriptor is the *single source of truth* for a tool's metadata.
    Runtime behavior (the actual implementation) is provided by AuditTool subclass.
    """
    name: str
    description: str
    input_schema: dict[str, Any]          # JSON Schema for inputs
    output_schema: dict[str, Any]         # JSON Schema for outputs
    risk_level: RiskLevel
    owner: str                            # Agent type that owns this tool (e.g. "code_intel_miner")
    availability: Optional[ToolAvailabilityExpression] = None
    audit_phases: tuple[AuditPhase, ...] = (
        AuditPhase.CODE_INTELLIGENCE,
        AuditPhase.VULN_MINING,
        AuditPhase.VERIFICATION,
    )

    def is_available(self, context: ToolContext) -> bool:
        """Check if this tool is available in the given context."""
        if self.availability is None:
            # Default: check phase + risk cap
            return (context.phase in self.audit_phases
                    and _risk_level_order(self.risk_level) <= _risk_level_order(context.current_risk_cap))
        return self.availability.evaluate(context)

    def availability_diagnosis(self, context: ToolContext) -> str:
        """Produce a human-readable diagnosis of why this tool is/isn't available."""
        # Phase check
        if context.phase not in self.audit_phases:
            allowed = ", ".join(p.value for p in self.audit_phases)
            return f"Phase {context.phase.value} not in allowed phases [{allowed}]"

        # Risk cap check
        if _risk_level_order(self.risk_level) > _risk_level_order(context.current_risk_cap):
            return (f"Risk level {self.risk_level.value} exceeds current cap "
                    f"{context.current_risk_cap.value}")

        # Custom availability expression
        if self.availability is not None:
            if not self.availability.evaluate(context):
                return f"Custom availability condition not met: {self.availability.describe()}"
            return "Available"

        return "Available"


# ──────────────────── Audit Tool Base Class ────────────────────


class AuditTool(abc.ABC):
    """
    Abstract base class for all VulnGuard audit tools.

    Borrows from Claude Code's tool interface:
    - is_concurrency_safe(): whether multiple invocations can run in parallel
    - is_read_only(): whether this tool only reads (no side effects)
    - is_destructive(): whether this tool can cause irreversible changes
    - check_permissions(): validate that required permissions are present

    Extended for audit domain:
    - risk_level: maps to which audit phases can use this tool
    - audit_phases: explicit phase membership
    - call(): the actual tool execution
    """

    @property
    @abc.abstractmethod
    def descriptor(self) -> ToolDescriptor:
        """Return this tool's descriptor."""
        ...

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def risk_level(self) -> RiskLevel:
        return self.descriptor.risk_level

    @property
    def audit_phases(self) -> tuple[AuditPhase, ...]:
        return self.descriptor.audit_phases

    @abc.abstractmethod
    async def call(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        ...

    def is_concurrency_safe(self) -> bool:
        """
        Whether multiple invocations of this tool can safely run concurrently.

        Concurrency-safe tools are executed in parallel by StreamingToolExecutor.
        Non-safe tools are serialized to prevent conflicts.

        Default: True for SAFE and READ_ONLY, False for ANALYSIS and above.
        """
        return _risk_level_order(self.risk_level) <= _risk_level_order(RiskLevel.READ_ONLY)

    def is_read_only(self) -> bool:
        """Whether this tool only reads data without modifying anything."""
        return _risk_level_order(self.risk_level) <= _risk_level_order(RiskLevel.READ_ONLY)

    def is_destructive(self) -> bool:
        """
        Whether this tool can cause irreversible changes.

        Destructive tools (WRITE, DANGEROUS) require special permissions
        and are only available in VERIFICATION phase.
        """
        return _risk_level_order(self.risk_level) >= _risk_level_order(RiskLevel.WRITE)

    def check_permissions(self, permissions: set[str]) -> tuple[bool, str]:
        """
        Validate that the required permissions are granted for this tool.

        Returns (granted, reason) tuple.
        """
        # Default: destructive tools need explicit "destructive" permission
        if self.is_destructive():
            if "destructive" not in permissions:
                return False, f"Tool '{self.name}' requires 'destructive' permission"
        # WRITE tools need "write" permission
        if self.risk_level == RiskLevel.WRITE:
            if "write" not in permissions:
                return False, f"Tool '{self.name}' requires 'write' permission"
        return True, ""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} risk={self.risk_level.value}>"