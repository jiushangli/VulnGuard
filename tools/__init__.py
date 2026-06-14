"""
VulnGuard Declarative Tool System.

Adapted from OpenClaw's ToolDescriptor + ToolAvailabilityExpression design,
with extensions for vulnerability audit scenarios:
- Risk-level classification (SAFE → DANGEROUS)
- Audit-phase filtering (CODE_INTELLIGENCE → VERIFICATION)
- Fact-type dependency declarations
- Streaming concurrent execution (from Claude Code pattern)

Core components:
- ToolDescriptor: Declarative tool metadata with availability expressions
- ToolAvailabilityExpression: Composable conditions (phase, risk_level, agent_type, requires, allOf, anyOf)
- AuditTool: Abstract base class for tool implementations
- AuditToolRegistry: Registration, discovery, and phase-based filtering
- StreamingToolExecutor: Parallel-safe concurrent execution with streaming results
"""

# ──── Descriptor System ────

from .descriptor import (
    # Core classes
    ToolDescriptor,
    ToolContext,
    ToolAvailabilityExpression,
    AuditTool,

    # Expression types
    _AvailabilityExpr,
    PhaseExpr,
    RiskCapExpr,
    AgentTypeExpr,
    RequiresExpr,
    HasPermissionExpr,
    AllOfExpr,
    AnyOfExpr,
    NotExpr,

    # Convenience constructors (fluent API)
    phase,
    risk_level,
    agent_type,
    requires,
    all_of,
    any_of,

    # Risk level ordering
    _risk_level_order,
)

# ──── Registry ────

from .registry import (
    AuditToolRegistry,
    ToolPlan,
    HiddenToolDiagnostic,
    PHASE_RISK_CAP,
)

# ──── Executor ────

from .executor import (
    StreamingToolExecutor,
    ExecutionPlan,
    ExecutionStatus,
    ToolResult,
    StreamEvent,
)

__all__ = [
    # Descriptor
    "ToolDescriptor",
    "ToolContext",
    "ToolAvailabilityExpression",
    "AuditTool",
    # Expression types
    "PhaseExpr",
    "RiskCapExpr",
    "AgentTypeExpr",
    "RequiresExpr",
    "HasPermissionExpr",
    "AllOfExpr",
    "AnyOfExpr",
    "NotExpr",
    # Convenience constructors
    "phase",
    "risk_level",
    "agent_type",
    "requires",
    "all_of",
    "any_of",
    # Risk ordering
    "_risk_level_order",
    # Registry
    "AuditToolRegistry",
    "ToolPlan",
    "HiddenToolDiagnostic",
    "PHASE_RISK_CAP",
    # Executor
    "StreamingToolExecutor",
    "ExecutionPlan",
    "ExecutionStatus",
    "ToolResult",
    "StreamEvent",
]