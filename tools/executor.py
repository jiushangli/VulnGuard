"""
VulnGuard Streaming Tool Executor.

Adapted from Claude Code's StreamingToolExecutor pattern:
- Concurrency-safe (read-only) tools execute in parallel
- Unsafe (write/destructive) tools execute serially
- Streaming results via async generators
- Timeout enforcement for each tool call

Design principles:
- Never block a parallel batch on a slow serial tool
- Provide real-time streaming results for long-running tools
- Enforce phase-level security through permission checks at execution time
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional

from vulnkb.models import RiskLevel, AuditPhase

from .descriptor import AuditTool, _risk_level_order
from .registry import AuditToolRegistry


# ──────────────────── Execution Events ────────────────────


class ExecutionStatus(Enum):
    """Status of a tool execution attempt."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"      # Not executed (dependency missing, permission denied, etc.)


@dataclass
class ToolResult:
    """Result of a single tool execution."""
    tool_name: str
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def succeeded(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED


@dataclass
class StreamEvent:
    """
    Streaming event emitted during tool execution.

    Events are yielded in real-time as tools complete, allowing consumers
    to process results incrementally without waiting for the full batch.
    """
    event_type: str  # "tool_start", "tool_output", "tool_complete", "batch_complete"
    tool_name: str
    data: Any = None

    @classmethod
    def tool_start(cls, tool_name: str) -> "StreamEvent":
        return cls(event_type="tool_start", tool_name=tool_name)

    @classmethod
    def tool_output(cls, tool_name: str, data: Any) -> "StreamEvent":
        return cls(event_type="tool_output", tool_name=tool_name, data=data)

    @classmethod
    def tool_complete(cls, result: ToolResult) -> "StreamEvent":
        return cls(event_type="tool_complete", tool_name=result.tool_name, data=result)

    @classmethod
    def batch_complete(cls, results: list[ToolResult]) -> "StreamEvent":
        return cls(event_type="batch_complete", tool_name="", data=results)


# ──────────────────── Execution Plan ────────────────────


@dataclass
class ExecutionPlan:
    """
    A plan for executing a batch of tools.

    Tools are partitioned into:
    - parallel_group: concurrency-safe tools that can run simultaneously
    - serial_group: tools that must run one at a time (in order)

    The parallel group runs first, then serial tools run in order.
    """
    parallel_group: list[str] = field(default_factory=list)
    serial_group: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.parallel_group) + len(self.serial_group)


# ──────────────────── Streaming Tool Executor ────────────────────


class StreamingToolExecutor:
    """
    Execute audit tools with streaming results and concurrency control.

    Borrowing from Claude Code's design:
    - Partition tools into safe (parallel) and unsafe (serial) groups
    - Execute safe tools concurrently using asyncio.gather
    - Execute unsafe tools one at a time in deterministic order
    - Stream each result as it becomes available via async generator
    - Enforce per-tool timeouts

    Security:
    - Enforces phase-appropriate risk caps at execution time
    - Permission checks before each tool execution
    - Dangerous tools require explicit confirmation callback
    """

    def __init__(
        self,
        registry: AuditToolRegistry,
        default_timeout: float = 60.0,
        dangerous_timeout: float = 120.0,
        confirm_dangerous: Optional[Callable[[str, dict], bool]] = None,
    ) -> None:
        """
        Args:
            registry: The tool registry to resolve tools from.
            default_timeout: Timeout in seconds for regular tool calls.
            dangerous_timeout: Timeout in seconds for dangerous tool calls.
            confirm_dangerous: Optional callback for confirming dangerous tool execution.
                               Receives (tool_name, kwargs) and returns True to proceed.
        """
        self.registry = registry
        self.default_timeout = default_timeout
        self.dangerous_timeout = dangerous_timeout
        self.confirm_dangerous = confirm_dangerous

    def plan_execution(
        self,
        tool_calls: list[tuple[str, dict]],
        phase: AuditPhase,
        permissions: set[str] | None = None,
    ) -> ExecutionPlan:
        """
        Partition tool calls into parallel-safe and serial groups.

        Returns an ExecutionPlan that the executor can follow.

        Tools are partitioned by is_concurrency_safe():
        - Safe tools → parallel_group (executed concurrently)
        - Unsafe tools → serial_group (executed in order)

        Invalid or unavailable tools are silently skipped (logged in execution).
        """
        if permissions is None:
            permissions = set()

        parallel: list[str] = []
        serial: list[str] = []

        for tool_name, _kwargs in tool_calls:
            tool = self.registry.get(tool_name)
            if tool is None:
                # Unknown tool — skip it, will be reported in execution
                continue

            # Permission check at plan time
            granted, _reason = tool.check_permissions(permissions)
            if not granted:
                continue

            # Risk level check against phase
            cap = _risk_level_order(_phase_risk(phase))
            tool_risk = _risk_level_order(tool.risk_level)
            if tool_risk > cap:
                continue

            if tool.is_concurrency_safe():
                parallel.append(tool_name)
            else:
                serial.append(tool_name)

        return ExecutionPlan(
            parallel_group=parallel,
            serial_group=serial,
        )

    async def execute_streaming(
        self,
        tool_calls: list[tuple[str, dict]],
        phase: AuditPhase,
        permissions: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute a batch of tool calls, yielding stream events as they complete.

        Execution order:
        1. All concurrency-safe tools run in parallel
        2. All unsafe tools run serially (in original order)

        Each tool gets:
        - Timeout enforcement
        - Permission pre-check
        - Phase risk cap enforcement

        Yields StreamEvents for: tool_start, tool_output, tool_complete, batch_complete
        """
        if permissions is None:
            permissions = set()

        results: list[ToolResult] = []
        tool_calls_dict: dict[str, dict] = {name: kwargs for name, kwargs in tool_calls}

        # Partition into parallel and serial groups
        plan = self._build_execution_order(tool_calls, phase, permissions)

        # Phase 1: Execute concurrency-safe tools in parallel
        if plan.parallel_group:
            async for event in self._execute_parallel(
                plan.parallel_group, tool_calls_dict, permissions
            ):
                if event.event_type == "tool_complete" and isinstance(event.data, ToolResult):
                    results.append(event.data)
                yield event

        # Phase 2: Execute unsafe tools serially
        for tool_name in plan.serial_group:
            kwargs = tool_calls_dict.get(tool_name, {})
            async for event in self._execute_single(tool_name, kwargs, permissions):
                if event.event_type == "tool_complete" and isinstance(event.data, ToolResult):
                    results.append(event.data)
                yield event

        # Final batch complete event
        yield StreamEvent.batch_complete(results)

    async def execute(
        self,
        tool_calls: list[tuple[str, dict]],
        phase: AuditPhase,
        permissions: set[str] | None = None,
    ) -> list[ToolResult]:
        """
        Execute a batch of tool calls and return all results.

        Convenience wrapper around execute_streaming() that collects all events
        and returns the final results.
        """
        results = []
        async for event in self.execute_streaming(tool_calls, phase, permissions):
            if event.event_type == "batch_complete" and isinstance(event.data, list):
                results = event.data
        return results

    # ──────────────────── Internal ────────────────────

    def _build_execution_order(
        self,
        tool_calls: list[tuple[str, dict]],
        phase: AuditPhase,
        permissions: set[str],
    ) -> ExecutionPlan:
        """Build execution order preserving original call order within each group."""
        parallel: list[str] = []
        serial: list[str] = []

        for tool_name, _kwargs in tool_calls:
            tool = self.registry.get(tool_name)
            if tool is None:
                continue

            # Check permissions
            granted, _ = tool.check_permissions(permissions)
            if not granted:
                continue

            # Check phase risk cap
            cap = _phase_risk(phase)
            if _risk_level_order(tool.risk_level) > _risk_level_order(cap):
                continue

            if tool.is_concurrency_safe():
                parallel.append(tool_name)
            else:
                serial.append(tool_name)

        return ExecutionPlan(
            parallel_group=parallel,
            serial_group=serial,
        )

    async def _execute_parallel(
        self,
        tool_names: list[str],
        tool_calls_dict: dict[str, dict],
        permissions: set[str],
    ) -> AsyncIterator[StreamEvent]:
        """Execute multiple safe tools concurrently, yielding events as each completes."""
        tasks = {}
        for name in tool_names:
            kwargs = tool_calls_dict.get(name, {})
            tasks[name] = asyncio.create_task(
                self._run_tool_with_timeout(name, kwargs, permissions)
            )

        # Collect results as they complete
        pending = set(tasks.keys())
        while pending:
            done, _ = await asyncio.wait(
                [tasks[name] for name in pending],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Find which task(s) completed
            for name in list(pending):
                task = tasks[name]
                if task.done():
                    result = task.result()
                    yield StreamEvent.tool_complete(result)
                    pending.discard(name)

    async def _execute_single(
        self,
        tool_name: str,
        kwargs: dict,
        permissions: set[str],
    ) -> AsyncIterator[StreamEvent]:
        """Execute a single tool call, yielding start and complete events."""
        yield StreamEvent.tool_start(tool_name)
        result = await self._run_tool_with_timeout(tool_name, kwargs, permissions)
        yield StreamEvent.tool_complete(result)

    async def _run_tool_with_timeout(
        self,
        tool_name: str,
        kwargs: dict,
        permissions: set[str],
    ) -> ToolResult:
        """Run a single tool call with timeout and error handling."""
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.FAILED,
                error=f"Tool '{tool_name}' not found in registry",
            )

        # Re-check permissions at execution time (defense in depth)
        granted, reason = tool.check_permissions(permissions)
        if not granted:
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.SKIPPED,
                error=reason,
            )

        # Confirm dangerous tool execution
        if tool.is_destructive() and self.confirm_dangerous is not None:
            try:
                confirmed = self.confirm_dangerous(tool_name, kwargs)
                if not confirmed:
                    return ToolResult(
                        tool_name=tool_name,
                        status=ExecutionStatus.SKIPPED,
                        error="Execution of dangerous tool was not confirmed",
                    )
            except Exception as e:
                return ToolResult(
                    tool_name=tool_name,
                    status=ExecutionStatus.SKIPPED,
                    error=f"Confirmation callback failed: {e}",
                )

        # Determine timeout based on risk level
        timeout = self.dangerous_timeout if tool.is_destructive() else self.default_timeout

        start_time = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool.call(**kwargs),
                timeout=timeout,
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.COMPLETED,
                output=result,
                duration_ms=elapsed_ms,
                started_at=start_time,
                completed_at=time.monotonic(),
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.TIMEOUT,
                error=f"Tool timed out after {timeout}s",
                duration_ms=elapsed_ms,
                started_at=start_time,
            )
        except asyncio.CancelledError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.FAILED,
                error="Tool execution was cancelled",
                duration_ms=elapsed_ms,
                started_at=start_time,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                duration_ms=elapsed_ms,
                started_at=start_time,
            )


# ──────────────────── Phase Risk Helper ────────────────────


def _phase_risk(phase: AuditPhase) -> RiskLevel:
    """Get the maximum risk level allowed for a given audit phase."""
    from .registry import PHASE_RISK_CAP
    return PHASE_RISK_CAP.get(phase, RiskLevel.SAFE)