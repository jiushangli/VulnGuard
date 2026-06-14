"""
VulnGuard Agent Base — OODA Loop and Agent Lifecycle.

Core design:
- AgentBase: abstract base class for all VulnGuard agents
- OODA loop: Observe → Orient → Decide → Act (from Cairn)
- Three agent types: Miner (specialized via MinerSpec), Observer, Verifier
- Heartbeat renewal: periodically renew claimed intents to prevent lease expiry
- Fact/Intent/FailureBoundary submission via VulnKB verified admission

OODA cycle integration with VulnKB:
- Observe: Read kb.build_context(level="gist") for current态势 (situation awareness)
- Orient: Combine current intent details with VulnKB context for evaluation
- Decide: Choose which analysis action to perform
- Act: Execute tools, produce Fact/Intent/FailureBoundary outputs

Agent lifecycle:
1. Agent starts, initializes PromptManager with role definition
2. run() enters main loop:
   a. Claim an Intent from VulnKB (claim_task)
   b. Enter OODA cycle:
      - observe(): read VulnKB, gather situational awareness
      - orient(): evaluate current intent against gathered facts
      - decide(): select tool/action to execute
      - act(): execute chosen action, produce outputs
   c. Submit results to VulnKB (submit_fact, submit_intent, submit_failure_boundary)
   d. Heartbeat renewal (heartbeat)
   e. Loop until no more pending intents or shutdown signal
"""

from __future__ import annotations

import abc
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from .vulnkb.models import (
    AuditPhase,
    Fact,
    FactType,
    FailureBoundary,
    Intent,
    IntentStatus,
    MinerSpec,
    VulnKB,
    make_fact,
    make_intent,
)

from .tools.descriptor import AuditTool

from .utils.llm import AgentRole, CompletionResult, LLMGateway
from .utils.prompt import PromptLayer, PromptManager

logger = logging.getLogger(__name__)


# ──────────────────────── Agent Type ────────────────────────


class AgentType(Enum):
    """Types of agents in the VulnGuard framework."""
    MINER = "miner"        # Specialized mining agents (further specified by MinerSpec)
    OBSERVER = "observer"  # Strategic oversight agent
    VERIFIER = "verifier"  # Independent verification agent


# ──────────────────────── OODA Cycle Result ────────────────────────


@dataclass
class OODAResult:
    """
    Result of a single OODA cycle.

    Captures what the agent observed, decided, and produced,
    enabling the main loop to decide whether to continue, course-correct, or stop.
    """
    observations: list[str] = field(default_factory=list)
    """Key observations from the observe phase."""

    orientation: str = ""
    """Assessment from the orient phase."""

    decision: str = ""
    """Chosen action from the decide phase."""

    action_output: Any = None
    """Output from the act phase (tool results, facts, etc.)."""

    facts_produced: list[Fact] = field(default_factory=list)
    """Facts submitted to VulnKB this cycle."""

    intents_produced: list[Intent] = field(default_factory=list)
    """New intents generated this cycle."""

    boundaries_produced: list[FailureBoundary] = field(default_factory=list)
    """Failure boundaries established this cycle."""

    should_continue: bool = True
    """Whether the agent should continue the OODA loop."""

    cycle_number: int = 0
    """Which OODA cycle this result is from."""


# ──────────────────────── Agent Config ────────────────────────


@dataclass
class AgentConfig:
    """
    Configuration for an AgentBase instance.

    Covers timing, limits, and behavioral parameters for the OODA loop.
    """
    max_ooda_cycles: int = 50
    """Maximum number of OODA cycles per intent before forcing completion."""

    heartbeat_interval_seconds: int = 60
    """Seconds between heartbeat renewals for claimed intents."""

    claim_lease_seconds: int = 300
    """Lease duration when claiming an intent (5 minutes default)."""

    max_consecutive_failures: int = 3
    """Max consecutive OODA cycle failures before giving up on an intent."""

    idle_poll_interval_seconds: int = 5
    """Seconds to wait between polls when no intent is available."""

    volatile_update_on_observe: bool = True
    """Whether to automatically update the prompt volatile layer during observe."""

    context_update_on_orient: bool = True
    """Whether to automatically update the prompt context layer during orient."""


# ──────────────────────── AgentBase ────────────────────────


class AgentBase(abc.ABC):
    """
    Abstract base class for all VulnGuard agents.

    Implements the OODA (Observe-Orient-Decide-Act) loop pattern from Cairn,
    integrated with VulnKB for shared knowledge and task coordination.

    Subclasses must implement the four OODA phases:
    - observe(): Read from VulnKB, gather situational awareness
    - orient(): Evaluate current intent and context, assess options
    - decide(): Choose which action/tool to execute
    - act(): Execute the chosen action, produce outputs

    The base class handles:
    - Intent claiming and lifecycle (claim_task)
    - Result submission (submit_fact, submit_intent, submit_failure_boundary)
    - Heartbeat renewal
    - Prompt management (three-layer caching)
    - Error handling and cycle management

    Attributes:
        agent_id: Unique identifier for this agent instance.
        agent_type: The type of agent (MINER/OBSERVER/VERIFIER).
        specialization: For MINER agents, the MinerSpec specialization.
        kb: Reference to the shared VulnKB instance.
        llm: LLM Gateway for LLM calls routed by agent role.
        tools: List of audit tools available to this agent.
        prompt_manager: Three-layer prompt caching manager.
        config: Agent configuration parameters.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        agent_type: AgentType = AgentType.MINER,
        specialization: MinerSpec | None = None,
        kb: VulnKB | None = None,
        llm: LLMGateway | None = None,
        tools: list[AuditTool] | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.agent_id: str = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.agent_type: AgentType = agent_type
        self.specialization: MinerSpec | None = specialization
        self.kb: VulnKB | None = kb
        self.llm: LLMGateway | None = llm
        self.tools: list[AuditTool] = tools or []

        self.prompt_manager: PromptManager = PromptManager()
        self.config: AgentConfig = config or AgentConfig()

        # Runtime state
        self._current_intent: Intent | None = None
        self._current_phase: AuditPhase = AuditPhase.CODE_INTELLIGENCE
        self._running: bool = False
        self._consecutive_failures: int = 0
        self._cycle_count: int = 0
        self._total_facts: int = 0
        self._total_intents: int = 0
        self._total_boundaries: int = 0

        # Determine agent role for LLM routing
        if agent_type == AgentType.MINER:
            self._agent_role = AgentRole.MINER
        elif agent_type == AgentType.OBSERVER:
            self._agent_role = AgentRole.OBSERVER
        elif agent_type == AgentType.VERIFIER:
            self._agent_role = AgentRole.VERIFIER
        else:
            self._agent_role = AgentRole.MINER

    # ──────────────────────── Properties ────────────────────────

    @property
    def current_intent(self) -> Intent | None:
        """The intent currently being worked on, if any."""
        return self._current_intent

    @property
    def current_phase(self) -> AuditPhase:
        """The current audit phase."""
        return self._current_phase

    @property
    def is_running(self) -> bool:
        """Whether the agent's main loop is currently running."""
        return self._running

    @property
    def stats(self) -> dict[str, Any]:
        """Agent execution statistics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "specialization": self.specialization.value if self.specialization else None,
            "cycle_count": self._cycle_count,
            "total_facts": self._total_facts,
            "total_intents": self._total_intents,
            "total_boundaries": self._total_boundaries,
            "consecutive_failures": self._consecutive_failures,
        }

    # ──────────────────────── Prompt Initialization ────────────────────────

    def _initialize_prompt_stable(self) -> None:
        """
        Initialize the STABLE layer of the prompt manager.

        This is called once at agent startup. The stable layer must remain
        byte-identical for the entire session to ensure prompt cache hits.
        Subclasses should override _get_stable_content() to customize.
        """
        role_def, framework, constraints, output_fmt = self._get_stable_content()
        self.prompt_manager.set_stable(
            role_definition=role_def,
            audit_framework=framework,
            security_constraints=constraints,
            output_format=output_fmt,
        )

    def _get_stable_content(self) -> tuple[str, str, str, str]:
        """
        Generate stable layer content for this agent.

        Returns:
            Tuple of (role_definition, audit_framework, security_constraints, output_format)
        """
        spec_str = self.specialization.value if self.specialization else "general"

        role_definition = (
            f"You are a VulnGuard {self.agent_type.value} agent "
            f"(specialization: {spec_str}). "
            f"Your agent_id is '{self.agent_id}'. "
            f"Your role is to perform white-box code security audit "
            f"using the OODA (Observe-Orient-Decide-Act) loop pattern."
        )

        audit_framework = (
            "You interact with a shared Vulnerability Knowledge Base (VulnKB) that stores:\n"
            "- Facts: Immutable observations about the codebase (append-only)\n"
            "- Intents: Directed exploration tasks with lifecycle states (pending→claimed→completed/failed)\n"
            "- Hints: External knowledge (OWASP, CWE patterns, user guidance)\n"
            "- Failure Boundaries: Precise descriptions of ruled-out attack directions\n\n"
            "All writes to VulnKB go through verified admission to prevent error propagation.\n"
            "You claim Intents from the task queue, work on them through the OODA loop, "
            "and submit Facts/Intents/FailureBoundaries as outputs."
        )

        security_constraints = (
            "SECURITY CONSTRAINTS:\n"
            "1. Never modify or delete existing Facts — they are immutable and append-only\n"
            "2. All writes to VulnKB must go through verified admission\n"
            "3. Never exfiltrate source code or sensitive data\n"
            "4. Use FailureBoundaries to record ruled-out attack directions precisely\n"
            "5. Maintain independence: do not bias verification based on mining results\n"
            "6. Report confidence scores honestly — overconfidence is dangerous\n"
            "7. When uncertain, submit hypotheses as VULN_HYPOTHESIS facts, not VULNERABILITY facts"
        )

        output_format = (
            "OUTPUT FORMAT:\n"
            "- When producing Facts, use the submit_fact() method\n"
            "- When generating new Intents, use the submit_intent() method\n"
            "- When ruling out attack directions, use the submit_failure_boundary() method\n"
            "- Include parent_intent IDs to maintain causal provenance"
        )

        return role_definition, audit_framework, security_constraints, output_format

    # ──────────────────────── OODA Abstract Methods ────────────────────────

    @abc.abstractmethod
    async def observe(self) -> dict[str, Any]:
        """
        Observe phase of the OODA loop.

        Read from VulnKB to gather situational awareness:
        - Call kb.build_context(level="gist") for current态势
        - Review relevant facts, hints, and the current state
        - Identify new information since last cycle

        Returns:
            Dictionary of observations (key topics → observation summaries).
        """
        ...

    @abc.abstractmethod
    async def orient(self, observations: dict[str, Any]) -> dict[str, Any]:
        """
        Orient phase of the OODA loop.

        Combine observations with current intent details and evaluate:
        - What does the current intent ask us to explore?
        - What facts are already known about this area?
        - What hints or patterns are relevant?
        - Are there course corrections from the Observer?

        Args:
            observations: Output from the observe phase.

        Returns:
            Dictionary containing 'assessment' (str) and any derived insights.
        """
        ...

    @abc.abstractmethod
    async def decide(self, orientation: dict[str, Any]) -> dict[str, Any]:
        """
        Decide phase of the OODA loop.

        Choose which analysis action to perform:
        - Select a tool or analytical approach
        - Determine parameters for the chosen action
        - Consider risk level and audit phase constraints

        Args:
            orientation: Output from the orient phase.

        Returns:
            Dictionary with 'action' (str) and 'parameters' (dict).
        """
        ...

    @abc.abstractmethod
    async def act(self, decision: dict[str, Any]) -> OODAResult:
        """
        Act phase of the OODA loop.

        Execute the chosen action:
        - Run the selected tool
        - Process results
        - Produce Facts, Intents, or FailureBoundaries
        - Submit outputs to VulnKB

        Args:
            decision: Output from the decide phase.

        Returns:
            OODAResult capturing what was produced this cycle.
        """
        ...

    # ──────────────────────── Main Loop ────────────────────────

    async def run(self) -> None:
        """
        Main agent loop.

        Lifecycle:
        1. Initialize prompt stable layer
        2. Enter main loop:
           a. Claim an Intent from VulnKB
           b. If no intent available, wait and retry
           c. Enter OODA cycle loop:
              - observe → orient → decide → act
              - Submit results to VulnKB
              - Heartbeat renewal
              - Check termination conditions
           d. Complete or fail the intent
        3. Exit cleanly on shutdown or completion
        """
        if self.kb is None:
            raise RuntimeError(f"Agent {self.agent_id} has no VulnKB reference")

        self._running = True
        logger.info(f"Agent {self.agent_id} ({self.agent_type.value}) starting main loop")

        # Initialize prompt stable layer (frozen for entire session)
        self._initialize_prompt_stable()

        try:
            while self._running:
                # Step 1: Claim a task
                intent = self.claim_task()
                if intent is None:
                    logger.debug(f"Agent {self.agent_id}: no pending intents, waiting...")
                    await asyncio.sleep(self.config.idle_poll_interval_seconds)
                    continue

                self._current_intent = intent
                self._cycle_count = 0
                self._consecutive_failures = 0
                logger.info(
                    f"Agent {self.agent_id} claimed intent {intent.intent_id}: "
                    f"{intent.description}"
                )

                # Step 2: OODA cycle loop
                ooda_result = await self._run_ooda_loop(intent)

                # Step 3: Finalize the intent
                if ooda_result is not None and ooda_result.should_continue:
                    # Intent completed normally
                    if self.kb is not None:
                        self.kb.complete_intent(intent.intent_id)
                    logger.info(
                        f"Agent {self.agent_id} completed intent {intent.intent_id}"
                    )
                else:
                    # Intent failed
                    if self.kb is not None:
                        self.kb.fail_intent(intent.intent_id)
                    logger.warning(
                        f"Agent {self.agent_id} failed intent {intent.intent_id}"
                    )

                self._current_intent = None

        except asyncio.CancelledError:
            logger.info(f"Agent {self.agent_id} cancelled, shutting down")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} fatal error: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info(f"Agent {self.agent_id} main loop exited")

    async def _run_ooda_loop(self, intent: Intent) -> OODAResult | None:
        """
        Run the OODA cycle loop for a claimed intent.

        Continues cycling until:
        - The intent is naturally completed (act sets should_continue=True and no more work)
        - Max cycles reached
        - Max consecutive failures reached
        - Agent is stopped
        """
        last_result: OODAResult | None = None

        for cycle in range(1, self.config.max_ooda_cycles + 1):
            if not self._running:
                break

            self._cycle_count = cycle
            logger.debug(
                f"Agent {self.agent_id} OODA cycle {cycle} "
                f"for intent {intent.intent_id}"
            )

            try:
                # Heartbeat renewal at the start of each cycle
                self.heartbeat()

                # ─── Observe ───
                observations = await self.observe()
                if self.config.volatile_update_on_observe:
                    self._update_volatile_from_observations(observations)

                # ─── Orient ───
                orientation = await self.orient(observations)
                if self.config.context_update_on_orient:
                    self._update_context_from_orientation(orientation)

                # ─── Decide ───
                decision = await self.decide(orientation)

                # ─── Act ───
                result = await self.act(decision)
                result.cycle_number = cycle

                # Track produced outputs
                self._total_facts += len(result.facts_produced)
                self._total_intents += len(result.intents_produced)
                self._total_boundaries += len(result.boundaries_produced)

                # Reset failure counter on success
                if result.facts_produced or result.boundaries_produced:
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1

                # Check termination conditions
                if self._consecutive_failures >= self.config.max_consecutive_failures:
                    logger.warning(
                        f"Agent {self.agent_id}: {self._consecutive_failures} consecutive "
                        f"failures, giving up on intent {intent.intent_id}"
                    )
                    return result

                if not result.should_continue:
                    logger.info(
                        f"Agent {self.agent_id}: OODA loop signaled stop "
                        f"after cycle {cycle}"
                    )
                    return result

                last_result = result

            except Exception as e:
                self._consecutive_failures += 1
                logger.error(
                    f"Agent {self.agent_id} OODA cycle {cycle} error: {e}",
                    exc_info=True,
                )
                if self._consecutive_failures >= self.config.max_consecutive_failures:
                    logger.error(
                        f"Agent {self.agent_id}: max failures reached, "
                        f"abandoning intent {intent.intent_id}"
                    )
                    return last_result

                # Brief backoff before retry
                await asyncio.sleep(min(2 ** self._consecutive_failures, 30))

        if cycle >= self.config.max_ooda_cycles:
            logger.warning(
                f"Agent {self.agent_id}: max OODA cycles ({self.config.max_ooda_cycles}) "
                f"reached for intent {intent.intent_id}"
            )

        return last_result

    # ──────────────────────── Prompt Updates ────────────────────────

    def _update_volatile_from_observations(self, observations: dict[str, Any]) -> None:
        """
        Update the volatile prompt layer based on observe() output.

        Subclass can override for custom behavior.
        """
        obs_lines = []
        for key, value in observations.items():
            obs_lines.append(f"- {key}: {value}")
        tool_results_str = "\n".join(obs_lines) if obs_lines else "No new observations"

        self.prompt_manager.update_volatile(
            tool_results=tool_results_str,
            corrections=self.prompt_manager.layer_content(PromptLayer.VOLATILE) if False else "",
        )

    def _update_context_from_orientation(self, orientation: dict[str, Any]) -> None:
        """
        Update the context prompt layer based on orient() output.

        Subclass can override for custom behavior.
        """
        # Update intent description in context if we have orientation assessment
        assessment = orientation.get("assessment", "")
        if assessment and self._current_intent:
            current_intent_str = (
                f"Intent: {self._current_intent.description}\n"
                f"Assessment: {assessment}"
            )
            self.prompt_manager.update_context(
                phase=self._current_phase.value,
                current_intent=current_intent_str,
            )

    def _refresh_kb_context(self) -> None:
        """
        Refresh the VulnKB gist in the prompt context layer.
        Called when the context needs to be synchronized with VulnKB state.
        """
        if self.kb is None:
            return

        kb_gist = self.kb.build_context(level="gist")
        intent_desc = ""
        if self._current_intent:
            intent_desc = (
                f"Intent ID: {self._current_intent.intent_id}\n"
                f"Description: {self._current_intent.description}\n"
                f"Priority: {self._current_intent.priority}"
            )

        self.prompt_manager.update_context(
            phase=self._current_phase.value,
            kb_gist=kb_gist,
            current_intent=intent_desc,
        )

    # ──────────────────────── VulnKB Interaction ────────────────────────

    def claim_task(self, specialization: MinerSpec | None = None) -> Intent | None:
        """
        Claim a pending intent from the VulnKB task queue.

        Args:
            specialization: Optional MinerSpec to filter for matching intents.
                           Defaults to the agent's own specialization.

        Returns:
            The claimed Intent, or None if no suitable intent is available.
        """
        if self.kb is None:
            logger.warning(f"Agent {self.agent_id}: no VulnKB, cannot claim task")
            return None

        spec = specialization or self.specialization
        intent = self.kb.claim_intent(
            agent_id=self.agent_id,
            specialization=spec,
            lease_seconds=self.config.claim_lease_seconds,
        )

        if intent is not None:
            logger.info(
                f"Agent {self.agent_id} claimed intent {intent.intent_id}: "
                f"{intent.description}"
            )
        else:
            logger.debug(f"Agent {self.agent_id}: no pending intents available")

        return intent

    def submit_fact(
        self,
        fact_type: FactType,
        content: str,
        evidence: str = "",
        confidence: float = 1.0,
        parent_intents: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Fact | None:
        """
        Submit a Fact to VulnKB through verified admission.

        Args:
            fact_type: Type of fact (e.g., DATAFLOW, VULN_HYPOTHESIS).
            content: Compact gist of the fact (≤220 tokens).
            evidence: Detailed evidence supporting the fact.
            confidence: Confidence score [0.0, 1.0].
            parent_intents: IDs of intents that led to this fact.
            metadata: Additional key-value metadata.

        Returns:
            The admitted Fact, or None if admission was denied.
        """
        if self.kb is None:
            logger.warning(f"Agent {self.agent_id}: no VulnKB, cannot submit fact")
            return None

        # Default parent_intents to the current intent
        if parent_intents is None and self._current_intent is not None:
            parent_intents = [self._current_intent.intent_id]

        fact = make_fact(
            fact_type=fact_type,
            content=content,
            source=self.agent_id,
            evidence=evidence,
            confidence=confidence,
            parent_intents=parent_intents or [],
            metadata=metadata or {},
        )

        result = self.kb.add_fact(fact)
        if result.admitted:
            logger.info(
                f"Agent {self.agent_id} submitted fact {fact.fact_id} "
                f"({fact_type.value}): {content[:80]}..."
            )
            return result.fact
        else:
            logger.warning(
                f"Agent {self.agent_id}: fact admission denied: {result.reason}"
            )
            return None

    def submit_intent(
        self,
        description: str,
        from_facts: list[str] | None = None,
        priority: float = 0.5,
        specialization: MinerSpec | None = None,
        metadata: dict | None = None,
    ) -> Intent:
        """
        Submit a new Intent to the VulnKB task queue.

        Args:
            description: What this intent proposes to explore.
            from_facts: IDs of facts that led to proposing this intent.
            priority: Priority score [0.0, 1.0].
            specialization: Preferred miner specialization for this intent.
            metadata: Additional key-value metadata.

        Returns:
            The created Intent.
        """
        if self.kb is None:
            logger.warning(f"Agent {self.agent_id}: no VulnKB, cannot submit intent")
            # Return a non-persisted intent for local tracking
            return make_intent(
                description=description,
                from_facts=from_facts or [],
                priority=priority,
                specialization=specialization,
                metadata=metadata or {},
            )

        # If current intent is set, add it as a parent fact
        effective_from_facts = list(from_facts or [])
        if self._current_intent and self._current_intent.intent_id not in effective_from_facts:
            effective_from_facts.append(self._current_intent.intent_id)

        intent = make_intent(
            description=description,
            from_facts=effective_from_facts,
            priority=priority,
            specialization=specialization,
            metadata=metadata or {},
        )

        self.kb.add_intent(intent)
        self._total_intents += 1
        logger.info(
            f"Agent {self.agent_id} submitted intent {intent.intent_id}: "
            f"{description[:80]}..."
        )
        return intent

    def submit_failure_boundary(
        self,
        vuln_type: str,
        ruled_out: str,
        remaining_risk: str,
        evidence: str,
        confidence: float = 0.8,
    ) -> FailureBoundary:
        """
        Submit a FailureBoundary to VulnKB as a FACT type fact.

        Failure boundaries are stored as FactType.FAILURE_BOUNDARY facts,
        with the FailureBoundary detail embedded in evidence and metadata.

        Args:
            vuln_type: Type of vulnerability tested.
            ruled_out: What has been ruled out and why.
            remaining_risk: What risks still remain.
            evidence: Code evidence supporting this judgment.
            confidence: Confidence in this boundary assessment.

        Returns:
            The FailureBoundary object (also submitted as a fact to VulnKB).
        """
        boundary = FailureBoundary(
            vuln_type=vuln_type,
            ruled_out=ruled_out,
            remaining_risk=remaining_risk,
            evidence=evidence,
            confidence=confidence,
        )

        # Submit as a FAILURE_BOUNDARY fact to VulnKB
        content = (
            f"Ruled out {vuln_type}: {ruled_out}. "
            f"Remaining risk: {remaining_risk}"
        )
        detailed_evidence = (
            f"Ruled out: {ruled_out}\n"
            f"Remaining risk: {remaining_risk}\n"
            f"Evidence: {evidence}"
        )

        self.submit_fact(
            fact_type=FactType.FAILURE_BOUNDARY,
            content=content,
            evidence=detailed_evidence,
            confidence=confidence,
            metadata={"boundary": boundary.__dict__},
        )

        self._total_boundaries += 1
        logger.info(
            f"Agent {self.agent_id} submitted failure boundary for {vuln_type}: "
            f"{ruled_out[:80]}..."
        )
        return boundary

    def heartbeat(self, lease_seconds: int | None = None) -> None:
        """
        Renew the lease for all intents claimed by this agent.

        This prevents the intent from being reclaimed by other agents
        when this agent is still actively working on it.

        Args:
            lease_seconds: Optional custom lease duration. Defaults to config value.
        """
        if self.kb is None:
            return

        duration = lease_seconds or self.config.claim_lease_seconds
        self.kb.heartbeat(self.agent_id, lease_seconds=duration)
        logger.debug(f"Agent {self.agent_id}: heartbeat renewed")

    # ──────────────────────── LLM Interaction ────────────────────────

    async def llm_complete(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        use_cached_prompt: bool = True,
    ) -> CompletionResult:
        """
        Convenience method for LLM completion using the agent's role.

        If messages is None, builds messages from the prompt manager.
        If use_cached_prompt is True, uses three-layer caching.

        Args:
            messages: Optional pre-built messages. If None, uses prompt manager.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Optional stop sequences.
            tools: Optional tool definitions for function calling.
            use_cached_prompt: Whether to use three-layer prompt caching.

        Returns:
            CompletionResult from the LLM.
        """
        if self.llm is None:
            raise RuntimeError(f"Agent {self.agent_id} has no LLM gateway configured")

        if messages is None:
            if use_cached_prompt:
                stable, context, volatile = self.prompt_manager.build_layered_messages()
                messages = self.llm.build_cached_messages(stable, context, volatile)
            else:
                messages = self.prompt_manager.build_messages()

        return await self.llm.complete(
            self._agent_role,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=tools,
            prompt_caching=use_cached_prompt,
        )

    # ──────────────────────── Lifecycle ────────────────────────

    def stop(self) -> None:
        """Signal the agent to stop its main loop."""
        self._running = False
        logger.info(f"Agent {self.agent_id} stop signal received")

    def set_phase(self, phase: AuditPhase) -> None:
        """
        Update the current audit phase.

        This affects the context layer of the prompt and
        which tools are available for use.
        """
        self._current_phase = phase

    def __repr__(self) -> str:
        spec = f" spec={self.specialization.value}" if self.specialization else ""
        return (
            f"<{self.__class__.__name__} id={self.agent_id} "
            f"type={self.agent_type.value}{spec}>"
        )