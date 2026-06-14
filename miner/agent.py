"""
VulnGuard Miner Agent — Specialized OODA-loop vulnerability mining agent.

MinerAgent inherits AgentBase and implements the full OODA cycle:
- observe(): Read VulnKB gist context + claimed Intent details
- orient(): Evaluate current situation based on specialization
- decide(): Choose which analysis tool to invoke next
- act(): Execute the chosen tool, produce Fact/Intent/FailureBoundary

Each Miner has a specialization direction (MinerSpec) that determines:
- Which Intents it claims (matched by specialization)
- Which analysis tools it prefers
- Its system prompt and audit experience heuristics
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from vulnkb.models import (
    AuditPhase,
    Fact,
    FactType,
    FailureBoundary,
    Intent,
    MinerSpec,
    VulnKB,
    make_fact,
    make_intent,
)

from agent_base import AgentBase, AgentConfig, AgentType, OODAResult
from tools.descriptor import AuditTool
from utils.llm import AgentRole, LLMGateway
from utils.prompt import PromptLayer, PromptManager

logger = logging.getLogger(__name__)


# ──────────────────────── Specialization Prompts ────────────────────────

MINER_SPECIALIZATION_PROMPTS: dict[MinerSpec, str] = {
    MinerSpec.API_SEQUENCE: (
        "You specialize in API call sequence analysis. Your expertise includes:\n"
        "- Identifying missing authorization checks between sequential API calls\n"
        "- Detecting business flow manipulation (e.g., skip payment → confirm order)\n"
        "- Finding TOCTOU vulnerabilities in multi-step operations\n"
        "- Recognizing privilege escalation through unexpected call chains\n"
        "- Mapping API endpoint dependencies and their security implications\n\n"
        "Preferred tools: code_reader (for tracing call sequences), pattern_matcher "
        "(for detecting known API abuse patterns).\n"
        "Key patterns to look for: BOLA via sequence manipulation, race conditions in "
        "state-changing APIs, missing state validation between steps."
    ),
    MinerSpec.DATAFLOW_TAINT: (
        "You specialize in data flow and taint analysis. Your expertise includes:\n"
        "- Tracing user input from entry points (API params, headers, body) to dangerous sinks\n"
        "- Identifying missing or bypassed sanitization/validation\n"
        "- Recognizing secondary injection vectors (LDAP, XPath, template injection)\n"
        "- Tracking data through serialization/deserialization boundaries\n"
        "- Detecting unsafe data flows through event buses, message queues, caches\n\n"
        "Preferred tools: dataflow_tracer (for source-to-sink propagation), code_reader "
        "(for understanding data transformations).\n"
        "Key patterns to look for: unvalidated input reaching SQL queries, SSRF via "
        "URL parameters, LDAP injection through search inputs, SSTI via template variables."
    ),
    MinerSpec.BUSINESS_LOGIC: (
        "You specialize in business logic vulnerability analysis. Your expertise includes:\n"
        "- Identifying flaws in financial transaction logic (missing amount checks, negative values)\n"
        "- Detecting state machine violations (skipping required states, reverting committed states)\n"
        "- Finding privilege escalation through role manipulation or feature toggles\n"
        "- Recognizing time-based attacks (coupon expiry bypass, subscription manipulation)\n"
        "- Discovering quantity/limit bypasses and race conditions in business rules\n\n"
        "Preferred tools: code_reader (for understanding business rule implementation), "
        "pattern_matcher (for known business logic anti-patterns).\n"
        "Key patterns: integer overflow in financial calculations, missing idempotency "
        "checks, race conditions in concurrent transactions, missing state guards."
    ),
    MinerSpec.ATTACK_SURFACE: (
        "You specialize in attack surface enumeration. Your expertise includes:\n"
        "- Mapping all HTTP endpoints, their methods, auth requirements, and rate limits\n"
        "- Identifying admin/debug endpoints exposed in production\n"
        "- Finding shadow APIs (undocumented, internal-only endpoints exposed externally)\n"
        "- Recognizing CORS misconfigurations and permissive access control\n"
        "- Detecting information disclosure through error messages and verbose responses\n\n"
        "Preferred tools: code_reader (for route and middleware mapping), pattern_matcher "
        "(for known misconfiguration patterns).\n"
        "Key patterns: unauthenticated admin routes, default credentials, debug mode "
        "left enabled, overly permissive CORS, missing rate limiting."
    ),
    MinerSpec.STATE_MACHINE: (
        "You specialize in state machine and workflow analysis. Your expertise includes:\n"
        "- Identifying invalid state transitions (skipping required approval steps)\n"
        "- Detecting missing state guards (operating on entities in wrong states)\n"
        "- Finding rollback/revert vulnerabilities (undoing committed changes)\n"
        "- Recognizing race conditions in state transitions (concurrent status changes)\n"
        "- Discovering authorization bypasses through state manipulation\n\n"
        "Preferred tools: code_reader (for understanding FSM definitions), dataflow_tracer "
        "(for tracking state variable propagation).\n"
        "Key patterns: missing state guards before critical operations, insufficient "
        "transition validation, concurrent state modification without locking, rollback "
        "without proper authorization."
    ),
    MinerSpec.AUTH_MODEL: (
        "You specialize in authentication and authorization model analysis. Your expertise includes:\n"
        "- Identifying broken authentication (session fixation, token predictability)\n"
        "- Detecting privilege escalation (vertical and horizontal)\n"
        "- Finding access control flaws (BOLA, BFLA, missing function-level checks)\n"
        "- Recognizing JWT vulnerabilities (algorithm confusion, weak secrets, none algorithm)\n"
        "- Discovering authorization bypasses through parameter tampering or force browsing\n\n"
        "Preferred tools: code_reader (for understanding auth middleware), dataflow_tracer "
        "(for tracking permission checks), pattern_matcher (for known auth anti-patterns).\n"
        "Key patterns: missing auth checks on admin endpoints, JWT none algorithm, "
        "permission checks using client-side data, IDOR through parameter manipulation."
    ),
}

# Tool preference per specialization
MINER_TOOL_PREFERENCES: dict[MinerSpec, list[str]] = {
    MinerSpec.API_SEQUENCE: ["code_reader", "pattern_matcher"],
    MinerSpec.DATAFLOW_TAINT: ["dataflow_tracer", "code_reader"],
    MinerSpec.BUSINESS_LOGIC: ["code_reader", "pattern_matcher"],
    MinerSpec.ATTACK_SURFACE: ["code_reader", "pattern_matcher"],
    MinerSpec.STATE_MACHINE: ["code_reader", "dataflow_tracer"],
    MinerSpec.AUTH_MODEL: ["code_reader", "dataflow_tracer", "pattern_matcher"],
}


# ──────────────────────── MinerAgent ────────────────────────


class MinerAgent(AgentBase):
    """
    Specialized OODA-loop mining agent.

    MinerAgent claims Intents from VulnKB that match its specialization,
    then executes OODA cycles to discover Facts (vulnerability hypotheses,
    data flows, security controls), generate new Intents for deeper
    exploration, and record FailureBoundaries for ruled-out directions.

    The specialization determines:
    - Which Intents the agent claims (filtered by MinerSpec)
    - The system prompt and audit heuristics
    - Preferred analysis tools
    """

    def __init__(
        self,
        specialization: MinerSpec,
        agent_id: str | None = None,
        kb: VulnKB | None = None,
        llm: LLMGateway | None = None,
        tools: list[AuditTool] | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.MINER,
            specialization=specialization,
            kb=kb,
            llm=llm,
            tools=tools or [],
            config=config or AgentConfig(),
        )
        self.specialization: MinerSpec = specialization
        self._preferred_tools: list[str] = MINER_TOOL_PREFERENCES.get(
            specialization, ["code_reader"]
        )
        self._specialization_prompt: str = MINER_SPECIALIZATION_PROMPTS.get(
            specialization, ""
        )
        # Track last known fact count to detect new information
        self._last_fact_count: int = 0

    # ──────────────────────── Prompt ────────────────────────

    def _get_stable_content(self) -> tuple[str, str, str, str]:
        """
        Generate stable layer content with specialization-specific prompt.

        Extends the base stable content with the miner's specialization
        direction and audit experience heuristics.
        """
        role_definition, audit_framework, security_constraints, output_format = (
            super()._get_stable_content()
        )

        # Augment role definition with specialization
        role_definition += (
            f"\n\n## Specialization: {self.specialization.value}\n"
            f"{self._specialization_prompt}"
        )

        # Add specialization-specific output guidance
        output_format += (
            "\n\n## Miner-Specific Output Guidelines\n"
            "- When you discover a potential vulnerability, submit it as a VULN_HYPOTHESIS fact, "
            "NOT a VULNERABILITY fact. Only the Verifier can confirm vulnerabilities.\n"
            "- When you identify a new exploration direction, submit an Intent with appropriate "
            f"specialization hint (not necessarily {self.specialization.value}).\n"
            "- When you can confidently rule out an attack direction, submit a FailureBoundary "
            "with precise remaining risk description.\n"
            "- Always include evidence (code references, data flow details) in your facts.\n"
            "- Set confidence scores honestly; overconfidence is dangerous in security audit."
        )

        return role_definition, audit_framework, security_constraints, output_format

    # ──────────────────────── OODA: Observe ────────────────────────

    async def observe(self) -> dict[str, Any]:
        """
        Observe phase: Read VulnKB gist context + current Intent details.

        Gathers:
        - VulnKB gist-level context (current facts, intents, hints at a glance)
        - Current claimed Intent details (description, priority, parent facts)
        - Relevant facts related to the current Intent
        - New facts since last observation cycle
        - Available tools and their current context
        """
        observations: dict[str, Any] = {}

        if self.kb is None:
            observations["error"] = "No VulnKB reference available"
            return observations

        # 1. Build gist-level context snapshot
        kb_gist = self.kb.build_context(level="gist")
        observations["kb_gist"] = kb_gist
        observations["total_facts"] = self.kb.get_fact_count()
        observations["pending_intents"] = self.kb.get_pending_count()

        # 2. Current Intent details
        if self._current_intent is not None:
            intent = self._current_intent
            observations["current_intent"] = {
                "intent_id": intent.intent_id,
                "description": intent.description,
                "priority": intent.priority,
                "from_facts": intent.from_facts,
                "specialization": intent.specialization.value if intent.specialization else None,
            }

            # 3. Facts that are parents of the current Intent
            parent_facts = []
            for fact_id in intent.from_facts:
                f = self.kb.get_fact(fact_id)
                if f is not None:
                    parent_facts.append({
                        "fact_id": f.fact_id,
                        "fact_type": f.fact_type.value,
                        "content": f.content,
                        "confidence": f.confidence,
                    })
            observations["parent_facts"] = parent_facts

            # 4. Facts related to the Intent's specialization
            relevant_fact_types = self._specialization_relevant_fact_types()
            relevant_facts = []
            for ft in relevant_fact_types:
                for f in self.kb.get_facts_by_type(ft):
                    relevant_facts.append({
                        "fact_id": f.fact_id,
                        "fact_type": f.fact_type.value,
                        "content": f.content,
                        "confidence": f.confidence,
                    })
            observations["relevant_facts"] = relevant_facts
        else:
            observations["current_intent"] = None

        # 5. Detect new facts since last cycle
        current_count = self.kb.get_fact_count()
        new_facts_count = current_count - self._last_fact_count
        observations["new_facts_since_last_cycle"] = new_facts_count
        self._last_fact_count = current_count

        # 6. Available tools for this specialization
        available_tools = [t.name for t in self.tools]
        observations["available_tools"] = available_tools

        logger.debug(
            f"Miner {self.agent_id} ({self.specialization.value}) observed: "
            f"{current_count} total facts, {observations.get('pending_intents', 0)} "
            f"pending intents"
        )

        return observations

    def _specialization_relevant_fact_types(self) -> list[FactType]:
        """Return fact types most relevant to this miner's specialization."""
        common = [FactType.FAILURE_BOUNDARY]
        spec_map: dict[MinerSpec, list[FactType]] = {
            MinerSpec.API_SEQUENCE: [FactType.API_ENDPOINT, FactType.CODE_STRUCTURE, FactType.BUSINESS_RULE],
            MinerSpec.DATAFLOW_TAINT: [FactType.DATAFLOW, FactType.API_ENDPOINT, FactType.CONFIGURATION],
            MinerSpec.BUSINESS_LOGIC: [FactType.BUSINESS_RULE, FactType.CODE_STRUCTURE, FactType.DATAFLOW],
            MinerSpec.ATTACK_SURFACE: [FactType.API_ENDPOINT, FactType.CONFIGURATION, FactType.SECURITY_CONTROL],
            MinerSpec.STATE_MACHINE: [FactType.CODE_STRUCTURE, FactType.BUSINESS_RULE, FactType.DATAFLOW],
            MinerSpec.AUTH_MODEL: [FactType.SECURITY_CONTROL, FactType.API_ENDPOINT, FactType.CONFIGURATION],
        }
        return spec_map.get(self.specialization, []) + common

    # ──────────────────────── OODA: Orient ────────────────────────

    async def orient(self, observations: dict[str, Any]) -> dict[str, Any]:
        """
        Orient phase: Evaluate current situation based on specialization.

        Assesses:
        - What has already been discovered (relevant to this specialization)
        - What remains unexplored or uncertain
        - Which directions are most promising (priority)
        - Any course corrections from Observer (via Hints)
        - Failure boundaries that constrain the search space
        """
        orientation: dict[str, Any] = {}

        # 1. Summarize what we know
        relevant_facts = observations.get("relevant_facts", [])
        parent_facts = observations.get("parent_facts", [])

        known_types: dict[str, int] = {}
        for f in relevant_facts:
            ftype = f["fact_type"]
            known_types[ftype] = known_types.get(ftype, 0) + 1

        orientation["known_fact_types"] = known_types
        orientation["parent_fact_summary"] = " | ".join(
            f["content"][:60] for f in parent_facts
        )

        # 2. Identify gaps — what types of facts are missing
        all_relevant = self._specialization_relevant_fact_types()
        missing_types = [
            ft.value for ft in all_relevant
            if ft.value not in known_types
        ]
        orientation["missing_fact_types"] = missing_types

        # 3. Check failure boundaries (what's already ruled out)
        if self.kb is not None:
            boundaries = self.kb.get_facts_by_type(FactType.FAILURE_BOUNDARY)
            orientation["ruled_out_directions"] = [
                b.content[:100] for b in boundaries
            ]
            # Also count hypotheses
            hypotheses = self.kb.get_facts_by_type(FactType.VULN_HYPOTHESIS)
            orientation["existing_hypotheses"] = len(hypotheses)

        # 4. Observer hints (course corrections)
        if self.kb is not None:
            hints = self.kb.get_all_hints()
            if hints:
                orientation["observer_hints"] = [
                    f"[{h.severity}] {h.pattern}: {h.applicability}"
                    for h in hints[:5]
                ]
            else:
                orientation["observer_hints"] = []

        # 5. Build assessment string for LLM prompt
        current_intent = observations.get("current_intent")
        if current_intent:
            assessment = (
                f"Working on intent '{current_intent['description'][:80]}' "
                f"(priority={current_intent['priority']:.2f}) "
                f"with specialization {self.specialization.value}. "
            )
        else:
            assessment = f"No current intent. Specialization: {self.specialization.value}. "

        if missing_types:
            assessment += f"Missing exploration in: {', '.join(missing_types)}. "

        ruled_out = orientation.get("ruled_out_directions", [])
        if ruled_out:
            assessment += f"Ruled-out directions: {'; '.join(ruled_out[:3])}. "

        orientation["assessment"] = assessment

        logger.debug(
            f"Miner {self.agent_id} oriented: "
            f"{len(known_types)} known types, "
            f"{len(missing_types)} missing types"
        )

        return orientation

    # ──────────────────────── OODA: Decide ────────────────────────

    async def decide(self, orientation: dict[str, Any]) -> dict[str, Any]:
        """
        Decide phase: Choose which analysis tool to invoke next.

        Decision logic:
        1. If there are missing fact types and we have a relevant tool, prefer that
        2. If there are unexplored hypotheses, choose tools to validate them
        3. Default to preferred tools for this specialization

        Uses LLM to make context-aware decisions when available.
        """
        decision: dict[str, Any] = {}

        # Collect available tool names
        tool_map = {t.name: t for t in self.tools}
        available = list(tool_map.keys())

        # Preferred tools for this specialization that are actually available
        preferred_available = [
            name for name in self._preferred_tools if name in available
        ]

        # Determine which tool to use based on orientation
        missing_types = orientation.get("missing_fact_types", [])
        assessment = orientation.get("assessment", "")

        # Use LLM for context-aware tool selection if available
        chosen_tool = None
        tool_params = {}

        if self.llm is not None:
            try:
                chosen_tool, tool_params = await self._llm_decide_tool(
                    assessment, preferred_available, available, orientation
                )
            except Exception as e:
                logger.warning(
                    f"Miner {self.agent_id}: LLM tool selection failed: {e}, "
                    f"fall back to preference order"
                )
                chosen_tool = None

        # Fallback: use preference order
        if chosen_tool is None:
            if preferred_available:
                # Cycle through preferred tools
                idx = self._cycle_count % len(preferred_available)
                chosen_tool = preferred_available[idx]
            elif available:
                chosen_tool = available[0]
            else:
                chosen_tool = "none"

        # Build tool parameters based on current intent
        if self._current_intent is not None:
            tool_params["intent_id"] = self._current_intent.intent_id
            tool_params["intent_description"] = self._current_intent.description
            tool_params["from_facts"] = self._current_intent.from_facts

        # Include relevant specialization context
        tool_params["specialization"] = self.specialization.value

        decision["action"] = chosen_tool
        decision["parameters"] = tool_params
        decision["preferred_tools"] = preferred_available
        decision["missing_types"] = missing_types

        logger.debug(
            f"Miner {self.agent_id} decided: tool={chosen_tool}, "
            f"params={list(tool_params.keys())}"
        )

        return decision

    async def _llm_decide_tool(
        self,
        assessment: str,
        preferred: list[str],
        available: list[str],
        orientation: dict[str, Any],
    ) -> tuple[str, dict]:
        """
        Use LLM to decide which tool to use.

        Returns (tool_name, parameters) tuple.
        """
        prompt = (
            f"You are a vulnerability mining agent specializing in {self.specialization.value}.\n"
            f"Current assessment: {assessment}\n\n"
            f"Available tools: {', '.join(available)}\n"
            f"Preferred tools: {', '.join(preferred)}\n\n"
            f"Choose ONE tool to use next and provide parameters as JSON.\n"
            f"Respond in this exact format:\n"
            f"TOOL: <tool_name>\n"
            f"PARAMS: <json_object>\n"
            f"REASON: <brief reasoning>"
        )

        result = await self.llm_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )

        content = result.content.strip()
        tool_name = preferred[0] if preferred else (available[0] if available else "none")
        params = {}

        # Parse TOOL: and PARAMS: lines
        for line in content.split("\n"):
            line = line.strip()
            if line.upper().startswith("TOOL:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate in available:
                    tool_name = candidate
            elif line.upper().startswith("PARAMS:"):
                try:
                    params = json.loads(line.split(":", 1)[1].strip())
                except (json.JSONDecodeError, IndexError):
                    pass

        return tool_name, params

    # ──────────────────────── OODA: Act ────────────────────────

    async def act(self, decision: dict[str, Any]) -> OODAResult:
        """
        Act phase: Execute the chosen analysis tool and produce outputs.

        Steps:
        1. Resolve and execute the selected tool
        2. Interpret the results
        3. Produce Facts (vulnerability hypotheses, data flows, etc.)
        4. Possibly produce new Intents for deeper exploration
        5. Possibly produce FailureBoundaries for ruled-out directions
        6. Submit all outputs to VulnKB
        """
        action = decision.get("action", "none")
        parameters = decision.get("parameters", {})

        facts_produced: list[Fact] = []
        intents_produced: list[Intent] = []
        boundaries_produced: list[FailureBoundary] = []
        observations_list: list[str] = []

        # Resolve tool
        tool_map = {t.name: t for t in self.tools}
        tool = tool_map.get(action)

        tool_result = None
        if tool is not None:
            try:
                tool_result = await tool.call(**parameters)
                observations_list.append(f"Tool '{action}' executed successfully")
            except Exception as e:
                logger.error(f"Miner {self.agent_id}: tool '{action}' failed: {e}")
                observations_list.append(f"Tool '{action}' failed: {e}")
                tool_result = None
        else:
            observations_list.append(f"No tool '{action}' available for execution")

        # Use LLM to interpret results and produce findings
        if self.llm is not None and (tool_result is not None or observations_list):
            try:
                produced = await self._llm_interpret_and_produce(
                    action, tool_result, observations_list, decision
                )
                facts_produced = produced.get("facts", [])
                intents_produced = produced.get("intents", [])
                boundaries_produced = produced.get("boundaries", [])
            except Exception as e:
                logger.error(
                    f"Miner {self.agent_id}: LLM interpretation failed: {e}"
                )

        # Determine if we should continue
        should_continue = True
        if self._current_intent is not None and not facts_produced and not intents_produced:
            # No progress this cycle — decrement tolerance
            if self._consecutive_failures >= self.config.max_consecutive_failures - 1:
                should_continue = False

        return OODAResult(
            observations=observations_list,
            orientation=decision.get("missing_types", []),
            decision=action,
            action_output=tool_result,
            facts_produced=facts_produced,
            intents_produced=intents_produced,
            boundaries_produced=boundaries_produced,
            should_continue=should_continue,
            cycle_number=self._cycle_count,
        )

    async def _llm_interpret_and_produce(
        self,
        action: str,
        tool_result: Any,
        observations: list[str],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Use LLM to interpret tool results and produce structured findings.

        Returns dict with 'facts', 'intents', 'boundaries' lists.
        """
        if self.kb is None:
            return {"facts": [], "intents": [], "boundaries": []}

        # Build context for the LLM
        kb_gist = self.kb.build_context(level="gist")
        current_intent_desc = ""
        if self._current_intent:
            current_intent_desc = self._current_intent.description

        result_str = str(tool_result) if tool_result else "No tool result available"
        obs_str = "\n".join(f"- {o}" for o in observations)

        prompt = (
            f"You are a {self.specialization.value} mining agent analyzing code for vulnerabilities.\n\n"
            f"## Current Knowledge Base Summary\n{kb_gist}\n\n"
            f"## Current Intent\n{current_intent_desc}\n\n"
            f"## Tool Used\n{action}\n\n"
            f"## Tool Result\n{result_str}\n\n"
            f"## Observations\n{obs_str}\n\n"
            f"Based on the above, produce findings in the following format. "
            f"If you find nothing, produce empty lists.\n\n"
            f"FACTS (each line: TYPE|content|evidence|confidence):\n"
            f"INTENTS (each line: description|priority|specialization):\n"
            f"BOUNDARIES (each line: vuln_type|ruled_out|remaining_risk|confidence):\n"
        )

        result = await self.llm_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )

        return self._parse_llm_output(result.content)

    def _parse_llm_output(self, content: str) -> dict[str, Any]:
        """
        Parse structured LLM output into Fact, Intent, and FailureBoundary objects.

        Expected format:
        FACTS:
        TYPE|content|evidence|confidence
        INTENTS:
        description|priority|specialization
        BOUNDARIES:
        vuln_type|ruled_out|remaining_risk|confidence
        """
        facts: list[Fact] = []
        intents: list[Intent] = []
        boundaries: list[FailureBoundary] = []

        current_section = None

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Section headers
            if line.upper().startswith("FACTS"):
                current_section = "facts"
                continue
            elif line.upper().startswith("INTENTS"):
                current_section = "intents"
                continue
            elif line.upper().startswith("BOUNDARIES"):
                current_section = "boundaries"
                continue

            # Parse entries
            try:
                if current_section == "facts":
                    parts = line.split("|", 3)
                    if len(parts) >= 3:
                        fact_type_str = parts[0].strip()
                        try:
                            fact_type = FactType(fact_type_str)
                        except ValueError:
                            fact_type = FactType.VULN_HYPOTHESIS

                        fact_content = parts[1].strip()
                        evidence = parts[2].strip() if len(parts) > 2 else fact_content
                        confidence = float(parts[3]) if len(parts) > 3 else 0.7

                        fact = self.submit_fact(
                            fact_type=fact_type,
                            content=fact_content,
                            evidence=evidence,
                            confidence=confidence,
                        )
                        if fact is not None:
                            facts.append(fact)

                elif current_section == "intents":
                    parts = line.split("|", 2)
                    if len(parts) >= 1:
                        desc = parts[0].strip()
                        priority = float(parts[1]) if len(parts) > 1 else 0.5
                        spec_str = parts[2].strip() if len(parts) > 2 else None
                        try:
                            spec = MinerSpec(spec_str) if spec_str else None
                        except ValueError:
                            spec = None

                        intent = self.submit_intent(
                            description=desc,
                            priority=priority,
                            specialization=spec,
                        )
                        intents.append(intent)

                elif current_section == "boundaries":
                    parts = line.split("|", 4)
                    if len(parts) >= 3:
                        vuln_type = parts[0].strip()
                        ruled_out = parts[1].strip()
                        remaining_risk = parts[2].strip()
                        confidence = float(parts[3]) if len(parts) > 3 else 0.8
                        evidence = parts[4].strip() if len(parts) > 4 else ruled_out

                        boundary = self.submit_failure_boundary(
                            vuln_type=vuln_type,
                            ruled_out=ruled_out,
                            remaining_risk=remaining_risk,
                            evidence=evidence,
                            confidence=confidence,
                        )
                        boundaries.append(boundary)

            except (ValueError, IndexError) as e:
                logger.debug(f"Miner {self.agent_id}: skipped malformed line: {line} ({e})")
                continue

        return {"facts": facts, "intents": intents, "boundaries": boundaries}

    # ──────────────────────── Lifecycle Override ────────────────────────

    async def run(self) -> None:
        """
        Miner main loop — claims intents matching its specialization.

        Overrides AgentBase.run() to filter intents by MinerSpec when claiming.
        """
        if self.kb is None:
            raise RuntimeError(f"MinerAgent {self.agent_id} has no VulnKB reference")

        self._running = True
        self._initialize_prompt_stable()
        self._last_fact_count = self.kb.get_fact_count()

        logger.info(
            f"MinerAgent {self.agent_id} ({self.specialization.value}) starting"
        )

        try:
            while self._running:
                # Claim an intent matching our specialization
                intent = self.claim_task(specialization=self.specialization)
                if intent is None:
                    # Also try claiming generic intents (no specialization set)
                    intent = self.claim_task(specialization=None)

                if intent is None:
                    logger.debug(
                        f"Miner {self.agent_id}: no matching intents, waiting..."
                    )
                    await asyncio.sleep(self.config.idle_poll_interval_seconds)
                    continue

                self._current_intent = intent
                self._cycle_count = 0
                self._consecutive_failures = 0
                logger.info(
                    f"Miner {self.agent_id} claimed intent {intent.intent_id}: "
                    f"{intent.description}"
                )

                # Run OODA loop
                ooda_result = await self._run_ooda_loop(intent)

                # Finalize intent
                if ooda_result is not None and ooda_result.should_continue:
                    if self.kb is not None:
                        self.kb.complete_intent(intent.intent_id)
                    logger.info(
                        f"Miner {self.agent_id} completed intent {intent.intent_id}"
                    )
                else:
                    if self.kb is not None:
                        self.kb.fail_intent(intent.intent_id)
                    logger.warning(
                        f"Miner {self.agent_id} failed intent {intent.intent_id}"
                    )

                self._current_intent = None

        except asyncio.CancelledError:
            logger.info(f"Miner {self.agent_id} cancelled, shutting down")
        except Exception as e:
            logger.error(f"Miner {self.agent_id} fatal error: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info(f"Miner {self.agent_id} main loop exited")


