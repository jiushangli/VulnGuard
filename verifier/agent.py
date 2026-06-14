"""
VulnGuard Verifier Agent — Independent vulnerability verification.

VerifierAgent inherits AgentBase and operates independently from Miners.
It does NOT claim Intents. Instead, it:

1. Reads VULN_HYPOTHESIS facts from VulnKB
2. Performs three-phase verification:
   a. Evidence chain validation (evidence_chain_validate)
   b. PoC construction (poc_generate)
   c. Boundary testing (boundary_test) — positive, negative, and edge cases
3. Writes results back to VulnKB:
   - Confirmed → Fact type changed to VULNERABILITY
   - Ruled out → New FAILURE_BOUNDARY Fact
   - Needs more → New Intent for Miners to continue

Key design: Verifier uses an independent LLM provider (configured via
config.verifier_provider) to ensure independence from Miner's analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..vulnkb.models import (
    AuditPhase,
    Fact,
    FactType,
    FailureBoundary,
    Intent,
    MinerSpec,
    VerificationResult,
    VulnKB,
    make_fact,
    make_intent,
)

from ..agent_base import AgentBase, AgentConfig, AgentType, OODAResult
from ..tools.descriptor import AuditTool
from ..utils.llm import AgentRole, LLMGateway, LLMProvider
from ..utils.prompt import PromptLayer, PromptManager

logger = logging.getLogger(__name__)


# ──────────────────────── Verification Phases ────────────────────────


class VerificationPhase(Enum):
    """Phases of the independent verification process."""
    EVIDENCE_CHAIN = "evidence_chain"    # Validate evidence chain completeness
    POC_GENERATE = "poc_generate"         # Construct minimal PoC
    BOUNDARY_TEST = "boundary_test"       # Positive, negative, and edge case testing


# ──────────────────────── Verification Result Detail ────────────────────────


@dataclass
class VerificationDetail:
    """
    Detailed result of a single verification phase.
    """
    phase: VerificationPhase
    passed: bool
    confidence: float
    findings: str = ""
    evidence: str = ""
    """Evidence supporting the phase result."""
    metadata: dict = field(default_factory=dict)


# ──────────────────────── VerifierAgent ────────────────────────


class VerifierAgent(AgentBase):
    """
    Independent verification agent.

    The Verifier reads VULN_HYPOTHESIS facts from VulnKB and performs
    a three-phase verification process:

    1. evidence_chain_validate: Check that the hypothesis's supporting
       evidence is complete, logically consistent, and sufficient to
       establish the claimed vulnerability.

    2. poc_generate: Construct a minimal proof-of-concept that demonstrates
       the vulnerability. This PoC must be the smallest possible
       demonstration.

    3. boundary_test: Test the vulnerability boundary with:
       - Positive test: the vulnerability IS exploitable (confirms)
       - Negative test: variations that DON'T trigger it (defines boundary)
       - Edge cases: boundary conditions that test the exact limits

    Results are written back to VulnKB:
    - confirmed: Update Fact type to VULNERABILITY
    - ruled_out: New FAILURE_BOUNDARY Fact with precise description
    - needs_more: New Intent for further Miner exploration

    The Verifier uses an independent LLM provider to ensure its analysis
    is not biased by the Miner's perspective.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        kb: VulnKB | None = None,
        llm: LLMGateway | None = None,
        tools: list[AuditTool] | None = None,
        config: AgentConfig | None = None,
        verifier_provider: LLMProvider | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.VERIFIER,
            specialization=None,
            kb=kb,
            llm=llm,
            tools=tools or [],
            config=config or AgentConfig(),
        )
        # Independent LLM provider for verification
        self._verifier_provider: LLMProvider | None = verifier_provider

        # Verification state tracking
        self._verified_hypothesis_ids: set[str] = set()
        """Set of hypothesis fact IDs that have been verified or queued."""

        self._pending_verifications: list[dict[str, Any]] = []
        """Queue of hypotheses awaiting verification."""

        # Verification configuration
        self.min_evidence_confidence: float = 0.5
        """Minimum evidence confidence to attempt verification."""
        self.verification_batch_size: int = 3
        """Number of hypotheses to verify in one cycle."""

    # ──────────────────────── Prompt ────────────────────────

    def _get_stable_content(self) -> tuple[str, str, str, str]:
        """Generate stable prompt content for Verifier role."""
        role_definition = (
            f"You are a VulnGuard Verifier agent (id: '{self.agent_id}'). "
            f"Your role is to independently verify vulnerability hypotheses.\n\n"
            f"You are INDEPENDENT from Miner agents. Your analysis must not be "
            f"biased by their perspective. You must:\n"
            f"1. Validate evidence chains objectively\n"
            f"2. Construct minimal PoCs that demonstrate the vulnerability\n"
            f"3. Test boundary conditions precisely\n\n"
            f"Verification outcomes:\n"
            f"- confirmed: The vulnerability is real and exploitable\n"
            f"- ruled_out: The vulnerability hypothesis is incorrect; describe the "
            f"  precise boundary of what IS ruled out and what risk remains\n"
            f"- needs_more: Insufficient evidence to confirm or rule out; specify "
            f"  what additional information is needed"
        )

        audit_framework = (
            "You read VULN_HYPOTHESIS facts from VulnKB and verify them through "
            "a three-phase process:\n"
            "1. EVIDENCE CHAIN VALIDATION: Verify the evidence is complete, "
            "   logically consistent, and sufficient for the claim.\n"
            "2. PoC CONSTRUCTION: Build the smallest possible demonstration of "
            "   the vulnerability. Focus on minimal, reproducible conditions.\n"
            "3. BOUNDARY TESTING: Test positive (exploitable), negative (not "
            "   exploitable), and edge cases to precisely define the vulnerability "
            "   boundary.\n\n"
            "Critical principle: Honest confidence scores are essential. "
            "Overconfidence is dangerous. When uncertain, report 'needs_more' "
            "rather than guessing."
        )

        security_constraints = (
            "SECURITY CONSTRAINTS:\n"
            "1. NEVER confirm a vulnerability without constructing a valid PoC\n"
            "2. NEVER rule out a vulnerability without precise boundary description\n"
            "3. All writes to VulnKB must go through verified admission\n"
            "4. Maintain independence — do NOT read Miner analysis as authoritative\n"
            "5. Use the separate verifier LLM provider for independence\n"
            "6. Report partial confirmations honestly (affects risk assessment)"
        )

        output_format = (
            "OUTPUT FORMAT:\n"
            "Each verification phase produces structured output:\n\n"
            "EVIDENCE CHAIN:\n"
            "  PHASE: evidence_chain\n"
            "  PASSED: true/false\n"
            "  CONFIDENCE: 0.0-1.0\n"
            "  GAPS: <description of any evidence gaps>\n"
            "  FINDINGS: <what the evidence establishes>\n\n"
            "POC CONSTRUCTION:\n"
            "  PHASE: poc_generate\n"
            "  PASSED: true/false\n"
            "  CONFIDENCE: 0.0-1.0\n"
            "  POC: <minimal proof of concept>\n"
            "  PREREQUISITES: <conditions needed>\n\n"
            "BOUNDARY TEST:\n"
            "  PHASE: boundary_test\n"
            "  POSITIVE: <exploitable conditions>\n"
            "  NEGATIVE: <safe conditions that don't trigger>\n"
            "  EDGE: <boundary conditions>\n\n"
            "VERIFICATION VERDICT:\n"
            "  RESULT: confirmed/partial/ruled_out/needs_more\n"
            "  CONFIDENCE: 0.0-1.0\n"
            "  SUMMARY: <concise description>\n"
        )

        return role_definition, audit_framework, security_constraints, output_format

    # ──────────────────────── Main Loop ────────────────────────

    async def run(self) -> None:
        """
        Verifier main loop — NOT the standard claim-Intent OODA loop.

        The Verifier reads VULN_HYPOTHESIS facts from VulnKB and verifies
        them through the three-phase process. It does NOT claim Intents.
        """
        if self.kb is None:
            raise RuntimeError(f"VerifierAgent {self.agent_id} has no VulnKB reference")

        self._running = True
        self._initialize_prompt_stable()

        logger.info(f"VerifierAgent {self.agent_id} starting")

        try:
            while self._running:
                # Collect hypotheses to verify
                hypotheses = self._collect_hypotheses()

                if not hypotheses:
                    logger.debug(
                        f"Verifier {self.agent_id}: no hypotheses to verify, waiting..."
                    )
                    await asyncio.sleep(self.config.idle_poll_interval_seconds)
                    continue

                # Process hypotheses in batches
                for hypothesis in hypotheses[:self.verification_batch_size]:
                    if not self._running:
                        break
                    await self._verify_hypothesis(hypothesis)

                # Brief pause between batches
                await asyncio.sleep(self.config.idle_poll_interval_seconds)

        except asyncio.CancelledError:
            logger.info(f"Verifier {self.agent_id} cancelled, shutting down")
        except Exception as e:
            logger.error(f"Verifier {self.agent_id} fatal error: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info(f"Verifier {self.agent_id} main loop exited")

    def _collect_hypotheses(self) -> list[Fact]:
        """
        Collect VULN_HYPOTHESIS facts that haven't been verified yet.

        Returns hypotheses sorted by confidence (highest first).
        """
        if self.kb is None:
            return []

        all_hypotheses = self.kb.get_facts_by_type(FactType.VULN_HYPOTHESIS)

        # Filter out already-verified hypotheses
        unverified = [
            h for h in all_hypotheses
            if h.fact_id not in self._verified_hypothesis_ids
            and h.confidence >= self.min_evidence_confidence
        ]

        # Sort by confidence (highest first)
        unverified.sort(key=lambda h: -h.confidence)

        return unverified

    # ──────────────────────── Three-Phase Verification ────────────────────────

    async def _verify_hypothesis(self, hypothesis: Fact) -> None:
        """
        Execute three-phase verification for a single hypothesis:

        1. Evidence chain validation
        2. PoC construction
        3. Boundary testing

        Writes the result back to VulnKB.
        """
        self._verified_hypothesis_ids.add(hypothesis.fact_id)

        logger.info(
            f"Verifier {self.agent_id}: verifying hypothesis "
            f"{hypothesis.fact_id} (conf={hypothesis.confidence:.2f}): "
            f"{hypothesis.content[:80]}..."
        )

        # Set current phase and update context
        self._current_phase = AuditPhase.VERIFICATION

        # Phase 1: Evidence chain validation
        evidence_result = await self.evidence_chain_validate(hypothesis)

        # If evidence chain fails badly, might still proceed to rule out
        if not evidence_result.passed and evidence_result.confidence < 0.3:
            # Insufficient evidence — needs more
            await self._write_verification_result(
                hypothesis, VerificationResult.NEEDS_MORE_INFO,
                [evidence_result],
                summary=f"Evidence chain too weak (conf={evidence_result.confidence:.2f}): "
                        f"{evidence_result.findings[:100]}"
            )
            return

        # Phase 2: PoC construction
        poc_result = await self.poc_generate(hypothesis, evidence_result)

        # If PoC cannot be constructed, needs more information
        if not poc_result.passed and evidence_result.passed:
            await self._write_verification_result(
                hypothesis, VerificationResult.NEEDS_MORE_INFO,
                [evidence_result, poc_result],
                summary=f"Cannot construct PoC (evidence chain passed, PoC failed): "
                        f"{poc_result.findings[:100]}"
            )
            return

        # Phase 3: Boundary testing
        boundary_result = await self.boundary_test(
            hypothesis, evidence_result, poc_result
        )

        # Aggregate the three phases to determine final verdict
        verdict = self._aggregate_verification(
            evidence_result, poc_result, boundary_result
        )

        # Write result to VulnKB
        await self._write_verification_result(
            hypothesis, verdict,
            [evidence_result, poc_result, boundary_result],
            summary=f"Verification complete: {verdict.value}"
        )

    async def evidence_chain_validate(self, hypothesis: Fact) -> VerificationDetail:
        """
        Phase 1: Validate the evidence chain for a vulnerability hypothesis.

        Checks:
        - Is the evidence complete (no critical gaps)?
        - Is it logically consistent (no contradictions)?
        - Is the source-to-sink chain intact?
        - Are there sufficient code references?
        """
        if self.llm is None and self._verifier_provider is None:
            return VerificationDetail(
                phase=VerificationPhase.EVIDENCE_CHAIN,
                passed=False,
                confidence=0.0,
                findings="No LLM provider available for verification",
            )

        # Collect related evidence from VulnKB
        related_facts = self._collect_related_facts(hypothesis)
        evidence_text = "\n".join(
            f"[{f.fact_type.value}] {f.content} (conf={f.confidence:.2f})"
            for f in related_facts
        )

        prompt = (
            f"You are an independent vulnerability verifier. Evaluate the evidence "
            f"chain for this hypothesis:\n\n"
            f"## Hypothesis\n"
            f"ID: {hypothesis.fact_id}\n"
            f"Content: {hypothesis.content}\n"
            f"Confidence: {hypothesis.confidence:.2f}\n"
            f"Evidence: {hypothesis.evidence}\n\n"
            f"## Related Facts\n{evidence_text}\n\n"
            f"Evaluate the evidence chain:\n"
            f"1. Is the evidence complete (no critical gaps)?\n"
            f"2. Is it logically consistent?\n"
            f"3. Does the source-to-sink chain hold?\n"
            f"4. Are there sufficient code references?\n\n"
            f"Respond in this format:\n"
            f"PASSED: true/false\n"
            f"CONFIDENCE: 0.0-1.0\n"
            f"GAPS: <description of evidence gaps, or 'none'>\n"
            f"FINDINGS: <what the evidence establishes>"
        )

        try:
            result = await self._verifier_complete(prompt, max_tokens=1024, temperature=0.2)
            return self._parse_evidence_chain_result(result, hypothesis)
        except Exception as e:
            logger.error(f"Verifier {self.agent_id}: evidence chain validation failed: {e}")
            return VerificationDetail(
                phase=VerificationPhase.EVIDENCE_CHAIN,
                passed=False,
                confidence=0.0,
                findings=f"Evidence chain validation error: {e}",
            )

    async def poc_generate(
        self, hypothesis: Fact, evidence_result: VerificationDetail
    ) -> VerificationDetail:
        """
        Phase 2: Construct a minimal proof-of-concept.

        The PoC must be:
        - Minimal: smallest possible demonstration
        - Reproducible: can be independently verified
        - Precise: targets exactly the claimed vulnerability
        """
        if not evidence_result.passed and evidence_result.confidence < 0.3:
            return VerificationDetail(
                phase=VerificationPhase.POC_GENERATE,
                passed=False,
                confidence=0.0,
                findings="Skipped: evidence chain too weak to construct PoC",
            )

        prompt = (
            f"You are constructing a minimal proof-of-concept for this vulnerability:\n\n"
            f"## Hypothesis\n{hypothesis.content}\n\n"
            f"## Evidence\n{hypothesis.evidence}\n\n"
            f"## Evidence Chain Assessment\n"
            f"Passed: {evidence_result.passed}\n"
            f"Confidence: {evidence_result.confidence:.2f}\n"
            f"Findings: {evidence_result.findings}\n\n"
            f"Construct the MINIMAL PoC that demonstrates this vulnerability.\n"
            f"Focus on: smallest input, simplest code path, fewest prerequisites.\n\n"
            f"Respond in this format:\n"
            f"PASSED: true/false (can you construct a valid PoC?)\n"
            f"CONFIDENCE: 0.0-1.0\n"
            f"POC: <minimal proof of concept>\n"
            f"PREREQUISITES: <conditions needed for the PoC to work>"
        )

        try:
            result = await self._verifier_complete(prompt, max_tokens=2048, temperature=0.2)
            return self._parse_poc_result(result, hypothesis)
        except Exception as e:
            logger.error(f"Verifier {self.agent_id}: PoC generation failed: {e}")
            return VerificationDetail(
                phase=VerificationPhase.POC_GENERATE,
                passed=False,
                confidence=0.0,
                findings=f"PoC generation error: {e}",
            )

    async def boundary_test(
        self,
        hypothesis: Fact,
        evidence_result: VerificationDetail,
        poc_result: VerificationDetail,
    ) -> VerificationDetail:
        """
        Phase 3: Test vulnerability boundaries.

        Three categories of tests:
        - Positive: The vulnerability IS exploitable (confirms the hypothesis)
        - Negative: Variations that DON'T trigger the vulnerability (defines scope)
        - Edge: Boundary conditions that test exact limits
        """
        prompt = (
            f"You are testing the boundaries of a vulnerability:\n\n"
            f"## Hypothesis\n{hypothesis.content}\n\n"
            f"## PoC\n{poc_result.evidence or poc_result.findings}\n\n"
            f"## Evidence Assessment\n"
            f"Confidence: {evidence_result.confidence:.2f}\n\n"
            f"Test the vulnerability boundary:\n"
            f"1. POSITIVE test: conditions where vulnerability IS exploitable\n"
            f"2. NEGATIVE test: similar conditions that DON'T trigger it\n"
            f"3. EDGE cases: exact boundary conditions\n\n"
            f"Respond in this format:\n"
            f"POSITIVE: <conditions where vulnerability triggers>\n"
            f"NEGATIVE: <conditions where it doesn't trigger>\n"
            f"EDGE: <boundary conditions>\n"
            f"BOUNDARY_DESCRIPTION: <precise description of vulnerability boundary>"
        )

        try:
            result = await self._verifier_complete(prompt, max_tokens=2048, temperature=0.2)
            return self._parse_boundary_result(result, hypothesis)
        except Exception as e:
            logger.error(f"Verifier {self.agent_id}: boundary testing failed: {e}")
            return VerificationDetail(
                phase=VerificationPhase.BOUNDARY_TEST,
                passed=False,
                confidence=0.0,
                findings=f"Boundary test error: {e}",
            )

    # ──────────────────────── Verdict Aggregation ────────────────────────

    def _aggregate_verification(
        self,
        evidence: VerificationDetail,
        poc: VerificationDetail,
        boundary: VerificationDetail,
    ) -> VerificationResult:
        """
        Aggregate the three verification phases into a final verdict.

        Logic:
        - All three passed with high confidence → CONFIRMED
        - Evidence + PoC passed but partial boundary → PARTIALLY_CONFIRMED
        - Evidence and PoC clearly disproven → RULED_OUT
        - Insufficient evidence to determine → NEEDS_MORE_INFO
        """
        # Weight the phases
        avg_confidence = (
            evidence.confidence * 0.3 + poc.confidence * 0.4 + boundary.confidence * 0.3
        )

        if evidence.passed and poc.passed:
            if avg_confidence >= 0.7:
                return VerificationResult.CONFIRMED
            elif avg_confidence >= 0.4:
                return VerificationResult.PARTIALLY_CONFIRMED
            else:
                return VerificationResult.NEEDS_MORE_INFO

        if not evidence.passed and not poc.passed:
            if evidence.confidence < 0.2 and poc.confidence < 0.2:
                return VerificationResult.RULED_OUT
            else:
                return VerificationResult.NEEDS_MORE_INFO

        # Mixed results
        if evidence.passed and not poc.passed:
            return VerificationResult.NEEDS_MORE_INFO

        return VerificationResult.NEEDS_MORE_INFO

    # ──────────────────────── Write Results ────────────────────────

    async def _write_verification_result(
        self,
        hypothesis: Fact,
        verdict: VerificationResult,
        details: list[VerificationDetail],
        summary: str,
    ) -> None:
        """
        Write the verification result back to VulnKB.

        - confirmed → Add VULNERABILITY fact + update hypothesis
        - ruled_out → Add FAILURE_BOUNDARY fact
        - needs_more → Add new Intent for further exploration
        - partially → Mark hypothesis as partially confirmed
        """
        if self.kb is None:
            return

        # Aggregate evidence from all phases
        full_evidence = "\n\n".join(
            f"[{d.phase.value}] passed={d.passed}, conf={d.confidence:.2f}\n{d.findings}"
            for d in details
        )

        if verdict == VerificationResult.CONFIRMED:
            # Add confirmed vulnerability fact
            fact = self.submit_fact(
                fact_type=FactType.VULNERABILITY,
                content=(
                    f"CONFIRMED: {hypothesis.content} "
                    f"(verified with confidence {details[-1].confidence:.2f})"
                ),
                evidence=full_evidence,
                confidence=max(d.confidence for d in details),
                parent_intents=hypothesis.parent_intents,
                metadata={
                    "original_hypothesis_id": hypothesis.fact_id,
                    "verification_verdict": verdict.value,
                    "verifier_id": self.agent_id,
                },
            )
            logger.info(
                f"Verifier {self.agent_id}: CONFIRMED vulnerability "
                f"from hypothesis {hypothesis.fact_id}"
            )

        elif verdict == VerificationResult.RULED_OUT:
            # Add failure boundary
            boundary_content = (
                f"Ruled out: {hypothesis.content}. "
                f"Boundary: {details[-1].findings[:200]}"
            )
            remaining_risk = "Investigation needed for related attack vectors"
            for d in details:
                if d.phase == VerificationPhase.BOUNDARY_TEST:
                    remaining_risk = d.metadata.get("boundary_description", remaining_risk)
                    break

            self.submit_failure_boundary(
                vuln_type=hypothesis.fact_type.value,
                ruled_out=hypothesis.content,
                remaining_risk=remaining_risk,
                evidence=full_evidence,
                confidence=max(0.8, 1.0 - hypothesis.confidence),
            )
            logger.info(
                f"Verifier {self.agent_id}: RULED OUT hypothesis "
                f"{hypothesis.fact_id}: {hypothesis.content[:60]}"
            )

        elif verdict == VerificationResult.NEEDS_MORE_INFO:
            # Add new Intent for Miners to continue exploring
            gaps = []
            for d in details:
                if d.phase == VerificationPhase.EVIDENCE_CHAIN:
                    gaps.append(d.findings[:100])

            gap_description = "; ".join(gaps) if gaps else "Additional evidence needed"

            intent = self.submit_intent(
                description=(
                    f"Further investigation needed for: {hypothesis.content[:80]}. "
                    f"Gaps: {gap_description}"
                ),
                priority=0.7,
                specialization=None,  # Let any Miner pick it up
            )
            logger.info(
                f"Verifier {self.agent_id}: NEEDS MORE INFO for hypothesis "
                f"{hypothesis.fact_id}, created intent {intent.intent_id if intent else 'N/A'}"
            )

        elif verdict == VerificationResult.PARTIALLY_CONFIRMED:
            # Add partial confirmation as VULN_HYPOTHESIS with updated confidence
            partial_conf = sum(d.confidence for d in details) / len(details)
            fact = self.submit_fact(
                fact_type=FactType.VULN_HYPOTHESIS,
                content=(
                    f"PARTIALLY CONFIRMED: {hypothesis.content} "
                    f"(partial verification, conf={partial_conf:.2f})"
                ),
                evidence=full_evidence,
                confidence=partial_conf,
                parent_intents=hypothesis.parent_intents,
                metadata={
                    "original_hypothesis_id": hypothesis.fact_id,
                    "verification_verdict": verdict.value,
                    "verifier_id": self.agent_id,
                },
            )
            logger.info(
                f"Verifier {self.agent_id}: PARTIALLY CONFIRMED hypothesis "
                f"{hypothesis.fact_id} (conf={partial_conf:.2f})"
            )

    # ──────────────────────── OODA Methods ────────────────────────

    async def observe(self) -> dict[str, Any]:
        """
        Observe phase for Verifier: read VULN_HYPOTHESIS facts from VulnKB.
        """
        observations: dict[str, Any] = {}

        if self.kb is None:
            observations["error"] = "No VulnKB reference"
            return observations

        # Collect unverified hypotheses
        hypotheses = self._collect_hypotheses()
        observations["unverified_hypotheses"] = [
            {
                "fact_id": h.fact_id,
                "content": h.content,
                "confidence": h.confidence,
                "evidence": h.evidence[:200] if h.evidence else "",
                "parent_intents": h.parent_intents,
            }
            for h in hypotheses
        ]
        observations["total_unverified"] = len(hypotheses)

        # Related context
        all_facts = self.kb.get_all_facts()
        observations["total_facts"] = len(all_facts)

        return observations

    async def orient(self, observations: dict[str, Any]) -> dict[str, Any]:
        """
        Orient phase for Verifier: prioritize hypotheses for verification.
        """
        orientation: dict[str, Any] = {}
        hypotheses = observations.get("unverified_hypotheses", [])

        # Prioritize: higher confidence first
        orientation["prioritized_hypotheses"] = sorted(
            hypotheses, key=lambda h: -h.get("confidence", 0)
        )
        orientation["assessment"] = (
            f"{len(hypotheses)} hypotheses awaiting verification"
        )

        return orientation

    async def decide(self, orientation: dict[str, Any]) -> dict[str, Any]:
        """
        Decide phase for Verifier: select next hypothesis to verify.
        """
        hypotheses = orientation.get("prioritized_hypotheses", [])

        if hypotheses:
            target = hypotheses[0]
            decision = {
                "action": "verify",
                "target_hypothesis": target,
                "target_id": target["fact_id"],
            }
        else:
            decision = {"action": "wait", "target_hypothesis": None}

        return decision

    async def act(self, decision: dict[str, Any]) -> OODAResult:
        """
        Act phase for Verifier: execute verification on selected hypothesis.
        """
        action = decision.get("action", "wait")

        if action != "verify" or self.kb is None:
            return OODAResult(
                observations=["No hypothesis to verify, waiting"],
                orientation="",
                decision=action,
                action_output=None,
                should_continue=True,
                cycle_number=self._cycle_count,
            )

        target_id = decision.get("target_id", "")
        hypothesis = self.kb.get_fact(target_id)

        if hypothesis is None:
            return OODAResult(
                observations=[f"Hypothesis {target_id} not found"],
                orientation="",
                decision=action,
                action_output=None,
                should_continue=True,
                cycle_number=self._cycle_count,
            )

        # Execute three-phase verification
        await self._verify_hypothesis(hypothesis)

        return OODAResult(
            observations=[f"Verified hypothesis {target_id}"],
            orientation="",
            decision=action,
            action_output=target_id,
            facts_produced=[],  # Facts are produced inside _verify_hypothesis
            intents_produced=[],
            boundaries_produced=[],
            should_continue=True,
            cycle_number=self._cycle_count,
        )

    # ──────────────────────── Helper Methods ────────────────────────

    def _collect_related_facts(self, hypothesis: Fact) -> list[Fact]:
        """Collect facts related to a hypothesis (parent intents + same source)."""
        if self.kb is None:
            return []

        related = []
        # Facts from parent intents
        for intent_id in hypothesis.parent_intents:
            # Get intents that might reference this fact
            pass  # VulnKB doesn't have a direct intent→facts lookup

        # Get facts from the same source for additional context
        all_facts = self.kb.get_all_facts()
        for f in all_facts:
            if f.fact_id == hypothesis.fact_id:
                continue
            # Include facts that share parent intents
            if set(f.parent_intents) & set(hypothesis.parent_intents):
                related.append(f)
            # Include other facts from the same source
            elif f.source == hypothesis.source and f.fact_type != FactType.FAILURE_BOUNDARY:
                related.append(f)

        # Limit to prevent token overflow
        return related[:20]

    async def _verifier_complete(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2
    ) -> str:
        """
        Complete using the verifier's independent LLM provider.

        Falls back to the main LLM gateway if no dedicated verifier provider.
        """
        if self._verifier_provider is not None:
            messages = [{"role": "user", "content": prompt}]
            # Use the prompt manager's stable layer as system message
            system_content = self.prompt_manager.layer_content(PromptLayer.STABLE)
            if system_content:
                messages.insert(0, {"role": "system", "content": system_content})

            result = await self._verifier_provider.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result.content

        # Fall back to main LLM gateway
        result = await self.llm_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result.content

    # ──────────────────────── Result Parsing ────────────────────────

    def _parse_evidence_chain_result(
        self, content: str, hypothesis: Fact
    ) -> VerificationDetail:
        """Parse LLM evidence chain validation output."""
        passed = False
        confidence = 0.0
        gaps = ""
        findings = ""

        for line in content.split("\n"):
            line = line.strip()
            if line.upper().startswith("PASSED:"):
                passed = line.split(":", 1)[1].strip().lower() == "true"
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.upper().startswith("GAPS:"):
                gaps = line.split(":", 1)[1].strip()
            elif line.upper().startswith("FINDINGS:"):
                findings = line.split(":", 1)[1].strip()

        return VerificationDetail(
            phase=VerificationPhase.EVIDENCE_CHAIN,
            passed=passed,
            confidence=confidence,
            findings=findings or f"Evidence chain gaps: {gaps}" if gaps else "No findings",
            evidence=hypothesis.evidence,
            metadata={"gaps": gaps},
        )

    def _parse_poc_result(
        self, content: str, hypothesis: Fact
    ) -> VerificationDetail:
        """Parse LLM PoC generation output."""
        passed = False
        confidence = 0.0
        poc = ""
        prerequisites = ""

        for line in content.split("\n"):
            line = line.strip()
            if line.upper().startswith("PASSED:"):
                passed = line.split(":", 1)[1].strip().lower() == "true"
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.upper().startswith("POC:"):
                poc = line.split(":", 1)[1].strip()
            elif line.upper().startswith("PREREQUISITES:"):
                prerequisites = line.split(":", 1)[1].strip()

        return VerificationDetail(
            phase=VerificationPhase.POC_GENERATE,
            passed=passed,
            confidence=confidence,
            findings=poc or "PoC construction attempted",
            evidence=f"PoC: {poc}\nPrerequisites: {prerequisites}",
            metadata={"prerequisites": prerequisites},
        )

    def _parse_boundary_result(
        self, content: str, hypothesis: Fact
    ) -> VerificationDetail:
        """Parse LLM boundary test output."""
        positive = ""
        negative = ""
        edge = ""
        boundary_desc = ""
        confidence = 0.5  # Default for boundary tests

        for line in content.split("\n"):
            line = line.strip()
            if line.upper().startswith("POSITIVE:"):
                positive = line.split(":", 1)[1].strip()
            elif line.upper().startswith("NEGATIVE:"):
                negative = line.split(":", 1)[1].strip()
            elif line.upper().startswith("EDGE:"):
                edge = line.split(":", 1)[1].strip()
            elif line.upper().startswith("BOUNDARY_DESCRIPTION:"):
                boundary_desc = line.split(":", 1)[1].strip()

        # Boundary test passes if we have positive test description
        passed = bool(positive)
        if positive and negative:
            confidence = 0.8
        elif positive:
            confidence = 0.6

        findings = (
            f"Positive: {positive}\n"
            f"Negative: {negative}\n"
            f"Edge: {edge}"
        )

        return VerificationDetail(
            phase=VerificationPhase.BOUNDARY_TEST,
            passed=passed,
            confidence=confidence,
            findings=findings,
            evidence=boundary_desc or findings,
            metadata={
                "positive": positive,
                "negative": negative,
                "edge": edge,
                "boundary_description": boundary_desc,
            },
        )


