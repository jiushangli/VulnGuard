"""
VulnGuard Observer Agent — Strategic oversight and course correction.

ObserverAgent inherits AgentBase but operates in a fundamentally different pattern:
instead of the claim-Intent → OODA loop of Miners, the Observer watches the
VulnKB for every N new Facts and performs strategic review cycles.

Key design principles (from BreachWeave):
- Default stance: NO_CHANGE > ADJUST_PRIORITY > ADD_INTENT
- Cooldown on reminders: same direction not intervened consecutively
- Failure boundaries must be precise
- "Close the loop first, then narrow, then expand"

Observer reviews:
1. Duplicate exploration detection (activity fingerprint dedup)
2. Intent priority adjustment
3. Missing attack surfaces
4. High-confidence hypothesis verification triggering
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from ..vulnkb.models import (
    AuditPhase,
    Fact,
    FactType,
    FailureBoundary,
    Hint,
    HintSource,
    Intent,
    IntentStatus,
    MinerSpec,
    VulnKB,
    make_fact,
    make_intent,
)

from ..agent_base import AgentBase, AgentConfig, AgentType, OODAResult
from ..tools.descriptor import AuditTool
from ..utils.llm import AgentRole, LLMGateway
from ..utils.prompt import PromptLayer, PromptManager

logger = logging.getLogger(__name__)


# ──────────────────────── Observer Action Types ────────────────────────


class ObserverAction(Enum):
    """
    Types of actions the Observer can recommend after reviewing VulnKB state.

    Ordered by preference: NO_CHANGE is the default stance.
    The Observer should prefer less intervention over more.
    """
    NO_CHANGE = "no_change"                # No adjustments needed
    ADJUST_PRIORITY = "adjust_priority"    # Reprioritize existing Intents
    ADD_INTENT = "add_intent"              # Add new exploration direction
    TRIGGER_VERIFICATION = "trigger_verification"  # Send hypothesis to Verifier
    ISSUE_REMINDER = "issue_reminder"      # Issue course correction reminder


# ──────────────────────── Review Record ────────────────────────


@dataclass
class ReviewRecord:
    """
    Record of a single Observer review cycle.

    Used for cooldown tracking — prevents the Observer from repeatedly
    intervening in the same direction.
    """
    timestamp: str
    fact_count_at_review: int
    actions_taken: list[ObserverAction] = field(default_factory=list)
    direction_fingerprints: list[str] = field(default_factory=list)
    """Fingerprint hashes of directions that were intervened on."""
    summary: str = ""


# ──────────────────────── ObserverAgent ────────────────────────


class ObserverAgent(AgentBase):
    """
    Strategic oversight agent that reviews VulnKB state periodically.

    Unlike Miners and Verifiers, the Observer does NOT claim Intents.
    Instead, it monitors the knowledge graph and performs strategic
    reviews triggered by new Fact additions.

    Review triggers:
    - Every N new Facts (review_every_n_facts)
    - Strategic evaluation of what's been discovered vs. what's missing

    Review behaviors:
    - Duplicate exploration detection (activity fingerprint dedup)
    - Intent priority adjustment
    - Missing attack surface identification
    - High-confidence hypothesis verification triggering
    - Course correction reminders (with cooldown)

    Design principle (BreachWeave):
    "Close the loop first → then narrow → then expand"
    Default stance: NO_CHANGE > adjust existing > add new
    """

    def __init__(
        self,
        agent_id: str | None = None,
        kb: VulnKB | None = None,
        llm: LLMGateway | None = None,
        tools: list[AuditTool] | None = None,
        config: AgentConfig | None = None,
        review_every_n_facts: int = 5,
        reminder_min_interval: int = 3,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.OBSERVER,
            specialization=None,
            kb=kb,
            llm=llm,
            tools=tools or [],
            config=config or AgentConfig(),
        )
        # Observer-specific configuration
        self.review_every_n_facts: int = review_every_n_facts
        """Trigger a review every N new Facts added to VulnKB."""

        self.reminder_min_interval: int = reminder_min_interval
        """Minimum number of new Facts between reminders in the same direction."""

        # Review tracking
        self._last_review_facts: int = 0
        """Fact count at the time of the last review."""
        self._last_reminder_facts: int = 0
        """Fact count at the time of the last reminder issued."""
        self._reminder_directions: dict[str, int] = {}
        """Map of direction fingerprint → fact count when last reminded."""
        self.recent_reviews: list[ReviewRecord] = []
        """Records of recent reviews for cooldown tracking."""

    # ──────────────────────── Prompt ────────────────────────

    def _get_stable_content(self) -> tuple[str, str, str, str]:
        """Generate stable prompt content for Observer role."""
        role_definition = (
            f"You are a VulnGuard Observer agent (id: '{self.agent_id}'). "
            f"Your role is strategic oversight of the vulnerability audit process.\n\n"
            f"You do NOT claim Intents or execute analysis tools. Instead, you:\n"
            f"1. Review the overall audit progress and coverage\n"
            f"2. Detect duplicate or redundant exploration\n"
            f"3. Adjust Intent priorities based on discoveries\n"
            f"4. Identify missing attack surfaces\n"
            f"5. Trigger verification of high-confidence hypotheses\n"
            f"6. Issue course-correction reminders to Miners\n\n"
            f"Core principle: 'Close the loop first, then narrow, then expand.'\n"
            f"Default stance: NO_CHANGE > adjust existing > add new.\n"
            f"Reminder cooldown: never intervene in the same direction consecutively."
        )

        audit_framework = (
            "You interact with the shared VulnKB knowledge graph. Your reviews are "
            "triggered periodically (every N new Facts). Your output takes the form of "
            "ObserverAction decisions:\n"
            "- NO_CHANGE: Audit is proceeding well, no intervention needed\n"
            "- ADJUST_PRIORITY: An existing Intent needs priority adjustment\n"
            "- ADD_INTENT: A new exploration direction should be added\n"
            "- TRIGGER_VERIFICATION: A VULN_HYPOTHESIS has enough evidence to verify\n"
            "- ISSUE_REMINDER: Miners need course correction\n\n"
            "Activity fingerprint dedup: Before adding new Intents, check that similar "
            "directions haven't already been explored. Use content hashing to detect duplicates."
        )

        security_constraints = (
            "SECURITY CONSTRAINTS:\n"
            "1. Never modify existing Facts — they are immutable\n"
            "2. All writes to VulnKB must go through verified admission\n"
            "3. Do not bias verification — let hypotheses stand on their evidence\n"
            "4. Respect reminder cooldown — don't nag Miners about the same issue\n"
            "5. Failure boundaries must be precise (not vague dismissals)\n"
            "6. Prefer NO_CHANGE over unnecessary intervention"
        )

        output_format = (
            "OUTPUT FORMAT:\n"
            "When you recommend actions, specify them as:\n"
            "ACTION: <ObserverAction type>\n"
            "DETAILS: <JSON details>\n\n"
            "For ADJUST_PRIORITY:\n"
            "  ACTION: adjust_priority\n"
            "  DETAILS: {\"intent_id\": \"...\", \"new_priority\": 0.8, \"reason\": \"...\"}\n\n"
            "For ADD_INTENT:\n"
            "  ACTION: add_intent\n"
            "  DETAILS: {\"description\": \"...\", \"priority\": 0.6, \"specialization\": \"...\"}\n\n"
            "For TRIGGER_VERIFICATION:\n"
            "  ACTION: trigger_verification\n"
            "  DETAILS: {\"fact_id\": \"...\", \"confidence\": 0.85, \"reason\": \"...\"}\n\n"
            "For ISSUE_REMINDER:\n"
            "  ACTION: issue_reminder\n"
            "  DETAILS: {\"direction\": \"...\", \"message\": \"...\"}\n\n"
            "For NO_CHANGE:\n"
            "  ACTION: no_change\n"
            "  DETAILS: {\"summary\": \"Audit proceeding well, no intervention needed.\"}\n"
        )

        return role_definition, audit_framework, security_constraints, output_format

    # ──────────────────────── Main Loop ────────────────────────

    async def run(self) -> None:
        """
        Observer main loop — NOT the standard claim-Intent OODA loop.

        The Observer monitors VulnKB and triggers reviews when enough
        new Facts have been accumulated. It does NOT claim Intents.
        """
        if self.kb is None:
            raise RuntimeError(f"ObserverAgent {self.agent_id} has no VulnKB reference")

        self._running = True
        self._initialize_prompt_stable()

        # Initialize baseline fact count
        self._last_review_facts = self.kb.get_fact_count()
        self._last_reminder_facts = self._last_review_facts

        logger.info(f"ObserverAgent {self.agent_id} starting (review every {self.review_every_n_facts} facts)")

        try:
            while self._running:
                current_facts = self.kb.get_fact_count()
                new_facts_since_review = current_facts - self._last_review_facts

                if new_facts_since_review >= self.review_every_n_facts:
                    # Trigger a review cycle
                    await self._review_cycle()
                    self._last_review_facts = current_facts

                # Poll interval — wait for more facts
                await asyncio.sleep(self.config.idle_poll_interval_seconds * 2)

        except asyncio.CancelledError:
            logger.info(f"Observer {self.agent_id} cancelled, shutting down")
        except Exception as e:
            logger.error(f"Observer {self.agent_id} fatal error: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info(f"Observer {self.agent_id} main loop exited")

    async def _review_cycle(self) -> None:
        """Execute a full review cycle: observe → review → act."""
        observations = await self.observe()
        review_result = self.review(self.kb, observations)

        if review_result:
            # Execute the recommended actions
            await self.act(review_result)

    # ──────────────────────── OODA: Observe ────────────────────────

    async def observe(self) -> dict[str, Any]:
        """
        Observe VulnKB state for review.

        Gathers:
        - Current fact counts by type
        - All hypotheses and their confidence levels
        - Pending and claimed intents with their specializations
        - Recent failure boundaries
        - Activity fingerprints for dedup detection
        """
        observations: dict[str, Any] = {}

        if self.kb is None:
            observations["error"] = "No VulnKB reference"
            return observations

        # 1. Fact statistics
        all_facts = self.kb.get_all_facts()
        fact_counts = {}
        for f in all_facts:
            key = f.fact_type.value
            fact_counts[key] = fact_counts.get(key, 0) + 1
        observations["fact_counts"] = fact_counts
        observations["total_facts"] = len(all_facts)

        # 2. Hypotheses with confidence
        hypotheses = self.kb.get_facts_by_type(FactType.VULN_HYPOTHESIS)
        observations["hypotheses"] = [
            {
                "fact_id": h.fact_id,
                "content": h.content,
                "confidence": h.confidence,
                "source": h.source,
                "parent_intents": h.parent_intents,
            }
            for h in hypotheses
        ]

        # 3. High-confidence hypotheses (candidates for verification)
        high_conf_hypotheses = [
            h for h in hypotheses if h.confidence >= 0.7
        ]
        observations["high_confidence_hypotheses"] = [
            {
                "fact_id": h.fact_id,
                "content": h.content,
                "confidence": h.confidence,
            }
            for h in high_conf_hypotheses
        ]

        # 4. Pending and claimed intents
        pending = self.kb.get_pending_intents()
        observations["pending_intents"] = [
            {
                "intent_id": i.intent_id,
                "description": i.description,
                "priority": i.priority,
                "specialization": i.spec.value if i.specialization else None,
            }
            for i in pending[:20]  # Limit for token budget
        ]

        # 5. Failure boundaries
        boundaries = self.kb.get_facts_by_type(FactType.FAILURE_BOUNDARY)
        observations["failure_boundaries"] = [
            {
                "fact_id": b.fact_id,
                "content": b.content,
            }
            for b in boundaries
        ]

        # 6. Activity fingerprint dedup
        # Hash the content of all facts to detect duplicate exploration
        activity_fingerprints = set()
        for f in all_facts:
            fp = self._activity_fingerprint(f.content)
            activity_fingerprints.add(fp)
        observations["activity_fingerprint_count"] = len(activity_fingerprints)
        observations["activity_fingerprints"] = list(activity_fingerprints)

        # 7. Coverage analysis: which specializations have pending intents?
        spec_coverage = {}
        for i in pending:
            spec = i.specialization.value if i.specialization else "general"
            spec_coverage[spec] = spec_coverage.get(spec, 0) + 1
        observations["specialization_coverage"] = spec_coverage

        return observations

    @staticmethod
    def _activity_fingerprint(content: str) -> str:
        """
        Generate a fingerprint hash for activity deduplication.

        Normalizes the content (lowercase, strip whitespace) and hashes it
        to detect semantically similar exploration directions.
        """
        normalized = content.lower().strip()
        # Simple normalization: remove extra whitespace
        normalized = " ".join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    # ──────────────────────── Review ────────────────────────

    def review(self, kb: VulnKB | None = None, observations: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Review the current state of VulnKB and determine if action is needed.

        This is the core strategic review method. It follows the principle:
        "Close the loop first, then narrow, then expand."

        Default stance: NO_CHANGE > adjust > add.

        Checks:
        1. Duplicate exploration detection (activity fingerprint dedup)
        2. Intent priority adjustment (based on discoveries)
        3. Missing attack surfaces (specializations without coverage)
        4. High-confidence hypotheses ready for verification
        5. Course correction reminders (with cooldown)

        Returns:
            Dictionary with 'actions' list and 'assessment' string.
        """
        if kb is None:
            kb = self.kb
        if observations is None:
            observations = {}

        actions: list[dict[str, Any]] = []
        assessment_parts: list[str] = []

        current_facts = observations.get("total_facts", 0)
        fact_counts = observations.get("fact_counts", {})
        hypotheses = observations.get("hypotheses", [])
        high_conf_hypotheses = observations.get("high_confidence_hypotheses", [])
        pending_intents = observations.get("pending_intents", [])
        spec_coverage = observations.get("specialization_coverage", {})
        activity_fingerprints = observations.get("activity_fingerprints", [])

        # ── Check 1: High-confidence hypotheses → trigger verification ──
        for h in high_conf_hypotheses:
            actions.append({
                "action": ObserverAction.TRIGGER_VERIFICATION,
                "fact_id": h["fact_id"],
                "confidence": h["confidence"],
                "content": h["content"],
                "reason": f"Hypothesis with confidence {h['confidence']:.2f} ready for verification",
            })
            assessment_parts.append(
                f"High-confidence hypothesis {h['fact_id'][:16]}... "
                f"(conf={h['confidence']:.2f}) → trigger verification"
            )

        # ── Check 2: Missing specialization coverage → add intents ──
        all_specs = [s.value for s in MinerSpec]
        missing_specs = [s for s in all_specs if s not in spec_coverage]
        # Only add if there are few facts (early stage) and we don't already
        # have too many pending intents
        if missing_specs and len(pending_intents) < 20:
            for spec in missing_specs[:2]:  # Add at most 2 new intents per review
                fp = self._activity_fingerprint(f"explore_{spec}")
                if not self._is_on_cooldown(fp):
                    actions.append({
                        "action": ObserverAction.ADD_INTENT,
                        "description": f"Systematic exploration of {spec} attack surface",
                        "priority": 0.5,
                        "specialization": spec,
                        "reason": f"No pending intents for {spec} specialization",
                    })
                    assessment_parts.append(
                        f"Missing coverage: {spec} → adding exploration intent"
                    )

        # ── Check 3: Duplicate exploration detection ──
        # Check for similar pending intents
        seen_descriptions = set()
        duplicates_found = 0
        for intent in pending_intents:
            fp = self._activity_fingerprint(intent["description"])
            if fp in seen_descriptions:
                duplicates_found += 1
            else:
                seen_descriptions.add(fp)
        if duplicates_found > 0:
            assessment_parts.append(f"Found {duplicates_found} duplicate intent descriptions")

        # ── Check 4: Priority adjustment based on discoveries ──
        # If we have confirmed vulnerabilities, lower priority of exploration;
        # if we have many hypotheses but few boundaries, raise exploration priority
        vuln_count = fact_counts.get("vulnerability", 0)
        hypothesis_count = fact_counts.get("vuln_hypothesis", 0)
        boundary_count = fact_counts.get("failure_boundary", 0)

        if hypothesis_count > 0 and boundary_count < hypothesis_count // 2:
            # More exploration needed — not enough boundaries
            for intent in pending_intents[:3]:
                if intent["priority"] < 0.7:
                    actions.append({
                        "action": ObserverAction.ADJUST_PRIORITY,
                        "intent_id": intent["intent_id"],
                        "new_priority": min(intent["priority"] + 0.1, 1.0),
                        "reason": "Elevating priority: high hypothesis count relative to boundaries",
                    })
                    assessment_parts.append(
                        f"Elevating intent {intent['intent_id'][:16]}... priority"
                    )
                    break  # Only adjust one per review to prevent overcorrection

        # ── Check 5: Course correction reminders (with cooldown) ──
        # If Miners are exploring too broadly without depth, issue reminders
        if current_facts > 0:
            exploration_ratio = len(pending_intents) / max(current_facts, 1)
            if exploration_ratio > 2.0:
                # Too many pending intents relative to facts produced
                direction = "depth_over_breadth"
                fp = self._activity_fingerprint(direction)
                if not self._is_on_cooldown(fp):
                    actions.append({
                        "action": ObserverAction.ISSUE_REMINDER,
                        "direction": direction,
                        "message": (
                            "Too many pending intents relative to facts produced. "
                            "Focus on depth over breadth — close open loops before "
                            "opening new exploration directions."
                        ),
                    })
                    assessment_parts.append("Reminder: focus on depth over breadth")
            elif vuln_count > 0 and current_facts < vuln_count * 5:
                # Confirmed vulns exist but not enough follow-up
                direction = "follow_up_vulns"
                fp = self._activity_fingerprint(direction)
                if not self._is_on_cooldown(fp):
                    actions.append({
                        "action": ObserverAction.ISSUE_REMINDER,
                        "direction": direction,
                        "message": (
                            "Confirmed vulnerabilities exist but insufficient follow-up "
                            "analysis. miners should trace exploit chains and identify "
                            "related attack vectors."
                        ),
                    })
                    assessment_parts.append("Reminder: follow up on confirmed vulnerabilities")

        # ── Default stance: prefer NO_CHANGE ──
        if not actions:
            actions.append({
                "action": ObserverAction.NO_CHANGE,
                "summary": "Audit proceeding well, no intervention needed.",
            })

        # Apply priority ordering: NO_CHANGE > ADJUST_PRIORITY > ADD_INTENT > TRIGGER > REMINDER
        actions = self._action_priority(actions)

        # Record this review
        review_record = ReviewRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            fact_count_at_review=current_facts,
            actions_taken=[a.get("action", ObserverAction.NO_CHANGE) for a in actions],
            direction_fingerprints=[
                self._activity_fingerprint(str(a.get("description", a.get("direction", ""))))
                for a in actions
                if a.get("action") != ObserverAction.NO_CHANGE
            ],
            summary="; ".join(assessment_parts) if assessment_parts else "No significant observations",
        )
        self.recent_reviews.append(review_record)

        # Keep only the last 50 reviews
        if len(self.recent_reviews) > 50:
            self.recent_reviews = self.recent_reviews[-50:]

        result = {
            "actions": actions,
            "assessment": " | ".join(assessment_parts) if assessment_parts else "No intervention needed",
            "review_number": len(self.recent_reviews),
        }

        logger.info(
            f"Observer {self.agent_id} review #{len(self.recent_reviews)}: "
            f"{len(actions)} actions, {assessment_parts[0] if assessment_parts else 'no changes'}"
        )

        return result

    def _is_on_cooldown(self, direction_fingerprint: str) -> bool:
        """
        Check if a direction is on reminder cooldown.

        A direction is on cooldown if we issued a reminder about it recently,
        and fewer than reminder_min_interval new Facts have been added since.
        """
        last_facts = self._reminder_directions.get(direction_fingerprint)
        if last_facts is None:
            return False

        current_facts = self.kb.get_fact_count() if self.kb else 0
        facts_since_reminder = current_facts - last_facts
        return facts_since_reminder < self.reminder_min_interval

    def _action_priority(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sort actions by priority preference.

        Default stance: NO_CHANGE > ADJUST_PRIORITY > ADD_INTENT > TRIGGER_VERIFICATION > ISSUE_REMINDER
        This ensures minimal intervention — the observer prefers to not change things.
        """
        priority_order = {
            ObserverAction.NO_CHANGE: 0,
            ObserverAction.ADJUST_PRIORITY: 1,
            ObserverAction.ADD_INTENT: 2,
            ObserverAction.TRIGGER_VERIFICATION: 3,
            ObserverAction.ISSUE_REMINDER: 4,
        }

        def sort_key(action_dict: dict) -> int:
            action = action_dict.get("action", ObserverAction.NO_CHANGE)
            if isinstance(action, ObserverAction):
                return priority_order.get(action, 99)
            return 99

        sorted_actions = sorted(actions, key=sort_key)

        # If the first action is NO_CHANGE, drop the rest (no intervention needed)
        if sorted_actions and sorted_actions[0].get("action") == ObserverAction.NO_CHANGE:
            return sorted_actions[:1]

        return sorted_actions

    # ──────────────────────── Build Review Prompt ────────────────────────

    def _build_review_prompt(self, observations: dict[str, Any]) -> str:
        """
        Build a review prompt for LLM-based analysis.

        This constructs a comprehensive prompt with current VulnKB state
        for the LLM to analyze and recommend ObserverActions.
        """
        fact_counts = observations.get("fact_counts", {})
        hypotheses = observations.get("hypotheses", [])
        high_conf = observations.get("high_confidence_hypotheses", [])
        pending = observations.get("pending_intents", [])
        coverage = observations.get("specialization_coverage", {})
        boundaries = observations.get("failure_boundaries", [])

        prompt_parts = [
            "## VulnKB State Review\n",
            f"Total facts: {observations.get('total_facts', 0)}",
            f"Fact breakdown: {json.dumps(fact_counts)}\n",
            f"High-confidence hypotheses: {len(high_conf)}",
        ]

        for h in high_conf:
            prompt_parts.append(
                f"  - [{h['fact_id'][:12]}] conf={h['confidence']:.2f}: {h['content'][:80]}"
            )

        prompt_parts.append(f"\nPending intents: {len(pending)}")
        for i in pending[:10]:
            prompt_parts.append(
                f"  - [{i['intent_id'][:12]}] pri={i['priority']:.2f} "
                f"spec={i['specialization'] or 'general'}: {i['description'][:80]}"
            )

        prompt_parts.append(f"\nSpecialization coverage: {json.dumps(coverage)}")
        prompt_parts.append(f"Failure boundaries: {len(boundaries)}\n")

        # Recent review history
        recent = self.recent_reviews[-5:]
        if recent:
            prompt_parts.append("## Recent Reviews\n")
            for r in recent:
                actions_str = ", ".join(
                    a.value if isinstance(a, ObserverAction) else str(a)
                    for a in r.actions_taken
                )
                prompt_parts.append(
                    f"  [{r.timestamp[:19]}] facts={r.fact_count_at_review}: "
                    f"{actions_str} — {r.summary[:60]}"
                )

        prompt_parts.append("\n## Instructions\n")
        prompt_parts.append(
            "Based on the above, determine if any ObserverActions are needed.\n"
            "Remember: default stance is NO_CHANGE. Prefer less intervention.\n"
            "Principle: 'Close the loop first, then narrow, then expand.'\n"
            "Check for: duplicate exploration, missing coverage, high-confidence "
            "hypotheses needing verification, and course corrections.\n"
        )

        return "\n".join(prompt_parts)

    # ──────────────────────── OODA: Orient ────────────────────────

    async def orient(self, observations: dict[str, Any]) -> dict[str, Any]:
        """
        Orient phase for Observer: evaluate audit progress and coverage.

        Since Observer doesn't follow the standard OODA loop (it reviews
        periodically rather than acting on Intents), orient assesses
        the overall audit state.
        """
        orientation: dict[str, Any] = {}
        orientation["assessment"] = "Observer review cycle"

        fact_counts = observations.get("fact_counts", {})
        total = observations.get("total_facts", 0)

        # Progressive vs stuck
        if total > 0 and len(self.recent_reviews) >= 2:
            prev_facts = self.recent_reviews[-1].fact_count_at_review
            if total > prev_facts:
                orientation["progress"] = "advancing"
            else:
                orientation["progress"] = "stalled"
        else:
            orientation["progress"] = "starting"

        return orientation

    # ──────────────────────── OODA: Decide ────────────────────────

    async def decide(self, orientation: dict[str, Any]) -> dict[str, Any]:
        """
        Decide phase for Observer: determine if any action is warranted.

        The review() method already computes the actions, so this is
        primarily a pass-through for OODA compatibility.
        """
        return {
            "action": "review",
            "assessment": orientation.get("assessment", ""),
            "progress": orientation.get("progress", "unknown"),
        }

    # ──────────────────────── OODA: Act ────────────────────────

    async def act(self, decision: dict[str, Any]) -> OODAResult:
        """
        Execute ObserverActions from a review.

        For each action in the review result:
        - ADJUST_PRIORITY: Update intent priority in VulnKB
        - ADD_INTENT: Submit a new Intent
        - TRIGGER_VERIFICATION: Transform VULN_HYPOTHESIS for Verifier
        - ISSUE_REMINDER: Add a Hint to VulnKB as course correction
        - NO_CHANGE: Do nothing
        """
        actions = decision.get("actions", [])
        facts_produced: list[Fact] = []
        intents_produced: list[Intent] = []
        boundaries_produced: list[FailureBoundary] = []
        observations_list: list[str] = []

        for action_dict in actions:
            action = action_dict.get("action", ObserverAction.NO_CHANGE)

            if isinstance(action, str):
                try:
                    action = ObserverAction(action)
                except ValueError:
                    continue

            if action == ObserverAction.NO_CHANGE:
                observations_list.append(action_dict.get("summary", "No changes needed"))

            elif action == ObserverAction.ADJUST_PRIORITY:
                # Update intent priority in VulnKB
                intent_id = action_dict.get("intent_id")
                new_priority = action_dict.get("new_priority", 0.5)
                if intent_id and self.kb is not None:
                    # VulnKB doesn't have a direct priority update, so we
                    # record this as an observer Hint
                    hint_content = (
                        f"Priority adjustment: Intent {intent_id} should be "
                        f"priority {new_priority:.2f}. Reason: {action_dict.get('reason', '')}"
                    )
                    observations_list.append(hint_content)

            elif action == ObserverAction.ADD_INTENT:
                # Submit a new Intent for mining
                desc = action_dict.get("description", "Explore unexplored attack surface")
                priority = action_dict.get("priority", 0.5)
                spec_str = action_dict.get("specialization")
                spec = None
                if spec_str:
                    try:
                        spec = MinerSpec(spec_str)
                    except ValueError:
                        spec = None
                intent = self.submit_intent(
                    description=desc,
                    priority=priority,
                    specialization=spec,
                )
                if intent is not None:
                    intents_produced.append(intent)
                    observations_list.append(f"Added intent: {desc[:60]}")

            elif action == ObserverAction.TRIGGER_VERIFICATION:
                # Mark hypothesis for verification — add a high-priority Intent
                # targeting the Verifier
                fact_id = action_dict.get("fact_id", "")
                content = action_dict.get("content", "")
                reason = action_dict.get("reason", "")
                intent = self.submit_intent(
                    description=f"Verify hypothesis {fact_id[:16]}: {content[:60]}. {reason}",
                    priority=0.9,  # High priority for verification
                    specialization=None,  # Verifier doesn't have a specialization
                )
                if intent is not None:
                    intents_produced.append(intent)
                    observations_list.append(f"Triggered verification for {fact_id[:16]}")

            elif action == ObserverAction.ISSUE_REMINDER:
                # Record direction fingerprint for cooldown
                direction = action_dict.get("direction", "unknown")
                fp = self._activity_fingerprint(direction)
                current_facts = self.kb.get_fact_count() if self.kb else 0
                self._reminder_directions[fp] = current_facts
                self._last_reminder_facts = current_facts

                # Add reminder as a Hint to VulnKB
                message = action_dict.get("message", "Course correction needed")
                if self.kb is not None:
                    from ..vulnkb.models import Hint, HintSource, make_hint
                    hint = make_hint(
                        pattern=f"Observer reminder: {direction}",
                        applicability=message,
                        severity="medium",
                        source=HintSource.OBSERVER,
                    )
                    self.kb.add_hint(hint)

                observations_list.append(f"Issued reminder: {message[:60]}")

        return OODAResult(
            observations=observations_list,
            orientation=decision.get("assessment", ""),
            decision="review",
            action_output=actions,
            facts_produced=facts_produced,
            intents_produced=intents_produced,
            boundaries_produced=boundaries_produced,
            should_continue=True,
            cycle_number=len(self.recent_reviews),
        )


