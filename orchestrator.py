"""
VulnGuard Orchestrator — Top-level controller for the entire audit pipeline.

Responsibilities:
1. Parse audit target and load configuration
2. Initialize VulnKB (knowledge base)
3. Run CodeIntelligenceEngine for Phase 0 (static analysis)
4. Inject initial Facts, Intents, and Hints into VulnKB
5. Start and monitor Miner/Observer/Verifier agent clusters
6. Produce the final AuditReport

Orchestration flow:
  Phase 0: Initialize → CodeIntelligence
  Phase 1: Vuln Mining (parallel Miner + Observer)
  Phase 2: Verification (Verifier agents)
  Phase 3: Report generation
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from vulnkb.models import (
    AuditPhase,
    Fact,
    FactType,
    FailureBoundary,
    Hint,
    HintSource,
    Intent,
    IntentStatus,
    MinerSpec,
    VerificationResult,
    VulnKB,
    make_fact,
    make_intent,
    make_hint,
)

from agent_base import AgentBase, AgentConfig, AgentType
from miner.agent import MinerAgent
from observer.agent import ObserverAgent
from verifier.agent import VerifierAgent

from intelligence.engine import CodeIntelligenceEngine, IntelligenceConfig

from tools.registry import AuditToolRegistry

from utils.llm import LLMGateway, LLMProvider, AgentRole

from config.settings import VulnGuardConfig

logger = logging.getLogger(__name__)


# ──────────────────────── Audit Report Data Classes ────────────────────────


@dataclass
class VulnerabilityEntry:
    """A confirmed vulnerability entry in the audit report."""
    fact_id: str
    vuln_type: str
    description: str
    evidence: str
    confidence: float
    severity: str = "high"
    remediation: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class FailureBoundaryEntry:
    """A ruled-out attack direction entry in the audit report."""
    fact_id: str
    vuln_type: str
    ruled_out: str
    remaining_risk: str
    confidence: float


@dataclass
class AuditReport:
    """
    Final audit report produced by the Orchestrator.

    Contains all findings (confirmed vulnerabilities and failure boundaries),
    statistics, and human-readable summaries.
    """
    target: str
    language: str
    total_findings: int
    vulnerabilities: list[VulnerabilityEntry] = field(default_factory=list)
    ruled_outs: list[FailureBoundaryEntry] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)
    executive_summary: str = ""
    detailed_report: str = ""
    started_at: str = ""
    completed_at: str = ""


# ──────────────────────── Rule Loading ────────────────────────


# Built-in vulnerability pattern hints (OWASP API Top 10 + CWE Top 25)
BUILTIN_RULES: dict[str, list[dict]] = {
    "owasp_api_top10": [
        {
            "pattern": "Broken Object Level Authorization (BOLA)",
            "applicability": "API endpoints that access objects by ID without ownership checks",
            "severity": "critical",
            "references": ["OWASP API1:2019"],
        },
        {
            "pattern": "Broken Function Level Authorization (BFLA)",
            "applicability": "Administrative endpoints accessible to regular users",
            "severity": "critical",
            "references": ["OWASP API2:2019"],
        },
        {
            "pattern": "Excessive Data Exposure",
            "applicability": "API endpoints returning full objects instead of filtered DTOs",
            "severity": "high",
            "references": ["OWASP API3:2019"],
        },
        {
            "pattern": "Lack of Resources & Rate Limiting",
            "applicability": "API endpoints without rate limiting or pagination",
            "severity": "medium",
            "references": ["OWASP API4:2019"],
        },
        {
            "pattern": "Broken Authentication",
            "applicability": "Endpoints with weak or missing authentication mechanisms",
            "severity": "critical",
            "references": ["OWASP API5:2019"],
        },
        {
            "pattern": "Mass Assignment",
            "applicability": "Endpoints accepting client-controlled object properties without whitelist",
            "severity": "high",
            "references": ["OWASP API6:2019"],
        },
        {
            "pattern": "Security Misconfiguration",
            "applicability": "CORS, headers, TLS, or error handling misconfigurations",
            "severity": "medium",
            "references": ["OWASP API7:2019"],
        },
        {
            "pattern": "Injection (SQL, NoSQL, Command, LDAP)",
            "applicability": "Endpoints with unsanitized user input reaching interpreters",
            "severity": "critical",
            "references": ["OWASP API8:2019"],
        },
        {
            "pattern": "Improper Inventory Management",
            "applicability": "Undocumented or unmonitored API endpoints in production",
            "severity": "medium",
            "references": ["OWASP API9:2019"],
        },
        {
            "pattern": "Insufficient Logging & Monitoring",
            "applicability": "Security-relevant events without proper logging",
            "severity": "low",
            "references": ["OWASP API10:2019"],
        },
    ],
    "cwe_top_25": [
        {
            "pattern": "CWE-79: Cross-site Scripting (XSS)",
            "applicability": "Web output reflecting unsanitized user input",
            "severity": "high",
            "references": ["CWE-79"],
        },
        {
            "pattern": "CWE-89: SQL Injection",
            "applicability": "Database queries with unsanitized user input",
            "severity": "critical",
            "references": ["CWE-89"],
        },
        {
            "pattern": "CWE-22: Path Traversal",
            "applicability": "File operations with user-controlled paths",
            "severity": "high",
            "references": ["CWE-22"],
        },
        {
            "pattern": "CWE-78: OS Command Injection",
            "applicability": "Shell/exec operations with unsanitized user input",
            "severity": "critical",
            "references": ["CWE-78"],
        },
        {
            "pattern": "CWE-287: Improper Authentication",
            "applicability": "Authentication logic flaws or bypass",
            "severity": "critical",
            "references": ["CWE-287"],
        },
        {
            "pattern": "CWE-862: Missing Authorization",
            "applicability": "Endpoints or functions lacking authorization checks",
            "severity": "critical",
            "references": ["CWE-862"],
        },
        {
            "pattern": "CWE-306: Missing Authentication",
            "applicability": "Functions or endpoints exposed without authentication",
            "severity": "critical",
            "references": ["CWE-306"],
        },
        {
            "pattern": "CWE-502: Deserialization of Untrusted Data",
            "applicability": "Deserializing objects from untrusted sources (pickle, YAML, JSON)",
            "severity": "high",
            "references": ["CWE-502"],
        },
        {
            "pattern": "CWE-918: Server-Side Request Forgery (SSRF)",
            "applicability": "User-controlled URLs causing server-side requests",
            "severity": "high",
            "references": ["CWE-918"],
        },
        {
            "pattern": "CWE-190: Integer Overflow/Underflow",
            "applicability": "Arithmetic operations on untrusted input without bounds checking",
            "severity": "medium",
            "references": ["CWE-190"],
        },
    ],
}


# ──────────────────────── Orchestrator ────────────────────────


class Orchestrator:
    """
    Top-level orchestrator for the VulnGuard audit framework.

    Coordinates all phases of the audit:
    - Phase 0: Initialization + Code Intelligence
    - Phase 1: Parallel vulnerability mining (Miners + Observer)
    - Phase 2: Independent verification (Verifiers)
    - Phase 3: Report generation
    """

    def __init__(self, config: VulnGuardConfig):
        self.config = config
        self.kb: Optional[VulnKB] = None
        self.intelligence_engine: Optional[CodeIntelligenceEngine] = None
        self.miners: list[MinerAgent] = []
        self.observers: list[ObserverAgent] = []
        self.verifiers: list[VerifierAgent] = []
        self.llm_gateway: Optional[LLMGateway] = None
        self.tool_registry: Optional[AuditToolRegistry] = None

        # Tracking
        self._started_at: str = ""
        self._completed_at: str = ""
        self._phase: str = "initialized"
        self._logger = logging.getLogger("orchestrator")

    async def run(self, phase: str | None = None) -> AuditReport:
        """
        Main execution flow for a complete audit.

        Args:
            phase: Optional phase limit. If specified, only run up to
                   that phase. One of: "intelligence", "mining",
                   "verification". None means run all phases.

        Returns:
            AuditReport with all findings.
        """
        self._started_at = datetime.now(timezone.utc).isoformat()

        try:
            # Phase 0: Initialize
            self._logger.info("=" * 60)
            self._logger.info("VulnGuard Audit Starting")
            self._logger.info(f"Target: {self.config.target.repo_path}")
            self._logger.info(f"Language: {self.config.target.language}")
            self._logger.info("=" * 60)

            await self._initialize()

            if phase == "intelligence":
                self._logger.info("Running intelligence phase only")
                await self._run_intelligence()
                return self._generate_report()

            # Phase 0: Code Intelligence
            await self._run_intelligence()

            if phase == "mining":
                self._logger.info("Running up to mining phase only")
                await self._run_mining()
                return self._generate_report()

            # Phase 1: Vuln Mining
            await self._run_mining()

            if phase == "verification":
                self._logger.info("Running up to verification phase only")
                await self._run_verification()
                return self._generate_report()

            # Phase 2: Verification
            await self._run_verification()

            # Phase 3: Report
            self._completed_at = datetime.now(timezone.utc).isoformat()
            return self._generate_report()

        except Exception as e:
            self._logger.error(f"Audit failed: {e}", exc_info=True)
            self._completed_at = datetime.now(timezone.utc).isoformat()
            report = self._generate_report()
            report.statistics["error"] = str(e)
            return report

    # ──────────────────────── Phase 0: Initialize ────────────────────────

    async def _initialize(self) -> None:
        """
        Initialize all components for the audit.

        Creates:
        - VulnKB instance (knowledge base)
        - LLMGateway with configured providers
        - AuditToolRegistry
        - Loads vulnerability rules as Hints
        - Creates agent instances (but does not start them)
        """
        self._phase = "initializing"
        self._logger.info("Phase 0: Initializing components...")

        # 1. Create VulnKB
        self._logger.info("  Creating VulnKB...")
        self.kb = VulnKB(db_path=self.config.vulnkb.db_path)
        self._logger.info(f"  VulnKB initialized (db_path={self.config.vulnkb.db_path})")

        # 2. Create LLMGateway
        self._logger.info("  Creating LLMGateway...")
        self.llm_gateway = LLMGateway()
        self._setup_llm_providers()
        self._logger.info(
            f"  LLMGateway initialized with roles: {self.llm_gateway.available_roles}"
        )

        # 3. Create ToolRegistry
        self._logger.info("  Creating AuditToolRegistry...")
        self.tool_registry = AuditToolRegistry()
        # TODO: Register built-in tools when available
        self._logger.info(f"  AuditToolRegistry initialized ({len(self.tool_registry)} tools)")

        # 4. Load vulnerability rules as Hints
        self._logger.info("  Loading vulnerability rules...")
        self._load_rules_as_hints()

        self._logger.info("Phase 0 initialization complete.")

    def _setup_llm_providers(self) -> None:
        """Set up LLM providers from configuration."""
        for name, prov_config in self.config.llm.providers.items():
            provider = self._create_provider(prov_config)
            if provider is None:
                self._logger.warning(
                    f"  Skipping LLM provider '{name}': no implementation registered"
                )
                continue

            # Map provider to roles based on role_mapping
            for role_name, mapped_provider in self.config.llm.role_mapping.items():
                if mapped_provider == name:
                    role = AgentRole(role_name)
                    fallback = (name == "primary")
                    self.llm_gateway.register_provider(role, provider, fallback=fallback)
                    self._logger.info(
                        f"  Registered provider '{prov_config.name}' for role '{role_name}'"
                    )

    def _create_provider(self, config) -> Optional[LLMProvider]:
        """
        Create an LLM provider instance from configuration.

        Currently returns None as concrete providers (OpenAI, Anthropic, etc.)
        are expected to be plugged in by the user or a separate provider package.
        The orchestrator sets up the routing structure; actual provider
        implementations need to be registered externally.
        """
        # Provider creation is a hook point for integration.
        # The LLMGateway supports dynamic registration, so providers
        # can be added after initialization.
        self._logger.debug(
            f"  Provider config: name={config.name}, model={config.model}"
        )
        return None

    def _load_rules_as_hints(self) -> None:
        """Load vulnerability rule libraries as Hints into VulnKB."""
        hint_count = 0
        for source_name in self.config.rules.sources:
            rules = BUILTIN_RULES.get(source_name, [])
            for rule in rules:
                hint = make_hint(
                    pattern=rule["pattern"],
                    applicability=rule["applicability"],
                    severity=rule["severity"],
                    source=HintSource.RULE_LIBRARY,
                    references=rule.get("references", []),
                )
                self.kb.add_hint(hint)
                hint_count += 1

        # Load custom rules if specified
        if self.config.rules.custom_rules_path:
            self._load_custom_rules(self.config.rules.custom_rules_path)

        self._logger.info(f"  Loaded {hint_count} vulnerability rule hints")

    def _load_custom_rules(self, path: str) -> None:
        """Load custom rule definitions from a YAML or JSON file."""
        import json
        from pathlib import Path

        rules_path = Path(path)
        if not rules_path.exists():
            self._logger.warning(f"  Custom rules file not found: {path}")
            return

        try:
            if rules_path.suffix in (".yaml", ".yml"):
                import yaml
                with open(rules_path) as f:
                    rules_data = yaml.safe_load(f)
            elif rules_path.suffix == ".json":
                with open(rules_path) as f:
                    rules_data = json.load(f)
            else:
                self._logger.warning(f"  Unsupported rules format: {rules_path.suffix}")
                return

            if isinstance(rules_data, list):
                for rule in rules_data:
                    hint = make_hint(
                        pattern=rule.get("pattern", ""),
                        applicability=rule.get("applicability", ""),
                        severity=rule.get("severity", "medium"),
                        source=HintSource.RULE_LIBRARY,
                        references=rule.get("references", []),
                    )
                    self.kb.add_hint(hint)
            elif isinstance(rules_data, dict):
                # Support dict format: {"rules": [...]}
                for rule in rules_data.get("rules", []):
                    hint = make_hint(
                        pattern=rule.get("pattern", ""),
                        applicability=rule.get("applicability", ""),
                        severity=rule.get("severity", "medium"),
                        source=HintSource.RULE_LIBRARY,
                        references=rule.get("references", []),
                    )
                    self.kb.add_hint(hint)

        except Exception as e:
            self._logger.error(f"  Failed to load custom rules from {path}: {e}")

    # ──────────────────────── Phase 0: Code Intelligence ────────────────────────

    async def _run_intelligence(self) -> None:
        """
        Run the CodeIntelligenceEngine and inject initial data into VulnKB.

        The intelligence pipeline:
        1. Parse source code → CodeNodes
        2. Build dependency graph
        3. Cluster into modules
        4. Extract API sequences
        5. Detect vulnerability hypotheses
        6. Convert intelligence results to VulnKB Facts and Intents
        """
        self._phase = "intelligence"
        self._logger.info("Phase 0: Running Code Intelligence...")

        # Build IntelligenceConfig from our config
        intel_config = IntelligenceConfig(
            repo_path=self.config.target.repo_path,
            max_tokens_per_module=self.config.intelligence.max_tokens_per_module,
            max_module_depth=self.config.intelligence.max_cluster_depth,
            enable_api_extraction=self.config.intelligence.extract_api_routes,
            enable_vuln_hypotheses=True,
            exclude_patterns=self.config.target.exclude_patterns,
        )

        # Create and run intelligence engine
        self.intelligence_engine = CodeIntelligenceEngine(config=intel_config)

        self._logger.info(f"  Running intelligence on: {self.config.target.repo_path}")
        result = self.intelligence_engine.run(
            repo_path=self.config.target.repo_path,
            config=intel_config,
        )

        self._logger.info(
            f"  Intelligence complete: {len(result.nodes)} nodes, "
            f"{len(result.initial_facts)} facts, "
            f"{len(result.initial_intents)} intents"
        )

        # Convert intelligence facts to VulnKB Facts and inject
        facts_added = 0
        for intel_fact in result.initial_facts:
            # Map intelligence fact type to VulnKB FactType
            fact_type = self._map_intel_category_to_fact_type(intel_fact.category.value)
            vulnkb_fact = make_fact(
                fact_type=fact_type,
                content=intel_fact.title + ": " + intel_fact.description,
                source="intelligence_engine",
                evidence="\n".join(intel_fact.evidence) if intel_fact.evidence else intel_fact.description,
                confidence=self._severity_to_confidence(intel_fact.severity.value),
                metadata=intel_fact.metadata,
            )
            result = self.kb.add_fact(vulnkb_fact, verify=False)
            if result.admitted:
                facts_added += 1

        # Convert intelligence intents to VulnKB Intents and inject
        intents_added = 0
        for intel_intent in result.initial_intents:
            # Map intelligence intent type to MinerSpec
            spec = self._map_intent_type_to_spec(intel_intent.type.value)
            vulnkb_intent = make_intent(
                description=f"{intel_intent.type.value}: {intel_intent.target} - {intel_intent.rationale}",
                from_facts=intel_intent.related_facts if intel_intent.related_facts else [],
                priority=intel_intent.priority / 10.0,  # Convert 1-10 scale to 0.0-1.0
                specialization=spec,
                metadata=intel_intent.metadata,
            )
            self.kb.add_intent(vulnkb_intent)
            intents_added += 1

        # Inject entry points as initial exploration intents
        if self.config.target.entry_points:
            for ep in self.config.target.entry_points:
                intent = make_intent(
                    description=f"Deep dive into entry point: {ep}",
                    priority=0.8,
                    specialization=None,
                    metadata={"entry_point": ep, "source": "config"},
                )
                self.kb.add_intent(intent)
                intents_added += 1

        self._logger.info(
            f"  Injected {facts_added} Facts and {intents_added} Intents into VulnKB"
        )

    def _map_intel_category_to_fact_type(self, category: str) -> FactType:
        """Map intelligence engine category to VulnKB FactType."""
        mapping = {
            "api_surface": FactType.API_ENDPOINT,
            "auth_mechanism": FactType.SECURITY_CONTROL,
            "data_model": FactType.DATAFLOW,
            "dependency": FactType.CODE_STRUCTURE,
            "architecture": FactType.CODE_STRUCTURE,
            "vuln_hypothesis": FactType.VULN_HYPOTHESIS,
        }
        return mapping.get(category, FactType.CODE_STRUCTURE)

    def _map_intent_type_to_spec(self, intent_type: str) -> Optional[MinerSpec]:
        """Map intelligence engine intent type to MinerSpec."""
        mapping = {
            "investigate": None,  # General investigation, no specific spec
            "verify": MinerSpec.ATTACK_SURFACE,
            "explore": None,
            "deep_dive": None,
        }
        return mapping.get(intent_type)

    def _severity_to_confidence(self, severity: str) -> float:
        """Convert severity string to confidence float."""
        mapping = {
            "critical": 0.95,
            "high": 0.85,
            "warning": 0.7,
            "medium": 0.5,
            "low": 0.3,
            "info": 0.2,
        }
        return mapping.get(severity, 0.5)

    # ──────────────────────── Phase 1: Vuln Mining ────────────────────────

    async def _run_mining(self) -> None:
        """
        Run parallel vulnerability mining with Miner and Observer agents.

        Creates N Miner agents (one per specialization) and optionally
        an Observer agent. They run concurrently, with Miners claiming
        Intents from VulnKB and the Observer monitoring progress.

        The mining phase continues until:
        - No more pending intents in VulnKB
        - No more claimed (in-progress) intents
        - Maximum budget/cycles reached
        """
        self._phase = "vuln_mining"
        self._logger.info("Phase 1: Starting Vuln Mining...")

        # Create Miner agents based on configuration
        self._create_miners()

        # Create Observer agent if enabled
        if self.config.agents.observer_enabled:
            self._create_observers()

        self._logger.info(
            f"  Starting {len(self.miners)} Miners, "
            f"{len(self.observers)} Observers"
        )

        # Build agent budget config
        agent_config = AgentConfig(
            max_ooda_cycles=self.config.budget.max_ooda_cycles,
            max_consecutive_failures=self.config.budget.max_consecutive_failures,
            heartbeat_interval_seconds=self.config.budget.heartbeat_interval,
            claim_lease_seconds=self.config.budget.claim_lease_seconds,
        )

        # Assign VulnKB, LLM, and tools to agents
        self._assign_agent_resources(agent_config)

        # Run all agents concurrently
        tasks = []

        # Start miners
        for miner in self.miners:
            tasks.append(asyncio.create_task(miner.run()))

        # Start observers
        for observer in self.observers:
            tasks.append(asyncio.create_task(observer.run()))

        # Wait for mining to complete or timeout
        # Use a simple polling mechanism: check VulnKB periodically
        # for completion (no pending + no claimed intents)
        max_wait_cycles = self.config.budget.max_ooda_cycles * 2
        poll_interval = self.config.budget.heartbeat_interval

        for cycle in range(max_wait_cycles):
            await asyncio.sleep(poll_interval)

            # Check if mining is complete
            pending = self.kb.get_pending_count()

            # Count claimed intents
            stats = self.kb.get_stats()
            claimed = stats.get("intents_by_status", {}).get("claimed", 0)

            if pending == 0 and claimed == 0:
                self._logger.info(
                    f"  Mining complete after {cycle + 1} poll cycles "
                    f"(no pending or claimed intents)"
                )
                break

            if cycle % 5 == 0:
                self._logger.info(
                    f"  Mining in progress: {pending} pending, "
                    f"{claimed} claimed intents (cycle {cycle + 1})"
                )

        else:
            self._logger.warning(
                f"  Mining timed out after {max_wait_cycles} poll cycles"
            )

        # Cancel all agent tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for clean shutdown
        await asyncio.gather(*tasks, return_exceptions=True)

        # Release expired intents
        released = self.kb.release_expired_intents()
        if released > 0:
            self._logger.info(f"  Released {released} expired intents")

        self._logger.info("Phase 1: Vuln Mining complete.")

    def _create_miners(self) -> None:
        """Create Miner agents based on configuration."""
        specs = self.config.agents.miner_specializations
        count = self.config.agents.miner_count

        # Distribute specializations across miner instances
        for i in range(count):
            spec_name = specs[i % len(specs)]
            try:
                spec = MinerSpec(spec_name)
            except ValueError:
                self._logger.warning(
                    f"  Unknown miner specialization: {spec_name}, defaulting to ATTACK_SURFACE"
                )
                spec = MinerSpec.ATTACK_SURFACE

            miner_id = f"miner_{spec.value}_{i}"
            miner = MinerAgent(
                specialization=spec,
                agent_id=miner_id,
            )
            self.miners.append(miner)

        self._logger.info(f"  Created {len(self.miners)} Miner agents")

    def _create_observers(self) -> None:
        """Create Observer agent based on configuration."""
        observer = ObserverAgent(
            agent_id="observer_main",
            review_every_n_facts=self.config.observer.review_every_n_facts,
            reminder_min_interval=self.config.observer.reminder_min_interval,
        )
        self.observers.append(observer)
        self._logger.info("  Created Observer agent")

    def _assign_agent_resources(self, agent_config: AgentConfig) -> None:
        """Assign VulnKB, LLM, and tools to all agents."""
        all_agents: list[AgentBase] = self.miners + self.observers + self.verifiers

        for agent in all_agents:
            agent.kb = self.kb
            agent.llm = self.llm_gateway
            agent.config = agent_config

            # Assign tools based on agent type and phase
            if self.tool_registry:
                phase = AuditPhase.VULN_MINING if isinstance(agent, MinerAgent) else AuditPhase.CODE_INTELLIGENCE
                if isinstance(agent, ObserverAgent):
                    phase = AuditPhase.VULN_MINING
                tools = self.tool_registry.get_tools_for_phase(phase)
                agent.tools = tools

    # ──────────────────────── Phase 2: Verification ────────────────────────

    async def _run_verification(self) -> None:
        """
        Run verification of vulnerability hypotheses.

        For each VULN_HYPOTHESIS fact in VulnKB, a Verifier agent
        performs independent three-phase verification:
        1. Evidence chain validation
        2. PoC construction
        3. Boundary testing

        Verified hypotheses become VULNERABILITY facts (confirmed) or
        FAILURE_BOUNDARY facts (ruled out).
        """
        self._phase = "verification"
        self._logger.info("Phase 2: Starting Verification...")

        # Get all VULN_HYPOTHESIS facts
        hypotheses = self.kb.get_facts_by_type(FactType.VULN_HYPOTHESIS)
        self._logger.info(f"  Found {len(hypotheses)} vulnerability hypotheses to verify")

        if not hypotheses:
            self._logger.info("  No hypotheses to verify, skipping verification phase")
            return

        # Create Verifier agents
        verifier_count = self.config.agents.verifier_count
        agent_config = AgentConfig(
            max_ooda_cycles=self.config.budget.max_ooda_cycles,
            max_consecutive_failures=self.config.budget.max_consecutive_failures,
            heartbeat_interval_seconds=self.config.budget.heartbeat_interval,
            claim_lease_seconds=self.config.budget.claim_lease_seconds,
        )

        # Set audit phase to verification
        self._phase = "verification"

        for i in range(verifier_count):
            verifier = VerifierAgent(
                agent_id=f"verifier_{i}",
            )
            verifier.kb = self.kb
            verifier.llm = self.llm_gateway
            verifier.config = agent_config

            # Assign verification-phase tools
            if self.tool_registry:
                tools = self.tool_registry.get_tools_for_phase(AuditPhase.VERIFICATION)
                verifier.tools = tools

            self.verifiers.append(verifier)

        self._logger.info(f"  Created {len(self.verifiers)} Verifier agents")

        # Create verification intents for each hypothesis
        for hyp in hypotheses:
            intent = make_intent(
                description=f"Verify vulnerability hypothesis: {hyp.content[:80]}",
                from_facts=[hyp.fact_id],
                priority=min(hyp.confidence, 0.95),
                specialization=None,  # Verifiers don't use MinerSpec
                metadata={
                    "hypothesis_id": hyp.fact_id,
                    "verification_type": "independent",
                },
            )
            self.kb.add_intent(intent)

        # Run verifiers concurrently
        tasks = []
        for verifier in self.verifiers:
            tasks.append(asyncio.create_task(verifier.run()))

        # Wait for verification to complete
        max_wait_cycles = len(hypotheses) + 10  # Generous buffer
        poll_interval = self.config.budget.heartbeat_interval

        for cycle in range(max_wait_cycles):
            await asyncio.sleep(poll_interval)

            # Check if all verification intents are done
            stats = self.kb.get_stats()
            pending = stats.get("intents_by_status", {}).get("pending", 0)
            claimed = stats.get("intents_by_status", {}).get("claimed", 0)

            if pending == 0 and claimed == 0:
                self._logger.info(
                    f"  Verification complete after {cycle + 1} poll cycles"
                )
                break

            if cycle % 5 == 0:
                self._logger.info(
                    f"  Verification in progress: {pending} pending, "
                    f"{claimed} claimed ({cycle + 1}/{max_wait_cycles})"
                )

        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        # Release expired intents
        self.kb.release_expired_intents()

        self._logger.info("Phase 2: Verification complete.")

    # ──────────────────────── Phase 3: Report Generation ────────────────────────

    def _generate_report(self) -> AuditReport:
        """
        Generate the final audit report from VulnKB contents.

        Collects:
        - Confirmed vulnerabilities (VULNERABILITY facts)
        - Ruled-out directions (FAILURE_BOUNDARY facts)
        - Statistics from VulnKB
        - Executive summary and detailed report text
        """
        self._phase = "report"
        self._logger.info("Phase 3: Generating audit report...")

        # Collect confirmed vulnerabilities
        vuln_facts = self.kb.get_facts_by_type(FactType.VULNERABILITY)
        vulnerabilities = []
        for f in vuln_facts:
            entry = VulnerabilityEntry(
                fact_id=f.fact_id,
                vuln_type=f.metadata.get("vuln_type", "unknown"),
                description=f.content,
                evidence=f.evidence,
                confidence=f.confidence,
                severity=f.metadata.get("severity", "high"),
                remediation=f.metadata.get("remediation", ""),
                metadata=f.metadata,
            )
            vulnerabilities.append(entry)

        # Collect failure boundaries (ruled-out directions)
        boundary_facts = self.kb.get_facts_by_type(FactType.FAILURE_BOUNDARY)
        ruled_outs = []
        for f in boundary_facts:
            entry = FailureBoundaryEntry(
                fact_id=f.fact_id,
                vuln_type=f.metadata.get("vuln_type", "unknown"),
                ruled_out=f.metadata.get("ruled_out", f.content),
                remaining_risk=f.metadata.get("remaining_risk", ""),
                confidence=f.confidence,
            )
            ruled_outs.append(entry)

        # Collect statistics
        stats = self.kb.get_stats()

        # Add agent statistics
        for miner in self.miners:
            miner_stats = miner.stats
            stats[f"miner_{miner.agent_id}"] = miner_stats
        for observer in self.observers:
            observer_stats = observer.stats
            stats[f"observer_{observer.agent_id}"] = observer_stats
        for verifier in self.verifiers:
            verifier_stats = verifier.stats
            stats[f"verifier_{verifier.agent_id}"] = verifier_stats

        total_findings = len(vulnerabilities) + len(ruled_outs)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            vulnerabilities, ruled_outs, stats
        )

        # Generate detailed report
        detailed_report = self._generate_detailed_report(
            vulnerabilities, ruled_outs, stats
        )

        report = AuditReport(
            target=self.config.target.repo_path,
            language=self.config.target.language,
            total_findings=total_findings,
            vulnerabilities=vulnerabilities,
            ruled_outs=ruled_outs,
            statistics=stats,
            executive_summary=executive_summary,
            detailed_report=detailed_report,
            started_at=self._started_at,
            completed_at=self._completed_at or datetime.now(timezone.utc).isoformat(),
        )

        self._logger.info(
            f"  Report generated: {total_findings} total findings "
            f"({len(vulnerabilities)} confirmed, {len(ruled_outs)} ruled out)"
        )

        return report

    def _generate_executive_summary(
        self,
        vulnerabilities: list[VulnerabilityEntry],
        ruled_outs: list[FailureBoundaryEntry],
        stats: dict,
    ) -> str:
        """Generate a human-readable executive summary."""
        lines = []
        lines.append(f"VulnGuard Audit Report — {self.config.target.repo_path}")
        lines.append("=" * 60)
        lines.append("")

        # Overall stats
        total_facts = stats.get("total_facts", 0)
        total_intents = stats.get("total_intents", 0)
        lines.append(f"Total facts discovered: {total_facts}")
        lines.append(f"Total intents processed: {total_intents}")
        lines.append("")

        # Vulnerability summary
        lines.append(f"CONFIRMED VULNERABILITIES: {len(vulnerabilities)}")
        if vulnerabilities:
            # Group by severity
            by_severity: dict[str, int] = {}
            for v in vulnerabilities:
                by_severity[v.severity] = by_severity.get(v.severity, 0) + 1
            for sev in ["critical", "high", "medium", "low"]:
                if sev in by_severity:
                    lines.append(f"  {severity_emoji(sev)} {sev.upper()}: {by_severity[sev]}")
        else:
            lines.append("  No confirmed vulnerabilities found.")

        lines.append("")

        # Hypothesis summary
        hyp_count = stats.get("facts_by_type", {}).get("vuln_hypothesis", 0)
        lines.append(f"Unverified hypotheses remaining: {hyp_count}")

        # Ruled-out summary
        lines.append(f"Ruled-out attack directions: {len(ruled_outs)}")

        lines.append("")
        lines.append(f"Audit started: {self._started_at}")
        lines.append(f"Audit completed: {self._completed_at}")

        return "\n".join(lines)

    def _generate_detailed_report(
        self,
        vulnerabilities: list[VulnerabilityEntry],
        ruled_outs: list[FailureBoundaryEntry],
        stats: dict,
    ) -> str:
        """Generate a detailed audit report."""
        lines = []
        lines.append("=" * 60)
        lines.append("DETAILED AUDIT REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Confirmed vulnerabilities
        lines.append("─" * 40)
        lines.append("CONFIRMED VULNERABILITIES")
        lines.append("─" * 40)
        if vulnerabilities:
            for i, v in enumerate(vulnerabilities, 1):
                lines.append(f"")
                lines.append(f"  [{i}] {v.vuln_type.upper()} — {v.severity.upper()}")
                lines.append(f"      ID: {v.fact_id}")
                lines.append(f"      Description: {v.description}")
                lines.append(f"      Confidence: {v.confidence:.2f}")
                if v.evidence:
                    lines.append(f"      Evidence: {v.evidence[:200]}")
                if v.remediation:
                    lines.append(f"      Remediation: {v.remediation}")
        else:
            lines.append("  No confirmed vulnerabilities found.")

        lines.append("")

        # Ruled-out directions
        lines.append("─" * 40)
        lines.append("RULED-OUT ATTACK DIRECTIONS (FAILURE BOUNDARIES)")
        lines.append("─" * 40)
        if ruled_outs:
            for i, r in enumerate(ruled_outs, 1):
                lines.append(f"")
                lines.append(f"  [{i}] {r.vuln_type.upper()}")
                lines.append(f"      Ruled out: {r.ruled_out}")
                if r.remaining_risk:
                    lines.append(f"      Remaining risk: {r.remaining_risk}")
                lines.append(f"      Confidence: {r.confidence:.2f}")
        else:
            lines.append("  No attack directions were ruled out.")

        lines.append("")

        # VulnKB Statistics
        lines.append("─" * 40)
        lines.append("KNOWLEDGE BASE STATISTICS")
        lines.append("─" * 40)
        for key, value in stats.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


# ──────────────────────── Helpers ────────────────────────


def severity_emoji(severity: str) -> str:
    """Return an emoji for a severity level."""
    mapping = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🔵",
        "info": "⚪",
    }
    return mapping.get(severity, "⚪")


def generate_report_from_db(db_path: str, target: str = "", language: str = "") -> AuditReport:
    """
    Generate an audit report from an existing VulnKB database.

    This is useful for the `report` CLI command which reads from
    a persisted database rather than running a fresh audit.
    """
    kb = VulnKB(db_path=db_path)

    # Collect vulnerability facts
    vuln_facts = kb.get_facts_by_type(FactType.VULNERABILITY)
    vulnerabilities = []
    for f in vuln_facts:
        entry = VulnerabilityEntry(
            fact_id=f.fact_id,
            vuln_type=f.metadata.get("vuln_type", "unknown"),
            description=f.content,
            evidence=f.evidence,
            confidence=f.confidence,
            severity=f.metadata.get("severity", "high"),
            remediation=f.metadata.get("remediation", ""),
            metadata=f.metadata,
        )
        vulnerabilities.append(entry)

    # Collect failure boundaries
    boundary_facts = kb.get_facts_by_type(FactType.FAILURE_BOUNDARY)
    ruled_outs = []
    for f in boundary_facts:
        entry = FailureBoundaryEntry(
            fact_id=f.fact_id,
            vuln_type=f.metadata.get("vuln_type", "unknown"),
            ruled_out=f.metadata.get("ruled_out", f.content),
            remaining_risk=f.metadata.get("remaining_risk", ""),
            confidence=f.confidence,
        )
        ruled_outs.append(entry)

    total_findings = len(vulnerabilities) + len(ruled_outs)
    stats = kb.get_stats()

    # Build summary
    lines = []
    lines.append(f"VulnGuard Audit Report (from database: {db_path})")
    lines.append(f"Target: {target or 'N/A'}")
    lines.append(f"Total findings: {total_findings}")
    lines.append(f"Confirmed vulnerabilities: {len(vulnerabilities)}")
    lines.append(f"Ruled-out directions: {len(ruled_outs)}")

    orchestrator_tmp = Orchestrator.__new__(Orchestrator)
    orchestrator_tmp.config = VulnGuardConfig(
        target=TargetConfig(repo_path=target, language=language)
    )
    orchestrator_tmp.kb = kb
    orchestrator_tmp.miners = []
    orchestrator_tmp.observers = []
    orchestrator_tmp.verifiers = []
    orchestrator_tmp._started_at = stats.get("started_at", "")
    orchestrator_tmp._completed_at = stats.get("completed_at", "")

    executive_summary = orchestrator_tmp._generate_executive_summary(
        vulnerabilities, ruled_outs, stats
    )
    detailed_report = orchestrator_tmp._generate_detailed_report(
        vulnerabilities, ruled_outs, stats
    )

    kb.close()

    return AuditReport(
        target=target,
        language=language,
        total_findings=total_findings,
        vulnerabilities=vulnerabilities,
        ruled_outs=ruled_outs,
        statistics=stats,
        executive_summary=executive_summary,
        detailed_report=detailed_report,
        started_at=stats.get("started_at", ""),
        completed_at=stats.get("completed_at", ""),
    )