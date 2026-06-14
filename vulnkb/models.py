"""
VulnKB — The shared knowledge graph for vulnerability audit.

Core data model combining:
- Cairn's Fact-Intent DAG (immutable facts, directed exploration intents)
- BreachWeave's bounded dashboard (capacity constraints, failure boundary)
- DELM's verified admission (compress → verify → admit)

Three primitives:
- Fact: immutable, append-only. Once confirmed, never changed.
- Intent: stateful, with lifecycle (pending → claimed → completed/failed).
- Hint: externally injected rule knowledge (OWASP, CWE, user input).

All writes go through verified admission to prevent error propagation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import sqlite3


# ──────────────────────── Enumerations ────────────────────────


class FactType(Enum):
    """Types of facts in the vulnerability knowledge graph."""
    CODE_STRUCTURE = "code_structure"       # Function signatures, class hierarchies, routes
    DATAFLOW = "dataflow"                   # Taint source → sink propagation paths
    API_ENDPOINT = "api_endpoint"          # HTTP endpoint definitions
    SECURITY_CONTROL = "security_control"  # Auth, authorization, input validation
    CONFIGURATION = "configuration"         # Middleware, database, env vars
    BUSINESS_RULE = "business_rule"         # Transaction logic, state machines, permission models
    VULNERABILITY = "vulnerability"         # Confirmed vulnerability
    VULN_HYPOTHESIS = "vuln_hypothesis"     # Unconfirmed vulnerability hypothesis (pending verification)
    FAILURE_BOUNDARY = "failure_boundary"   # Precise boundary of a ruled-out attack direction


class IntentStatus(Enum):
    """Lifecycle states for an exploration intent."""
    PENDING = "pending"      # Waiting to be claimed
    CLAIMED = "claimed"      # Claimed by a miner, in progress
    COMPLETED = "completed"  # Successfully completed, produced facts
    FAILED = "failed"        # Execution failed


class HintSource(Enum):
    """Origin of hint knowledge."""
    RULE_LIBRARY = "rule_library"    # OWASP, CWE, custom rules
    CVE_DATABASE = "cve_database"  # Known vulnerability patterns
    USER_INPUT = "user_input"       # User-provided guidance
    OBSERVER = "observer"           # Observer agent injection


class MinerSpec(Enum):
    """Miner specialization directions."""
    API_SEQUENCE = "api_sequence"      # API call sequence analysis
    DATAFLOW_TAINT = "dataflow_taint"  # Data flow / taint tracking
    BUSINESS_LOGIC = "business_logic"  # Business logic rule extraction
    ATTACK_SURFACE = "attack_surface"  # Attack surface enumeration
    STATE_MACHINE = "state_machine"    # State machine analysis
    AUTH_MODEL = "auth_model"           # Auth/authorization model analysis


class AuditPhase(Enum):
    """Audit phases with different security policies."""
    CODE_INTELLIGENCE = "code_intelligence"  # Phase 0: AST, dependency graphs
    VULN_MINING = "vuln_mining"               # Phase 1: Parallel exploration
    VERIFICATION = "verification"              # Phase 2: PoC construction & validation


class RiskLevel(Enum):
    """Tool risk levels for audit security."""
    SAFE = "safe"              # Pure computation, formatting
    READ_ONLY = "read_only"    # Read-only file/code access
    ANALYSIS = "analysis"      # Static analysis tools
    WRITE = "write"           # Write files (PoC)
    DANGEROUS = "dangerous"    # Execute dynamic tests, send requests


class VerificationResult(Enum):
    """Result of vulnerability verification."""
    CONFIRMED = "confirmed"           # Vulnerability confirmed
    PARTIALLY_CONFIRMED = "partially" # Partially confirmed, needs more evidence
    RULED_OUT = "ruled_out"          # Ruled out with failure boundary
    NEEDS_MORE_INFO = "needs_more"   # Needs additional evidence


# ──────────────────────── Data Classes ────────────────────────


@dataclass(frozen=True)
class Fact:
    """
    Immutable fact — append-only, never modified once created.
    Inspired by Cairn's Fact model, with domain-specific typing.
    Only admitted to VulnKB after verified admission.
    """
    fact_id: str
    fact_type: FactType
    content: str              # Compact gist (≤220 tokens)
    evidence: str             # Detailed evidence (expanded on demand)
    source: str               # Agent ID or tool call that produced this fact
    confidence: float          # [0.0, 1.0]
    timestamp: str            # ISO 8601 datetime
    parent_intents: list[str] = field(default_factory=list)  # Causal: which intents led here
    metadata: dict = field(default_factory=dict)             # Extensible key-value pairs


@dataclass
class Intent:
    """
    Directed exploration intent — an edge from known facts toward new discoveries.
    Borrowed from Cairn's Intent model with state machine lifecycle.
    Agents claim intents from the task queue to work on them.
    """
    intent_id: str
    from_facts: list[str]      # Source facts (supports multi-parent = hyperedge)
    description: str           # What to explore
    priority: float             # [0.0, 1.0]
    status: IntentStatus
    claimed_by: Optional[str]  # Agent ID that claimed this intent
    lease_expiry: Optional[str] # ISO 8601, when claim expires if no heartbeat
    created_at: str            # ISO 8601
    specialization: Optional[MinerSpec] = None  # Preferred miner specialization
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Hint:
    """
    Externally injected strategic knowledge.
    Rules, vulnerability patterns, CVE references, user guidance.
    """
    hint_id: str
    source: HintSource
    pattern: str               # Vulnerability pattern description
    applicability: str         # When this pattern applies
    severity: str               # critical / high / medium / low / info
    references: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class FailureBoundary:
    """
    Precise boundary of a ruled-out attack direction.
    Inspired by BreachWeave's failure memory, but immutable and detailed.

    Instead of "SQL injection test failed", records:
    "Parameterized queries have ruled out SQL injection, but the ORDER BY clause
     still uses string concatenation and the sort direction parameter comes from
     user input, which may allow sort injection."
    """
    vuln_type: str              # Type of vulnerability tested
    ruled_out: str              # What has been ruled out and why
    remaining_risk: str         # What risks still remain and their boundary conditions
    evidence: str               # Code evidence supporting this judgment
    confidence: float           # Confidence in this boundary assessment


@dataclass
class VerificationAdmissionResult:
    """Result of the verified admission gate."""
    admitted: bool
    fact: Optional[Fact] = None
    reason: str = ""


# ──────────────────────── VulnKB ────────────────────────


class VulnKB:
    """
    The vulnerability knowledge graph — the central shared state.

    Key properties:
    - Facts are immutable (append-only)
    - Intents have lifecycle states with lease-based claiming
    - All writes go through verified admission
    - Supports coarse-to-fine context unfolding (gist → summary → raw)
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
        self._verifier = None  # Set later via set_verifier()

    def _init_tables(self):
        """Initialize database tables."""
        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                fact_type TEXT NOT NULL,
                content TEXT NOT NULL,
                evidence TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                parent_intents TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS intents (
                intent_id TEXT PRIMARY KEY,
                from_facts TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL,
                priority REAL NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'pending',
                claimed_by TEXT,
                lease_expiry TEXT,
                created_at TEXT NOT NULL,
                specialization TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS hints (
                hint_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                pattern TEXT NOT NULL,
                applicability TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'medium',
                refs TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(fact_type);
            CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence);
            CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status);
            CREATE INDEX IF NOT EXISTS idx_intents_priority ON intents(priority);
        """)
        self._conn.commit()

    def set_verifier(self, verifier):
        """Set the verified admission verifier (LLM-based)."""
        self._verifier = verifier

    # ─── Fact Operations ───

    def add_fact(self, fact: Fact, verify: bool = True) -> VerificationAdmissionResult:
        """
        Add a fact to the knowledge graph.
        If verify=True, goes through verified admission gate.
        """
        if verify and self._verifier:
            result = self._verifier.verify_admission(fact)
            if not result.admitted:
                return result
            fact = result.fact or fact

        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO facts
            (fact_id, fact_type, content, evidence, source, confidence, timestamp, parent_intents, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact.fact_id, fact.fact_type.value, fact.content, fact.evidence,
            fact.source, fact.confidence, fact.timestamp,
            json.dumps(fact.parent_intents), json.dumps(fact.metadata)
        ))
        self._conn.commit()
        return VerificationAdmissionResult(admitted=True, fact=fact)

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a single fact by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_fact(row)

    def get_facts_by_type(self, fact_type: FactType) -> list[Fact]:
        """Retrieve all facts of a given type."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM facts WHERE fact_type = ? ORDER BY timestamp", (fact_type.value,))
        return [self._row_to_fact(row) for row in cursor.fetchall()]

    def get_all_facts(self) -> list[Fact]:
        """Retrieve all facts."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM facts ORDER BY timestamp")
        return [self._row_to_fact(row) for row in cursor.fetchall()]

    def get_fact_count(self) -> int:
        """Get total number of facts."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM facts")
        return cursor.fetchone()[0]

    def get_facts_since(self, since: str, level: str = "gist") -> list[Fact]:
        """
        Get facts added after a timestamp.
        level: 'gist' returns compact content, 'summary' returns evidence.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM facts WHERE timestamp > ? ORDER BY timestamp", (since,))
        facts = [self._row_to_fact(row) for row in cursor.fetchall()]
        if level == "gist":
            return facts  # content field is already the gist
        return facts  # evidence field is the summary/raw

    # ─── Intent Operations ───

    def add_intent(self, intent: Intent):
        """Add an intent to the task queue."""
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO intents
            (intent_id, from_facts, description, priority, status, claimed_by,
             lease_expiry, created_at, specialization, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            intent.intent_id, json.dumps(intent.from_facts), intent.description,
            intent.priority, intent.status.value, intent.claimed_by,
            intent.lease_expiry, intent.created_at,
            intent.specialization.value if intent.specialization else None,
            json.dumps(intent.metadata)
        ))
        self._conn.commit()

    def claim_intent(self, agent_id: str, specialization: MinerSpec = None,
                     lease_seconds: int = 300) -> Optional[Intent]:
        """
        Claim a pending intent from the task queue.
        Returns None if no matching intent is available.
        """
        cursor = self._conn.cursor()

        # Build query based on specialization
        if specialization:
            cursor.execute("""
                SELECT * FROM intents
                WHERE status = 'pending'
                AND (specialization IS NULL OR specialization = ?)
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """, (specialization.value,))
        else:
            cursor.execute("""
                SELECT * FROM intents
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """)

        row = cursor.fetchone()
        if row is None:
            return None

        intent = self._row_to_intent(row)
        now = datetime.now(timezone.utc)
        from datetime import timedelta
        lease_expiry = (now + timedelta(seconds=lease_seconds)).isoformat()

        cursor.execute("""
            UPDATE intents
            SET status = 'claimed', claimed_by = ?, lease_expiry = ?
            WHERE intent_id = ? AND status = 'pending'
        """, (agent_id, lease_expiry, intent.intent_id))

        if cursor.rowcount == 0:
            return None  # Race condition: another agent claimed it

        self._conn.commit()
        intent.status = IntentStatus.CLAIMED
        intent.claimed_by = agent_id
        intent.lease_expiry = lease_expiry
        return intent

    def complete_intent(self, intent_id: str):
        """Mark an intent as completed."""
        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE intents SET status = 'completed' WHERE intent_id = ?
        """, (intent_id,))
        self._conn.commit()

    def fail_intent(self, intent_id: str):
        """Mark an intent as failed."""
        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE intents SET status = 'failed' WHERE intent_id = ?
        """, (intent_id,))
        self._conn.commit()

    def release_expired_intents(self):
        """Reset claimed intents whose lease has expired."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE intents
            SET status = 'pending', claimed_by = NULL, lease_expiry = NULL
            WHERE status = 'claimed' AND lease_expiry < ?
        """, (now,))
        self._conn.commit()
        return cursor.rowcount

    def heartbeat(self, agent_id: str, lease_seconds: int = 300):
        """Renew the lease for all intents claimed by an agent."""
        from datetime import timedelta
        new_expiry = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE intents SET lease_expiry = ? WHERE claimed_by = ? AND status = 'claimed'
        """, (new_expiry, agent_id))
        self._conn.commit()

    def get_pending_count(self) -> int:
        """Get number of pending intents."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM intents WHERE status = 'pending'")
        return cursor.fetchone()[0]

    def get_pending_intents(self, specialization: MinerSpec = None) -> list[Intent]:
        """Get pending intents, optionally filtered by specialization."""
        cursor = self._conn.cursor()
        if specialization:
            cursor.execute("""
                SELECT * FROM intents
                WHERE status = 'pending'
                AND (specialization IS NULL OR specialization = ?)
                ORDER BY priority DESC, created_at ASC
            """, (specialization.value,))
        else:
            cursor.execute("""
                SELECT * FROM intents WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
            """)
        return [self._row_to_intent(row) for row in cursor.fetchall()]

    # ─── Hint Operations ───

    def add_hint(self, hint: Hint):
        """Add a hint (externally injected strategic knowledge)."""
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO hints
            (hint_id, source, pattern, applicability, severity, refs, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            hint.hint_id, hint.source.value, hint.pattern,
            hint.applicability, hint.severity,
            json.dumps(hint.references), json.dumps(hint.metadata)
        ))
        self._conn.commit()

    def get_all_hints(self) -> list[Hint]:
        """Retrieve all hints."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM hints")
        return [self._row_to_hint(row) for row in cursor.fetchall()]

    def get_hints_by_severity(self, severities: list[str]) -> list[Hint]:
        """Retrieve hints filtered by severity levels."""
        cursor = self._conn.cursor()
        placeholders = ','.join('?' * len(severities))
        cursor.execute(f"""
            SELECT * FROM hints WHERE severity IN ({placeholders})
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 3
                    WHEN 'info' THEN 4
                END
        """, severities)
        return [self._row_to_hint(row) for row in cursor.fetchall()]

    # ─── Context Building ───

    def build_context(self, level: str = "gist", token_budget: int = 4000,
                      fact_types: list[FactType] = None,
                      priority_types: list[FactType] = None) -> str:
        """
        Build context string from VulnKB at the specified compression level.

        Level hierarchy (from DELM):
        - gist: compact summary (≤220 tokens each), always loaded
        - summary: reference-level summary (≤2000 tokens each), loaded on demand
        - raw: full evidence, loaded only when needed

        Within the token budget, facts are prioritized by:
        1. priority_types (if specified)
        2. Fact type order: VULNERABILITY > VULN_HYPOTHESIS > FAILURE_BOUNDARY > ...
        3. Recency (newer facts first)
        """
        cursor = self._conn.cursor()

        type_priority = {
            FactType.VULNERABILITY: 0,
            FactType.VULN_HYPOTHESIS: 1,
            FactType.FAILURE_BOUNDARY: 2,
            FactType.DATAFLOW: 3,
            FactType.SECURITY_CONTROL: 4,
            FactType.BUSINESS_RULE: 5,
            FactType.API_ENDPOINT: 6,
            FactType.CODE_STRUCTURE: 7,
            FactType.CONFIGURATION: 8,
        }

        facts = self.get_all_facts()

        # Filter by fact_types if specified
        if fact_types:
            facts = [f for f in facts if f.fact_type in fact_types]

        # Sort by priority
        priority_set = set(priority_types or [])
        facts.sort(key=lambda f: (
            0 if f.fact_type in priority_set else 1,
            type_priority.get(f.fact_type, 9),
            -f.confidence
        ))

        lines = []
        current_tokens = 0

        for fact in facts:
            if level == "gist":
                text = f"[{fact.fact_type.value}] {fact.content} (conf={fact.confidence:.2f})"
            elif level == "summary":
                text = f"[{fact.fact_type.value}] {fact.evidence}"
            else:  # raw
                text = f"[{fact.fact_type.value}] {fact.evidence}"

            # Rough token estimate (4 chars ≈ 1 token)
            est_tokens = len(text) // 4
            if current_tokens + est_tokens > token_budget:
                lines.append(f"... ({len(facts)} facts total, showing {len(lines)} within budget)")
                break

            lines.append(text)
            current_tokens += est_tokens

        # Add pending intents
        pending_intents = self.get_pending_intents()
        if pending_intents:
            lines.append("\n--- Pending Intents ---")
            for intent in pending_intents[:10]:  # Show top 10
                lines.append(f"[{intent.status.value}] {intent.description} (priority={intent.priority:.2f})")

        # Add hints (compact)
        hints = self.get_all_hints()
        if hints:
            lines.append("\n--- Hints ---")
            for hint in hints[:10]:
                lines.append(f"[{hint.severity}] {hint.pattern}: {hint.applicability}")

        return "\n".join(lines)

    # ─── Statistics ───

    def get_stats(self) -> dict:
        """Get VulnKB statistics."""
        cursor = self._conn.cursor()
        stats = {}

        cursor.execute("SELECT fact_type, COUNT(*) FROM facts GROUP BY fact_type")
        stats["facts_by_type"] = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT status, COUNT(*) FROM intents GROUP BY status")
        stats["intents_by_status"] = {row[0]: row[1] for row in cursor.fetchall()}

        stats["total_facts"] = self.get_fact_count()
        stats["total_intents"] = sum(stats["intents_by_status"].values())
        stats["total_hints"] = len(self.get_all_hints())

        return stats

    # ─── Row Converters ───

    def _row_to_fact(self, row) -> Fact:
        return Fact(
            fact_id=row["fact_id"],
            fact_type=FactType(row["fact_type"]),
            content=row["content"],
            evidence=row["evidence"],
            source=row["source"],
            confidence=row["confidence"],
            timestamp=row["timestamp"],
            parent_intents=json.loads(row["parent_intents"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    def _row_to_intent(self, row) -> Intent:
        return Intent(
            intent_id=row["intent_id"],
            from_facts=json.loads(row["from_facts"]),
            description=row["description"],
            priority=row["priority"],
            status=IntentStatus(row["status"]),
            claimed_by=row["claimed_by"],
            lease_expiry=row["lease_expiry"],
            created_at=row["created_at"],
            specialization=MinerSpec(row["specialization"]) if row["specialization"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    def _row_to_hint(self, row) -> Hint:
        return Hint(
            hint_id=row["hint_id"],
            source=HintSource(row["source"]),
            pattern=row["pattern"],
            applicability=row["applicability"],
            severity=row["severity"],
            references=json.loads(row["refs"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    def close(self):
        """Close the database connection."""
        self._conn.close()


# ──────────────────────── Factory Functions ────────────────────────


def make_fact(fact_type: FactType, content: str, source: str,
              evidence: str = "", confidence: float = 1.0,
              parent_intents: list[str] = None,
              metadata: dict = None) -> Fact:
    """Factory function to create a Fact with auto-generated ID and timestamp."""
    return Fact(
        fact_id=f"fact_{uuid.uuid4().hex[:12]}",
        fact_type=fact_type,
        content=content,
        evidence=evidence or content,
        source=source,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc).isoformat(),
        parent_intents=parent_intents or [],
        metadata=metadata or {}
    )


def make_intent(description: str, from_facts: list[str] = None,
                priority: float = 0.5,
                specialization: MinerSpec = None,
                metadata: dict = None) -> Intent:
    """Factory function to create an Intent with auto-generated ID and timestamp."""
    return Intent(
        intent_id=f"intent_{uuid.uuid4().hex[:12]}",
        from_facts=from_facts or [],
        description=description,
        priority=priority,
        status=IntentStatus.PENDING,
        claimed_by=None,
        lease_expiry=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        specialization=specialization,
        metadata=metadata or {}
    )


def make_hint(pattern: str, applicability: str, severity: str = "medium",
              source: HintSource = HintSource.RULE_LIBRARY,
              references: list[str] = None,
              metadata: dict = None) -> Hint:
    """Factory function to create a Hint."""
    return Hint(
        hint_id=f"hint_{uuid.uuid4().hex[:12]}",
        source=source,
        pattern=pattern,
        applicability=applicability,
        severity=severity,
        references=references or [],
        metadata=metadata or {}
    )