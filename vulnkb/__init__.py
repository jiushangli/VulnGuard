"""
VulnKB package — The shared knowledge graph for vulnerability audit.
"""

from .models import (
    # Enumerations
    FactType, IntentStatus, HintSource, MinerSpec, AuditPhase, RiskLevel,
    VerificationResult,
    # Data classes
    Fact, Intent, Hint, FailureBoundary, VerificationAdmissionResult,
    # Core class
    VulnKB,
    # Factory functions
    make_fact, make_intent, make_hint,
)

__all__ = [
    "FactType", "IntentStatus", "HintSource", "MinerSpec", "AuditPhase",
    "RiskLevel", "VerificationResult",
    "Fact", "Intent", "Hint", "FailureBoundary", "VerificationAdmissionResult",
    "VulnKB",
    "make_fact", "make_intent", "make_hint",
]