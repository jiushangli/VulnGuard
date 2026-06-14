"""
VulnGuard Intelligence - CodeIntelligenceEngine Orchestrator

Top-level orchestrator that runs the full intelligence pipeline:
  parse → build_dependency_graph → cluster_modules →
  extract_api_sequences → detect_vuln_hypotheses

Produces initial_facts and initial_intents for injection into VulnKB.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .parser import CodeNode, parse_repository
from .dependency import DependencyGraph, build_dependency_graph
from .module import Module, ModuleTree, cluster_modules
from .api_sequence import (
    APISequenceGraph,
    VulnHypothesis,
    build_api_sequence_graph,
    detect_vuln_hypotheses,
)


# ---------------------------------------------------------------------------
# Fact & Intent — knowledge-base primitives
# ---------------------------------------------------------------------------

class FactSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FactCategory(str, Enum):
    API_SURFACE = "api_surface"
    AUTH_MECHANISM = "auth_mechanism"
    DATA_MODEL = "data_model"
    DEPENDENCY = "dependency"
    ARCHITECTURE = "architecture"
    VULN_HYPOTHESIS = "vuln_hypothesis"


@dataclass
class Fact:
    """A verified or hypothesized piece of knowledge about the codebase."""
    fact_id: str
    category: FactCategory
    severity: FactSeverity
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    source_nodes: List[str] = field(default_factory=list)  # CodeNode IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "source_nodes": self.source_nodes,
            "metadata": self.metadata,
        }


class IntentType(str, Enum):
    INVESTIGATE = "investigate"
    VERIFY = "verify"
    EXPLORE = "explore"
    DEEP_DIVE = "deep_dive"


@dataclass
class Intent:
    """An actionable investigation directive derived from the intelligence phase."""
    intent_id: str
    type: IntentType
    target: str                    # What to investigate (endpoint, module, etc.)
    rationale: str                 # Why this is worth investigating
    related_facts: List[str] = field(default_factory=list)  # Fact IDs
    related_hypotheses: List[str] = field(default_factory=list)
    priority: int = 5              # 1–10, higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intent_id": self.intent_id,
            "type": self.type.value,
            "target": self.target,
            "rationale": self.rationale,
            "related_facts": self.related_facts,
            "related_hypotheses": self.related_hypotheses,
            "priority": self.priority,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IntelligenceConfig:
    repo_path: str = ""
    max_tokens_per_module: int = 36_000
    max_module_depth: int = 3
    max_workers: int = 4
    enable_api_extraction: bool = True
    enable_vuln_hypotheses: bool = True
    llm_cluster_callback: Any = None  # Optional[LLMClusterCallback]
    exclude_patterns: List[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", "vendor",
        ".venv", "venv", "dist", "build", "target",
    ])


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class IntelligenceResult:
    nodes: List[CodeNode] = field(default_factory=list)
    dependency_graph: Optional[DependencyGraph] = None
    module_tree: Optional[ModuleTree] = None
    api_sequence_graph: Optional[APISequenceGraph] = None
    vuln_hypotheses: List[VulnHypothesis] = field(default_factory=list)
    initial_facts: List[Fact] = field(default_factory=list)
    initial_intents: List[Intent] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_count": len(self.nodes),
            "dependency_graph": self.dependency_graph.to_dict() if self.dependency_graph else None,
            "module_tree": self.module_tree.to_dict() if self.module_tree else None,
            "api_sequence_graph": self.api_sequence_graph.to_dict() if self.api_sequence_graph else None,
            "vuln_hypothesis_count": len(self.vuln_hypotheses),
            "vuln_hypotheses": [h.to_dict() for h in self.vuln_hypotheses],
            "facts": [f.to_dict() for f in self.initial_facts],
            "intents": [i.to_dict() for i in self.initial_intents],
            "statistics": self.statistics,
        }


# ---------------------------------------------------------------------------
# CodeIntelligenceEngine
# ---------------------------------------------------------------------------

class CodeIntelligenceEngine:
    """Main orchestrator for the VulnGuard Code Intelligence pipeline."""

    def __init__(self, config: Optional[IntelligenceConfig] = None):
        self.config = config or IntelligenceConfig()

    def run(self, repo_path: Optional[str] = None, config: Optional[IntelligenceConfig] = None) -> IntelligenceResult:
        """Execute the full intelligence pipeline.

        Pipeline:
          1. Parse source files into CodeNodes
          2. Build dependency graph
          3. Cluster into modules
          4. Extract API sequence graph
          5. Detect vulnerability hypotheses
          6. Generate initial facts and intents
        """
        cfg = config or self.config
        path = repo_path or cfg.repo_path
        if not path:
            raise ValueError("repo_path must be specified in config or as argument")

        result = IntelligenceResult()
        start_time = time.time()

        # Step 1: Parse
        print(f"[Intelligence] Phase 1: Parsing repository at {path}")
        nodes = parse_repository(path)
        result.nodes = nodes
        print(f"[Intelligence]   Parsed {len(nodes)} code nodes")

        if not nodes:
            result.statistics = {"parse_time": time.time() - start_time, "total_time": time.time() - start_time}
            return result

        # Step 2: Build dependency graph
        print("[Intelligence] Phase 2: Building dependency graph")
        dep_start = time.time()
        graph = build_dependency_graph(nodes)
        result.dependency_graph = graph
        print(f"[Intelligence]   Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        # Step 3: Cluster modules
        print("[Intelligence] Phase 3: Clustering modules")
        tree = cluster_modules(
            graph,
            max_tokens=cfg.max_tokens_per_module,
            max_depth=cfg.max_module_depth,
            llm_cluster=cfg.llm_cluster_callback,
        )
        result.module_tree = tree
        print(f"[Intelligence]   Created {len(tree.all_modules())} modules ({len(tree.leaf_modules())} leaves)")

        # Step 4: Extract API sequences
        if cfg.enable_api_extraction:
            print("[Intelligence] Phase 4: Extracting API sequences")
            api_graph = build_api_sequence_graph(nodes)
            result.api_sequence_graph = api_graph
            print(f"[Intelligence]   Found {len(api_graph.endpoints)} API endpoints, {len(api_graph.edges)} edges")

            # Step 5: Detect vuln hypotheses
            if cfg.enable_vuln_hypotheses:
                print("[Intelligence] Phase 5: Detecting vulnerability hypotheses")
                hypotheses = detect_vuln_hypotheses(api_graph)
                result.vuln_hypotheses = hypotheses
                print(f"[Intelligence]   Generated {len(hypotheses)} vulnerability hypotheses")

                # Summarize by type
                type_counts: Dict[str, int] = {}
                for h in hypotheses:
                    type_counts[h.vuln_type.value] = type_counts.get(h.vuln_type.value, 0) + 1
                for vt, count in type_counts.items():
                    print(f"[Intelligence]     {vt}: {count}")

        # Step 6: Generate facts and intents
        print("[Intelligence] Phase 6: Generating facts and intents")
        result.initial_facts = self._generate_facts(result)
        result.initial_intents = self._generate_intents(result)
        print(f"[Intelligence]   Generated {len(result.initial_facts)} facts, {len(result.initial_intents)} intents")

        total_time = time.time() - start_time
        result.statistics = {
            "total_time_seconds": round(total_time, 2),
            "node_count": len(nodes),
            "dependency_edges": len(graph.edges),
            "module_count": len(tree.all_modules()),
            "leaf_module_count": len(tree.leaf_modules()),
            "api_endpoint_count": len(result.api_sequence_graph.endpoints) if result.api_sequence_graph else 0,
            "api_edge_count": len(result.api_sequence_graph.edges) if result.api_sequence_graph else 0,
            "vuln_hypothesis_count": len(result.vuln_hypotheses),
            "fact_count": len(result.initial_facts),
            "intent_count": len(result.initial_intents),
        }

        print(f"[Intelligence] Complete in {total_time:.2f}s")
        return result

    # -----------------------------------------------------------------------
    # Fact generation
    # -----------------------------------------------------------------------

    def _generate_facts(self, result: IntelligenceResult) -> List[Fact]:
        facts: List[Fact] = []
        counter = 0

        # Fact: API surface overview
        if result.api_sequence_graph:
            api_graph = result.api_sequence_graph
            counter += 1
            facts.append(Fact(
                fact_id=f"fact_{counter:04d}",
                category=FactCategory.API_SURFACE,
                severity=FactSeverity.INFO,
                title="API Surface Overview",
                description=f"Detected {len(api_graph.endpoints)} API endpoints across "
                            f"{len(set(ep.file_path for ep in api_graph.endpoints.values()))} files",
                evidence=[
                    f"Endpoints: {', '.join(ep.method.value + ' ' + ep.path for ep in list(api_graph.endpoints.values())[:10])}"
                ],
                metadata={"endpoint_count": len(api_graph.endpoints)},
            ))

            # Fact: Unauthenticated endpoints
            unauth = [ep for ep in api_graph.endpoints.values() if not ep.auth_required]
            if unauth:
                counter += 1
                facts.append(Fact(
                    fact_id=f"fact_{counter:04d}",
                    category=FactCategory.AUTH_MECHANISM,
                    severity=FactSeverity.WARNING,
                    title="Unauthenticated API Endpoints",
                    description=f"Found {len(unauth)} endpoints without authentication requirements",
                    evidence=[f"{ep.method.value} {ep.path}" for ep in unauth[:20]],
                    source_nodes=[ep.endpoint_id for ep in unauth],
                    metadata={"unauth_count": len(unauth)},
                ))

            # Fact: Endpoints by auth mechanism
            auth_mechanisms: Dict[str, int] = {}
            for ep in api_graph.endpoints.values():
                for mw in ep.auth_middleware:
                    auth_mechanisms[mw] = auth_mechanisms.get(mw, 0) + 1
            if auth_mechanisms:
                counter += 1
                facts.append(Fact(
                    fact_id=f"fact_{counter:04d}",
                    category=FactCategory.AUTH_MECHANISM,
                    severity=FactSeverity.INFO,
                    title="Authentication Mechanisms Detected",
                    description=f"Identified {len(auth_mechanisms)} distinct authentication mechanisms",
                    evidence=[f"{k}: {v} endpoint(s)" for k, v in auth_mechanisms.items()],
                    metadata=auth_mechanisms,
                ))

        # Fact: Dependency graph summary
        if result.dependency_graph:
            graph = result.dependency_graph
            type_counts: Dict[str, int] = {}
            for edge in graph.edges:
                type_counts[edge.edge_type.value] = type_counts.get(edge.edge_type.value, 0) + 1
            counter += 1
            facts.append(Fact(
                fact_id=f"fact_{counter:04d}",
                category=FactCategory.DEPENDENCY,
                severity=FactSeverity.INFO,
                title="Dependency Graph Summary",
                description=f"Built dependency graph: {len(graph.nodes)} nodes, "
                            f"{len(graph.edges)} edges",
                evidence=[f"{k}: {v}" for k, v in type_counts.items()],
                metadata={"node_count": len(graph.nodes), "edge_count": len(graph.edges)},
            ))

        # Fact: Module structure
        if result.module_tree:
            tree = result.module_tree
            counter += 1
            facts.append(Fact(
                fact_id=f"fact_{counter:04d}",
                category=FactCategory.ARCHITECTURE,
                severity=FactSeverity.INFO,
                title="Module Structure",
                description=f"Clustered codebase into {len(tree.all_modules())} modules "
                            f"({len(tree.leaf_modules())} leaf modules)",
                evidence=[f"Module: {m.name} ({m.total_token_count()} tokens)" for m in tree.leaf_modules()[:20]],
                metadata={"total_modules": len(tree.all_modules()), "leaf_modules": len(tree.leaf_modules())},
            ))

        # Fact: Vulnerability hypotheses
        for hyp in result.vuln_hypotheses:
            counter += 1
            severity = FactSeverity.WARNING if hyp.confidence >= 0.6 else FactSeverity.INFO
            facts.append(Fact(
                fact_id=f"fact_{counter:04d}",
                category=FactCategory.VULN_HYPOTHESIS,
                severity=severity,
                title=f"Potential {hyp.vuln_type.value.upper()} Vulnerability",
                description=hyp.description,
                evidence=hyp.evidence,
                source_nodes=hyp.endpoint_ids,
                metadata={
                    "vuln_type": hyp.vuln_type.value,
                    "confidence": hyp.confidence,
                    "remediation": hyp.remediation_hint,
                },
            ))

        return facts

    # -----------------------------------------------------------------------
    # Intent generation
    # -----------------------------------------------------------------------

    def _generate_intents(self, result: IntelligenceResult) -> List[Intent]:
        intents: List[Intent] = []
        counter = 0

        # Intent: Investigate each vulnerability hypothesis
        for hyp in result.vuln_hypotheses:
            counter += 1
            intent_type = IntentType.INVESTIGATE
            priority = min(10, max(1, int(hyp.confidence * 10)))

            if hyp.vuln_type.value in ("bola", "bfla"):
                intent_type = IntentType.VERIFY
                priority = min(10, priority + 1)

            intents.append(Intent(
                intent_id=f"intent_{counter:04d}",
                type=intent_type,
                target=", ".join(hyp.endpoint_ids),
                rationale=hyp.description,
                related_hypotheses=[hyp.vuln_type.value],
                priority=priority,
                metadata={
                    "vuln_type": hyp.vuln_type.value,
                    "confidence": hyp.confidence,
                    "endpoints": hyp.endpoint_ids,
                },
            ))

        # Intent: Explore unauthenticated endpoints
        if result.api_sequence_graph:
            unauth = [ep for ep in result.api_sequence_graph.endpoints.values() if not ep.auth_required]
            if unauth:
                counter += 1
                intents.append(Intent(
                    intent_id=f"intent_{counter:04d}",
                    type=IntentType.EXPLORE,
                    target=", ".join(ep.endpoint_id for ep in unauth),
                    rationale=f"Explore {len(unauth)} unauthenticated endpoints for authorization bypass",
                    priority=7,
                    metadata={"endpoint_count": len(unauth)},
                ))

        # Intent: Deep dive on high-centrality modules
        if result.dependency_graph and result.module_tree:
            # Find modules with most incoming/outgoing edges
            node_edge_count: Dict[str, int] = {}
            for edge in result.dependency_graph.edges:
                node_edge_count[edge.target_id] = node_edge_count.get(edge.target_id, 0) + 1
                node_edge_count[edge.source_id] = node_edge_count.get(edge.source_id, 0) + 1

            if node_edge_count and result.module_tree:
                hot_modules = []
                for mod in result.module_tree.leaf_modules():
                    mod_edges = 0
                    for comp in mod.components:
                        mod_edges += node_edge_count.get(comp.node_id, 0)
                    if mod_edges > 5:  # arbitrary threshold
                        hot_modules.append((mod.name, mod_edges))

                if hot_modules:
                    hot_modules.sort(key=lambda x: -x[1])
                    counter += 1
                    intents.append(Intent(
                        intent_id=f"intent_{counter:04d}",
                        type=IntentType.DEEP_DIVE,
                        target=", ".join(name for name, _ in hot_modules[:5]),
                        rationale=f"Deep dive into high-centrality modules that may contain security-relevant logic",
                        priority=6,
                        metadata={"module_centrality": dict(hot_modules[:10])},
                    ))

        # Sort intents by priority (descending)
        intents.sort(key=lambda i: -i.priority)

        return intents