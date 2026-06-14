"""
VulnGuard Intelligence - Dependency Graph Builder

Constructs a dependency graph from CodeNode lists, inferring IMPORT,
INHERITANCE, CALL, and DATAFLOW edges between code units.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .parser import CodeNode, CodeNodeType


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

class EdgeType(str, Enum):
    IMPORT = "import"
    INHERITANCE = "inheritance"
    CALL = "call"
    DATAFLOW = "dataflow"


@dataclass
class DependencyEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    description: str = ""
    weight: float = 1.0

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.edge_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DependencyEdge):
            return NotImplemented
        return (self.source_id, self.target_id, self.edge_type) == \
               (other.source_id, other.target_id, other.edge_type)


# ---------------------------------------------------------------------------
# Dependency Graph
# ---------------------------------------------------------------------------

@dataclass
class DependencyGraph:
    nodes: Dict[str, CodeNode] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)

    # Indexes for fast lookup
    _adj_out: Dict[str, List[DependencyEdge]] = field(default_factory=lambda: defaultdict(list))
    _adj_in: Dict[str, List[DependencyEdge]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node: CodeNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: DependencyEdge) -> None:
        if edge not in self.edges:
            self.edges.append(edge)
            self._adj_out[edge.source_id].append(edge)
            self._adj_in[edge.target_id].append(edge)

    def outgoing(self, node_id: str) -> List[DependencyEdge]:
        return self._adj_out.get(node_id, [])

    def incoming(self, node_id: str) -> List[DependencyEdge]:
        return self._adj_in.get(node_id, node_id)

    def successors(self, node_id: str) -> List[str]:
        return [e.target_id for e in self.outgoing(node_id)]

    def predecessors(self, node_id: str) -> List[str]:
        return [e.source_id for e in self.incoming(node_id)]

    # -- aggregation -------------------------------------------------------

    def aggregate_by_file(self) -> Dict[str, List[CodeNode]]:
        by_file: Dict[str, List[CodeNode]] = defaultdict(list)
        for node in self.nodes.values():
            by_file[node.file_path].append(node)
        return dict(by_file)

    def aggregate_by_directory(self) -> Dict[str, List[CodeNode]]:
        by_dir: Dict[str, List[CodeNode]] = defaultdict(list)
        for node in self.nodes.values():
            dir_path = os.path.dirname(node.file_path)
            by_dir[dir_path].append(node)
        return dict(by_dir)

    def aggregate_by_package(self) -> Dict[str, List[CodeNode]]:
        """Group nodes by language-appropriate package/module."""
        by_pkg: Dict[str, List[CodeNode]] = defaultdict(list)
        for node in self.nodes.values():
            pkg = self._infer_package(node)
            by_pkg[pkg].append(node)
        return dict(by_pkg)

    @staticmethod
    def _infer_package(node: CodeNode) -> str:
        dir_path = os.path.dirname(node.file_path)
        lang = node.language
        if lang == "java":
            # Java package from import or path
            parts = dir_path.replace(os.sep, ".").split(".")
            # Convention: src/main/java/com/... → strip prefix
            for i, p in enumerate(parts):
                if p == "java" and i + 1 < len(parts):
                    return ".".join(parts[i + 1:])
            return dir_path
        if lang == "go":
            # Go package = directory
            return dir_path
        if lang in ("python", "javascript"):
            return dir_path
        return dir_path

    def stats(self) -> Dict[str, int]:
        type_counts: Dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            type_counts[n.type.value] += 1
        edge_counts: Dict[str, int] = defaultdict(int)
        for e in self.edges:
            edge_counts[e.edge_type.value] += 1
        return {**type_counts, **edge_counts}

    def to_dict(self) -> dict:
        return {
            "nodes": {nid: _node_to_dict(n) for nid, n in self.nodes.items()},
            "edges": [_edge_to_dict(e) for e in self.edges],
        }


def _node_to_dict(n: CodeNode) -> dict:
    return {
        "node_id": n.node_id,
        "name": n.name,
        "type": n.type.value,
        "source": n.source[:200] + "..." if len(n.source) > 200 else n.source,
        "file_path": n.file_path,
        "imports": n.imports,
        "calls": n.calls,
        "decorators": n.decorators,
        "start_line": n.start_line,
        "end_line": n.end_line,
        "language": n.language,
    }


def _edge_to_dict(e: DependencyEdge) -> dict:
    return {
        "source_id": e.source_id,
        "target_id": e.target_id,
        "edge_type": e.edge_type.value,
        "description": e.description,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_dependency_graph(nodes: List[CodeNode]) -> DependencyGraph:
    """Construct a DependencyGraph from a flat list of CodeNodes.

    The builder infers four kinds of edges:
      - IMPORT:   a node imports a symbol defined by another node
      - INHERITANCE: a class extends / inherits from another class
      - CALL:     a function/method calls another function/method
      - DATAFLOW: a node references a class in its parameters/return type
    """
    graph = DependencyGraph()

    # Index nodes by name and by qualified name for quick lookup
    name_index: Dict[str, List[CodeNode]] = defaultdict(list)
    qname_index: Dict[str, CodeNode] = {}

    for node in nodes:
        graph.add_node(node)
        name_index[node.name].append(node)
        qname_index[node.qualified_name()] = node

    # -- Resolve edges per node --------------------------------------------
    for node in nodes:
        _resolve_import_edges(node, graph, name_index, qname_index)
        _resolve_inheritance_edges(node, graph, name_index, qname_index)
        _resolve_call_edges(node, graph, name_index, qname_index)
        _resolve_dataflow_edges(node, graph, name_index, qname_index)

    return graph


# -- Edge resolvers --------------------------------------------------------

def _resolve_import_edges(
    node: CodeNode,
    graph: DependencyGraph,
    name_index: Dict[str, List[CodeNode]],
    qname_index: Dict[str, CodeNode],
) -> None:
    """Add IMPORT edges from a node's import list to matching nodes."""
    for imp in node.imports:
        # Try direct qualified name match
        if imp in qname_index:
            target = qname_index[imp]
            graph.add_edge(DependencyEdge(
                source_id=node.node_id,
                target_id=target.node_id,
                edge_type=EdgeType.IMPORT,
                description=f"{node.name} imports {target.name}",
            ))
            continue

        # Try loose name match (last segment)
        imp_parts = imp.split(".") if "." in imp else [imp]
        short_name = imp_parts[-1]
        if short_name in name_index:
            for target in name_index[short_name]:
                if target.node_id != node.node_id:
                    graph.add_edge(DependencyEdge(
                        source_id=node.node_id,
                        target_id=target.node_id,
                        edge_type=EdgeType.IMPORT,
                        description=f"{node.name} imports {imp}",
                    ))


def _resolve_inheritance_edges(
    node: CodeNode,
    graph: DependencyGraph,
    name_index: Dict[str, List[CodeNode]],
    qname_index: Dict[str, CodeNode],
) -> None:
    """Add INHERITANCE edges based on class bases / extends."""
    if node.type != CodeNodeType.CLASS:
        return
    bases = node.metadata.get("bases", [])
    for base in bases:
        if base in name_index:
            for target in name_index[base]:
                if target.node_id != node.node_id:
                    graph.add_edge(DependencyEdge(
                        source_id=node.node_id,
                        target_id=target.node_id,
                        edge_type=EdgeType.INHERITANCE,
                        description=f"{node.name} extends {target.name}",
                    ))


def _resolve_call_edges(
    node: CodeNode,
    graph: DependencyGraph,
    name_index: Dict[str, List[CodeNode]],
    qname_index: Dict[str, CodeNode],
) -> None:
    """Add CALL edges from a node's call list to matching nodes."""
    for call_name in node.calls:
        if call_name in name_index:
            for target in name_index[call_name]:
                if target.node_id != node.node_id:
                    graph.add_edge(DependencyEdge(
                        source_id=node.node_id,
                        target_id=target.node_id,
                        edge_type=EdgeType.CALL,
                        description=f"{node.name} calls {target.name}",
                    ))


def _resolve_dataflow_edges(
    node: CodeNode,
    graph: DependencyGraph,
    name_index: Dict[str, List[CodeNode]],
    qname_index: Dict[str, CodeNode],
) -> None:
    """Add DATAFLOW edges when a node references a class in its source
    (heuristic: class names appear as identifiers in function bodies)."""
    if node.type in (CodeNodeType.FUNCTION, CodeNodeType.METHOD):
        for name, targets in name_index.items():
            if name in node.source and len(targets) == 1:
                target = targets[0]
                if target.type == CodeNodeType.CLASS and target.node_id != node.node_id:
                    # Avoid duplicating CALL edges
                    existing = {(e.target_id, e.edge_type) for e in graph.outgoing(node.node_id)}
                    if (target.node_id, EdgeType.DATAFLOW) not in existing:
                        graph.add_edge(DependencyEdge(
                            source_id=node.node_id,
                            target_id=target.node_id,
                            edge_type=EdgeType.DATAFLOW,
                            description=f"{node.name} uses {target.name}",
                        ))