"""
VulnGuard Intelligence - Module Clustering

Clusters CodeNodes into Modules based on directory structure and token
budgets.  Modules that exceed the token threshold can be further split
or semantically re-clustered via an optional LLM callback.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .parser import CodeNode
from .dependency import DependencyGraph


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Module:
    name: str
    components: List[CodeNode] = field(default_factory=list)
    token_count: int = 0
    sub_modules: List["Module"] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    depth: int = 0

    def total_token_count(self) -> int:
        own = sum(c.token_estimate() for c in self.components)
        child = sum(m.total_token_count() for m in self.sub_modules)
        return own + child

    def all_components(self) -> List[CodeNode]:
        result = list(self.components)
        for sm in self.sub_modules:
            result.extend(sm.all_components())
        return result

    def flatten(self) -> List["Module"]:
        """Return this module and all descendants in depth-first order."""
        result = [self]
        for sm in self.sub_modules:
            result.extend(sm.flatten())
        return result

    def leaf_modules(self) -> List["Module"]:
        if not self.sub_modules:
            return [self]
        result: List[Module] = []
        for sm in self.sub_modules:
            result.extend(sm.leaf_modules())
        return result

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "token_count": self.total_token_count(),
            "component_count": len(self.components) + sum(len(m.components) for m in self.sub_modules),
            "file_paths": self.file_paths,
            "sub_modules": [m.to_dict() for m in self.sub_modules],
            "components": [
                {"node_id": c.node_id, "name": c.name, "type": c.type.value}
                for c in self.components
            ],
        }


@dataclass
class ModuleTree:
    root_modules: List[Module]

    def all_modules(self) -> List[Module]:
        result: List[Module] = []
        for m in self.root_modules:
            result.extend(m.flatten())
        return result

    def leaf_modules(self) -> List[Module]:
        result: List[Module] = []
        for m in self.root_modules:
            result.extend(m.leaf_modules())
        return result

    def to_dict(self) -> dict:
        return {
            "module_count": len(self.all_modules()),
            "leaf_count": len(self.leaf_modules()),
            "modules": [m.to_dict() for m in self.root_modules],
        }


# ---------------------------------------------------------------------------
# LLM clustering callback type
# ---------------------------------------------------------------------------

LLMClusterCallback = Callable[[Module], List[Module]]


def _noop_llm_cluster(module: Module) -> List[Module]:
    """Default: no-op LLM clustering (returns module as-is)."""
    return [module]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_modules(
    graph: DependencyGraph,
    max_tokens: int = 36_000,
    max_depth: int = 3,
    llm_cluster: Optional[LLMClusterCallback] = None,
) -> ModuleTree:
    """Cluster code nodes into a ModuleTree.

    Strategy:
      1. Pre-cluster by directory structure.
      2. Recursively split modules that exceed *max_tokens*.
      3. Optionally use LLM callback for semantic re-clustering.
    """
    if llm_cluster is None:
        llm_cluster = _noop_llm_cluster

    nodes = list(graph.nodes.values())

    # Step 1: cluster by directory
    dir_modules = cluster_by_directory(nodes)

    # Step 2: enforce token budget recursively
    root_modules: List[Module] = []
    for mod in dir_modules:
        rooted = _enforce_budget(mod, max_tokens, max_depth, depth=0, llm_cluster=llm_cluster)
        root_modules.extend(rooted)

    return ModuleTree(root_modules=root_modules)


# ---------------------------------------------------------------------------
# Directory-based clustering
# ---------------------------------------------------------------------------

def cluster_by_directory(nodes: List[CodeNode]) -> List[Module]:
    """Group CodeNodes by their parent directory, producing nested Modules
    that mirror the directory tree."""
    by_dir: Dict[str, List[CodeNode]] = defaultdict(list)
    for node in nodes:
        d = os.path.dirname(node.file_path)
        by_dir[d].append(node)

    # Build nested modules from directory structure
    dir_set = set(by_dir.keys())
    # Find common prefixes to create parent modules
    modules: Dict[str, Module] = {}

    for dir_path, dir_nodes in by_dir.items():
        # Walk up the directory tree to create modules
        parts = dir_path.replace("\\", "/").split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            if prefix not in modules:
                modules[prefix] = Module(
                    name=parts[i - 1] if i > 0 else prefix,
                    file_paths=[],
                    depth=i,
                )

        # Assign nodes to their most specific directory module
        mod = modules[dir_path]
        mod.components = dir_nodes
        mod.file_paths = list({n.file_path for n in dir_nodes})

    # Build hierarchy — attach children to parents
    root_modules: List[Module] = []
    for dir_path, mod in modules.items():
        parts = dir_path.replace("\\", "/").split("/")
        if len(parts) <= 1:
            root_modules.append(mod)
        else:
            parent_path = "/".join(parts[:-1])
            parent = modules.get(parent_path)
            if parent and parent is not mod:
                if mod not in parent.sub_modules:
                    parent.sub_modules.append(mod)

    # If only one root with sub-modules, flatten
    return root_modules if root_modules else list(modules.values())


# ---------------------------------------------------------------------------
# Token budget enforcement
# ---------------------------------------------------------------------------

def cluster_by_tokens(
    module: Module,
    max_tokens: int,
) -> List[Module]:
    """Split a module whose token count exceeds *max_tokens* into smaller ones.

    Strategy: separate each top-level component into its own module if needed,
    or group by sub-directory.
    """
    total = module.total_token_count()
    if total <= max_tokens:
        return [module]

    # If the module has sub-modules, try splitting at the sub-module level
    if module.sub_modules:
        result: List[Module] = []
        current_bucket: List[Module] = []
        current_tokens = 0

        for sm in sorted(module.sub_modules, key=lambda m: m.total_token_count(), reverse=True):
            sm_tokens = sm.total_token_count()
            if current_tokens + sm_tokens > max_tokens and current_bucket:
                # Flush current bucket
                merged = _merge_modules(current_bucket, module.name)
                result.append(merged)
                current_bucket = []
                current_tokens = 0

            # If a single sub-module exceeds the budget, recursively split it
            if sm_tokens > max_tokens:
                result.append(sm)  # will be further split by recursive call
            else:
                current_bucket.append(sm)
                current_tokens += sm_tokens

        if current_bucket:
            merged = _merge_modules(current_bucket, module.name)
            result.append(merged)
        return result

    # No sub-modules — split the flat component list
    if len(module.components) <= 1:
        return [module]  # can't split further

    # Greedy bin packing
    bins: List[List[CodeNode]] = []
    bin_tokens: List[int] = []

    for comp in sorted(module.components, key=lambda c: c.token_estimate(), reverse=True):
        ct = comp.token_estimate()
        placed = False
        for i, bt in enumerate(bin_tokens):
            if bt + ct <= max_tokens:
                bins[i].append(comp)
                bin_tokens[i] += ct
                placed = True
                break
        if not placed:
            bins.append([comp])
            bin_tokens.append(ct)

    result_modules: List[Module] = []
    for i, (bin_nodes, bt) in enumerate(zip(bins, bin_tokens)):
        sub = Module(
            name=f"{module.name}_part{i + 1}",
            components=bin_nodes,
            file_paths=list({n.file_path for n in bin_nodes}),
            depth=module.depth,
        )
        result_modules.append(sub)

    return result_modules


def _merge_modules(modules: List[Module], parent_name: str) -> Module:
    """Merge several modules into one parent module."""
    all_components: List[CodeNode] = []
    all_files: List[str] = []
    sub_modules: List[Module] = []

    for m in modules:
        all_components.extend(m.components)
        all_files.extend(m.file_paths)
        sub_modules.extend(m.sub_modules)

    return Module(
        name=f"{parent_name}_merged",
        components=all_components,
        file_paths=list(set(all_files)),
        sub_modules=sub_modules,
    )


def _enforce_budget(
    module: Module,
    max_tokens: int,
    max_depth: int,
    depth: int,
    llm_cluster: LLMClusterCallback,
) -> List[Module]:
    """Recursively enforce token budget on a module tree."""
    if depth >= max_depth:
        return [module]

    total = module.total_token_count()
    if total <= max_tokens:
        return [module]

    # Try LLM-based semantic clustering first (if provided and module is big enough)
    if total > max_tokens and llm_cluster is not _noop_llm_cluster:
        clustered = llm_cluster(module)
        if len(clustered) > 1:
            result: List[Module] = []
            for sm in clustered:
                result.extend(_enforce_budget(sm, max_tokens, max_depth, depth + 1, llm_cluster))
            return result

    # Token-based splitting
    split = cluster_by_tokens(module, max_tokens)

    result: List[Module] = []
    for sm in split:
        if sm.total_token_count() > max_tokens and depth + 1 < max_depth:
            # Recurse
            # First, recursively split sub-modules
            new_subs: List[Module] = []
            for sub in sm.sub_modules:
                new_subs.extend(_enforce_budget(sub, max_tokens, max_depth, depth + 1, llm_cluster))
            sm.sub_modules = new_subs

            if sm.total_token_count() > max_tokens:
                deeper = cluster_by_tokens(sm, max_tokens)
                for d in deeper:
                    result.extend(
                        _enforce_budget(d, max_tokens, max_depth, depth + 1, llm_cluster)
                    )
            else:
                result.append(sm)
        else:
            result.append(sm)

    return result