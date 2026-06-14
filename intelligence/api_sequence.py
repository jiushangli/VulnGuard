"""
VulnGuard Intelligence - API Sequence Graph Extractor

This is the core innovation of VulnGuard: extracting API endpoint graphs
from code and detecting vulnerability hypotheses through inter-API
dependency analysis.

Supports: Spring (@RequestMapping), Express (router), Flask (@app.route),
Gin (r.GET), FastAPI (@router), and Django (path/url).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .parser import CodeNode, CodeNodeType


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"
    ALL = "ALL"


class APIEdgeType(str, Enum):
    RESOURCE_DEP = "resource_dep"       # Two APIs operate on the same resource
    DATA_DEP = "data_dep"               # One API's output feeds another's input
    STATE_DEP = "state_dep"             # One API modifies state another reads
    CONFLICTS_WITH = "conflicts_with"   # APIs that should not both be accessible


@dataclass
class APIEndpoint:
    endpoint_id: str
    method: HTTPMethod
    path: str
    handler: str                      # qualified name of handler function
    file_path: str = ""
    middleware_chain: List[str] = field(default_factory=list)
    auth_required: bool = False
    auth_middleware: List[str] = field(default_factory=list)
    params: List[str] = field(default_factory=list)
    source_node: Optional[CodeNode] = None
    metadata: Dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.endpoint_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, APIEndpoint):
            return NotImplemented
        return self.endpoint_id == other.endpoint_id

    def to_dict(self) -> dict:
        return {
            "endpoint_id": self.endpoint_id,
            "method": self.method.value,
            "path": self.path,
            "handler": self.handler,
            "file_path": self.file_path,
            "middleware_chain": self.middleware_chain,
            "auth_required": self.auth_required,
            "auth_middleware": self.auth_middleware,
            "params": self.params,
        }


@dataclass
class APIEdge:
    source_id: str
    target_id: str
    edge_type: APIEdgeType
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "description": self.description,
        }


@dataclass
class APISequenceGraph:
    endpoints: Dict[str, APIEndpoint] = field(default_factory=dict)
    edges: List[APIEdge] = field(default_factory=list)
    _adj: Dict[str, List[APIEdge]] = field(default_factory=lambda: defaultdict(list))

    def add_endpoint(self, ep: APIEndpoint) -> None:
        self.endpoints[ep.endpoint_id] = ep

    def add_edge(self, edge: APIEdge) -> None:
        self.edges.append(edge)
        self._adj[edge.source_id].append(edge)

    def endpoints_by_path_prefix(self, prefix: str) -> List[APIEndpoint]:
        return [ep for ep in self.endpoints.values() if ep.path.startswith(prefix)]

    def edges_from(self, endpoint_id: str) -> List[APIEdge]:
        return self._adj.get(endpoint_id, [])

    def edges_to(self, endpoint_id: str) -> List[APIEdge]:
        return [e for e in self.edges if e.target_id == endpoint_id]

    def to_dict(self) -> dict:
        return {
            "endpoints": {eid: ep.to_dict() for eid, ep in self.endpoints.items()},
            "edges": [e.to_dict() for e in self.edges],
            "stats": {
                "endpoint_count": len(self.endpoints),
                "edge_count": len(self.edges),
            },
        }


# ---------------------------------------------------------------------------
# Vulnerability hypotheses
# ---------------------------------------------------------------------------

class VulnType(str, Enum):
    BOLA = "bola"              # Broken Object-Level Authorization
    BFLA = "bfla"              # Broken Function-Level Authorization
    STATE_BYPASS = "state_bypass"   # State machine bypass
    TOCTOU = "toctou"          # Time-of-check / Time-of-use
    MASS_ASSIGNMENT = "mass_assignment"
    IDOR = "idor"              # Insecure Direct Object Reference


@dataclass
class VulnHypothesis:
    vuln_type: VulnType
    endpoint_ids: List[str]
    description: str
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    remediation_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "vuln_type": self.vuln_type.value,
            "endpoint_ids": self.endpoint_ids,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "remediation_hint": self.remediation_hint,
        }


# ---------------------------------------------------------------------------
# Route extractors (per-framework)
# ---------------------------------------------------------------------------

class RouteExtractor:
    """Base class for framework-specific route extraction from CodeNodes."""

    language: str = ""

    @staticmethod
    def _make_ep_id(method: str, path: str) -> str:
        raw = f"{method}:{path}"
        return re.sub(r"[^a-zA-Z0-9]", "_", raw)

    def extract(self, nodes: List[CodeNode]) -> List[APIEndpoint]:
        raise NotImplementedError


class FlaskRouteExtractor(RouteExtractor):
    language = "python"

    _RE_ROUTE = re.compile(
        r"@(\w+)\.route\s*\(\s*['\"]([^'\"]+)['\"]"
        r"(?:\s*,\s*methods\s*=\s*\[([^\]]+)\])?",
    )
    _RE_BLUEPRINT = re.compile(r"@(\w+)\.route")

    def extract(self, nodes: List[CodeNode]) -> List[APIEndpoint]:
        endpoints: List[APIEndpoint] = []
        for node in nodes:
            if node.language != "python":
                continue
            if node.type not in (CodeNodeType.FUNCTION, CodeNodeType.METHOD):
                continue
            for m in self._RE_ROUTE.finditer(node.source):
                app_or_bp = m.group(1)
                path = m.group(2)
                methods_str = m.group(3) or "'GET'"
                methods = [m.strip().strip("'\"") for m in methods_str.split(",")]
                # Check auth decorators
                auth_required, auth_mw = self._check_auth_decorators(node)
                for http_method in methods:
                    ep = APIEndpoint(
                        endpoint_id=self._make_ep_id(http_method, path),
                        method=HTTPMethod(http_method.upper()),
                        path=path,
                        handler=node.name,
                        file_path=node.file_path,
                        middleware_chain=self._get_middleware_chain(node),
                        auth_required=auth_required,
                        auth_middleware=auth_mw,
                        params=self._extract_params(path, node),
                        source_node=node,
                    )
                    endpoints.append(ep)
        return endpoints

    @staticmethod
    def _check_auth_decorators(node: CodeNode) -> Tuple[bool, List[str]]:
        auth_mw: List[str] = []
        for dec in node.decorators:
            dec_lower = dec.lower()
            if any(k in dec_lower for k in ("login_required", "auth", "jwt", "token", "permission")):
                auth_mw.append(dec)
        return len(auth_mw) > 0, auth_mw

    @staticmethod
    def _get_middleware_chain(node: CodeNode) -> List[str]:
        return list(node.decorators)

    @staticmethod
    def _extract_params(path: str, node: CodeNode) -> List[str]:
        params: List[str] = []
        # Path params
        params.extend(re.findall(r"<(\w+)(?::\w+)?>", path))
        # Function params from source (simplified)
        func_match = re.search(r"def\s+\w+\s*\(([^)]*)\)", node.source)
        if func_match:
            param_str = func_match.group(1)
            params.extend(p.strip().split(":")[0].strip() for p in param_str.split(",") if p.strip())
        return params


class SpringRouteExtractor(RouteExtractor):
    language = "java"

    _RE_MAPPING = re.compile(
        r"@(?:Get|Post|Put|Delete|Patch|Request)Mapping\s*\(\s*"
        r'(?:value\s*=\s*|path\s*=\s*)?["\']([^"\']+)["\']',
        re.IGNORECASE,
    )

    def extract(self, nodes: List[CodeNode]) -> List[APIEndpoint]:
        endpoints: List[APIEndpoint] = []
        for node in nodes:
            if node.language != "java":
                continue
            for dec in node.decorators:
                mapping = self._parse_mapping_annotation(dec)
                if mapping:
                    method, path = mapping
                    auth_required, auth_mw = self._check_auth(node)
                    ep = APIEndpoint(
                        endpoint_id=self._make_ep_id(method, path),
                        method=HTTPMethod(method),
                        path=path,
                        handler=node.name,
                        file_path=node.file_path,
                        middleware_chain=list(node.decorators),
                        auth_required=auth_required,
                        auth_middleware=auth_mw,
                        params=self._extract_params(path, node),
                        source_node=node,
                    )
                    endpoints.append(ep)
        return endpoints

    @staticmethod
    def _parse_mapping_annotation(dec: str) -> Optional[Tuple[str, str]]:
        dec_lower = dec.lower()
        if "getmapping" in dec_lower:
            path_match = re.search(r'\(["\']([^"\']+)["\']', dec)
            return ("GET", path_match.group(1)) if path_match else None
        if "postmapping" in dec_lower:
            path_match = re.search(r'\(["\']([^"\']+)["\']', dec)
            return ("POST", path_match.group(1)) if path_match else None
        if "putmapping" in dec_lower:
            path_match = re.search(r'\(["\']([^"\']+)["\']', dec)
            return ("PUT", path_match.group(1)) if path_match else None
        if "deletemapping" in dec_lower:
            path_match = re.search(r'\(["\']([^"\']+)["\']', dec)
            return ("DELETE", path_match.group(1)) if path_match else None
        if "requestmapping" in dec_lower:
            path_match = re.search(r'(?:value|path)\s*=\s*["\']([^"\']+)["\']', dec)
            if not path_match:
                path_match = re.search(r'\(["\']([^"\']+)["\']', dec)
            # Try to extract method
            method_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', dec)
            method = method_match.group(1) if method_match else "ALL"
            path = path_match.group(1) if path_match else "/"
            return (method, path)
        return None

    @staticmethod
    def _check_auth(node: CodeNode) -> Tuple[bool, List[str]]:
        auth_mw: List[str] = []
        for dec in node.decorators:
            d = dec.lower()
            if any(k in d for k in ("preauthorize", "postauthorize", "secured",
                                      "rolesallowed", "withrolean")):
                auth_mw.append(dec)
        return len(auth_mw) > 0, auth_mw

    @staticmethod
    def _extract_params(path: str, node: CodeNode) -> List[str]:
        params: List[str] = []
        params.extend(re.findall(r"\{(\w+)\}", path))
        # Method params from source
        method_match = re.search(r"\(([^)]*)\)", node.source.split("{")[0] if "{" in node.source else node.source)
        if method_match:
            param_str = method_match.group(1)
            for p in param_str.split(","):
                p = p.strip()
                if p and not p.startswith("@"):
                    parts = p.split()
                    if len(parts) >= 2:
                        params.append(parts[-1])
        return params


class ExpressRouteExtractor(RouteExtractor):
    language = "javascript"

    _RE_ROUTE = re.compile(
        r"(?:router|app)\.\s*(get|post|put|delete|patch|all)\s*\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    )

    def extract(self, nodes: List[CodeNode]) -> List[APIEndpoint]:
        endpoints: List[APIEndpoint] = []
        for node in nodes:
            if node.language not in ("javascript",):
                continue
            # Search in all nodes' source for route definitions
            for m in self._RE_ROUTE.finditer(node.source):
                method = m.group(1).upper()
                path = m.group(2)
                auth_required, auth_mw = self._check_auth(node, m.start())
                ep = APIEndpoint(
                    endpoint_id=self._make_ep_id(method, path),
                    method=HTTPMethod(method),
                    path=path,
                    handler=node.name,
                    file_path=node.file_path,
                    auth_required=auth_required,
                    auth_middleware=auth_mw,
                    params=self._extract_params(path),
                    source_node=node,
                )
                endpoints.append(ep)
        return endpoints

    @staticmethod
    def _check_auth(node: CodeNode, route_pos: int) -> Tuple[bool, List[str]]:
        # Look for auth middleware before the handler in the route chain
        source_before = node.source[:route_pos]
        auth_mw: List[str] = []
        if re.search(r"authenticate|jwt|auth|passport|verifyToken", source_before, re.IGNORECASE):
            auth_mw.append("auth_middleware_detected")
        return len(auth_mw) > 0, auth_mw

    @staticmethod
    def _extract_params(path: str) -> List[str]:
        return re.findall(r":(\w+)", path)


class GinRouteExtractor(RouteExtractor):
    language = "go"

    _RE_ROUTE = re.compile(
        r"(\w+)\.\s*(GET|POST|PUT|DELETE|PATCH)\s*\(\s*[\"']([^\"']+)[\"']",
        re.IGNORECASE,
    )

    def extract(self, nodes: List[CodeNode]) -> List[APIEndpoint]:
        endpoints: List[APIEndpoint] = []
        for node in nodes:
            if node.language != "go":
                continue
            for m in self._RE_ROUTE.finditer(node.source):
                group_var = m.group(1)
                method = m.group(2).upper()
                path = m.group(3)
                # Check for auth middleware in the group
                auth_required, auth_mw = self._check_auth(node, m.start())
                ep = APIEndpoint(
                    endpoint_id=self._make_ep_id(method, path),
                    method=HTTPMethod(method),
                    path=path,
                    handler=node.name,
                    file_path=node.file_path,
                    auth_required=auth_required,
                    auth_middleware=auth_mw,
                    params=self._extract_params(path),
                    source_node=node,
                )
                endpoints.append(ep)
        return endpoints

    @staticmethod
    def _check_auth(node: CodeNode, route_pos: int) -> Tuple[bool, List[str]]:
        source_before = node.source[:route_pos]
        auth_mw: List[str] = []
        if re.search(r"AuthMiddleware|JwtAuth|RequireAuth|authMiddleware|authRequired",
                      source_before, re.IGNORECASE):
            auth_mw.append("auth_middleware_detected")
        return len(auth_mw) > 0, auth_mw

    @staticmethod
    def _extract_params(path: str) -> List[str]:
        return re.findall(r":(\w+)", path)


class FastAPIRouteExtractor(RouteExtractor):
    """Extract FastAPI routes from CodeNodes.

    Handles two decorator formats:
    1. Source-level: @router.get("/path")  (from regex-based parsing)
    2. AST repr-level: Attribute(value=Name(id='router'), attr='get', ctx=Load())
       (from Python AST parsing, which stores decorators as ast node repr strings)

    For AST repr, the route path must be extracted from the function source
    since decorators don't contain the path string.
    """
    language = "python"

    # Matches source-level decorator: @router.get("/path")
    _RE_ROUTE_SOURCE = re.compile(
        r"@(\w+)\.\s*(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    )

    # Matches AST repr decorator: Attribute(value=Name(id='router'), attr='get', ...)
    # Captures: router variable name and HTTP method
    _RE_ROUTE_AST = re.compile(
        r"Attribute\(value=Name\(id='(\w+)',\s*ctx=Load\(\)\),\s*attr='(\w+)',\s*ctx=Load\(\)\)",
    )

    # Matches path from source-level decorator in function source text
    _RE_PATH_FROM_SOURCE = re.compile(
        r"@\w+\.\s*(?:get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    )

    # Matches router prefix: router = APIRouter(prefix="/vulnerable", ...)
    _RE_ROUTER_PREFIX = re.compile(
        r"""(?:router|app)\s*=\s*APIRouter\s*\([^)]*prefix\s*=\s*['"]([^'"]+)['"]""",
    )

    def extract(self, nodes: List[CodeNode]) -> List[APIEndpoint]:
        endpoints: List[APIEndpoint] = []

        # First pass: collect router prefixes from all nodes
        router_prefixes: Dict[str, str] = {}
        for node in nodes:
            for m in self._RE_ROUTER_PREFIX.finditer(node.source):
                router_prefixes[m.group(1)] = m.group(2)

        # Second pass: extract routes
        for node in nodes:
            if node.language != "python":
                continue
            if node.type not in (CodeNodeType.FUNCTION, CodeNodeType.METHOD):
                continue

            for dec in node.decorators:
                # Try AST repr format first
                ast_match = self._RE_ROUTE_AST.match(dec)
                if ast_match:
                    router_var = ast_match.group(1)
                    http_method = ast_match.group(2).upper()
                    if http_method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                        continue
                    # Extract path from function source (decorator line)
                    path = self._extract_path_from_source(node) or "/"
                    prefix = router_prefixes.get(router_var, "")
                    full_path = prefix + path if path != "/" else (prefix or "/")
                    auth_required, auth_mw = self._check_auth(node)
                    ep = APIEndpoint(
                        endpoint_id=self._make_ep_id(http_method, full_path),
                        method=HTTPMethod(http_method),
                        path=full_path,
                        handler=node.name,
                        file_path=node.file_path,
                        middleware_chain=list(node.decorators),
                        auth_required=auth_required,
                        auth_middleware=auth_mw,
                        params=self._extract_params(full_path, node),
                        source_node=node,
                    )
                    endpoints.append(ep)
                    continue

                # Try source-level format
                for m in self._RE_ROUTE_SOURCE.finditer(dec):
                    router_var = m.group(1)
                    method = m.group(2).upper()
                    path = m.group(3)
                    prefix = router_prefixes.get(router_var, "")
                    full_path = prefix + path if path != "/" else (prefix or "/")
                    auth_required, auth_mw = self._check_auth(node)
                    ep = APIEndpoint(
                        endpoint_id=self._make_ep_id(method, full_path),
                        method=HTTPMethod(method),
                        path=full_path,
                        handler=node.name,
                        file_path=node.file_path,
                        middleware_chain=list(node.decorators),
                        auth_required=auth_required,
                        auth_middleware=auth_mw,
                        params=self._extract_params(full_path, node),
                        source_node=node,
                    )
                    endpoints.append(ep)

        return endpoints

    @staticmethod
    def _extract_path_from_source(node: CodeNode) -> Optional[str]:
        """Extract the route path from the function's source text.

        Searches for decorator patterns like @router.get("/path") in
        the raw source of the function.
        """
        # Look at lines before the function definition for the route decorator
        lines = node.source.split("\n")
        for line in lines:
            m = re.search(
                r"""@\w+\.\s*(?:get|post|put|delete|patch)\s*\(\s*['"]([^'"]+)['"]""",
                line, re.IGNORECASE,
            )
            if m:
                return m.group(1)
            # Handle multi-line decorators: @router.get(
            #     "/path",
            m2 = re.search(r"""['"](/[^'"]+)['"]""", line)
            if m2 and line.strip().startswith("'") or line.strip().startswith('"'):
                # Only match paths that start with /
                candidate = m2.group(1) if m2 else None
                if candidate and candidate.startswith("/"):
                    return candidate
        return None

    @staticmethod
    def _check_auth(node: CodeNode) -> Tuple[bool, List[str]]:
        auth_mw: List[str] = []
        # Check decorators for auth patterns
        for dec in node.decorators:
            d = dec.lower()
            if any(k in d for k in ("depends", "auth", "jwt", "token", "security")):
                auth_mw.append(dec)
        # Also check function source for auth-related patterns
        source_lower = node.source.lower()
        if any(kw in source_lower for kw in (
            "require_auth", "_require_auth", "current_user", "jwt.decode",
            "verify_token", "check_auth", "authorization",
        )):
            # But only if it's in the function body, not just in a comment
            func_match = re.search(r"def\s+\w+\s*\([^)]*\)\s*:", node.source)
            if func_match:
                body = node.source[func_match.end():]
                if any(kw in body.lower() for kw in (
                    "require_auth", "jwt.decode", "verify_token", "authorization",
                )):
                    auth_mw.append("auth_check_in_body")
        return len(auth_mw) > 0, auth_mw

    @staticmethod
    def _extract_params(path: str, node: CodeNode) -> List[str]:
        params: List[str] = []
        # FastAPI path params: {note_id}
        params.extend(re.findall(r"\{(\w+)\}", path))
        # Function params from source (for query params and request body)
        func_match = re.search(r"def\s+\w+\s*\(([^)]*)\)", node.source)
        if func_match:
            param_str = func_match.group(1)
            for p in param_str.split(","):
                p = p.strip()
                if not p:
                    continue
                # Handle type-annotated params: note_id: int, request: Request
                param_name = p.split(":")[0].strip().split("=")[0].strip()
                if param_name and param_name not in ("self", "cls"):
                    params.append(param_name)
        return params


# ---------------------------------------------------------------------------
# API Sequence Graph builder
# ---------------------------------------------------------------------------

_ALL_EXTRACTORS: List[RouteExtractor] = [
    FlaskRouteExtractor(),
    SpringRouteExtractor(),
    ExpressRouteExtractor(),
    GinRouteExtractor(),
    FastAPIRouteExtractor(),
]


def extract_api_endpoints(nodes: List[CodeNode]) -> List[APIEndpoint]:
    """Extract API endpoints from all supported frameworks."""
    all_endpoints: List[APIEndpoint] = []
    for extractor in _ALL_EXTRACTORS:
        all_endpoints.extend(extractor.extract(nodes))
    return all_endpoints


def build_api_sequence_graph(nodes: List[CodeNode]) -> APISequenceGraph:
    """Build the full API sequence graph from code nodes."""
    graph = APISequenceGraph()

    # Step 1: Extract endpoints
    endpoints = extract_api_endpoints(nodes)
    for ep in endpoints:
        graph.add_endpoint(ep)

    if not endpoints:
        return graph

    # Step 2: Detect inter-API relationships
    _detect_resource_deps(graph)
    _detect_data_deps(graph, nodes)
    _detect_state_deps(graph, nodes)
    _detect_conflicts(graph)

    return graph


# ---------------------------------------------------------------------------
# Inter-API dependency detection
# ---------------------------------------------------------------------------

def _detect_resource_deps(graph: APISequenceGraph) -> None:
    """Two endpoints that share a path parameter are RESOURCE_DEP.

    E.g., GET /users/{id} and DELETE /users/{id} both operate on the
    same user resource.
    """
    # Group by resource pattern (path without method)
    by_resource: Dict[str, List[APIEndpoint]] = defaultdict(list)
    for ep in graph.endpoints.values():
        # Normalize path: replace params with placeholder
        resource = re.sub(r"\{[^}]+\}|:[^/]+|<[^>]+>", "{id}", ep.path)
        # Strip trailing slashes
        resource = resource.rstrip("/")
        by_resource[resource].append(ep)

    for resource, eps in by_resource.items():
        for i, ep_a in enumerate(eps):
            for ep_b in eps[i + 1:]:
                graph.add_edge(APIEdge(
                    source_id=ep_a.endpoint_id,
                    target_id=ep_b.endpoint_id,
                    edge_type=APIEdgeType.RESOURCE_DEP,
                    description=f"Both operate on resource: {resource}",
                ))
                graph.add_edge(APIEdge(
                    source_id=ep_b.endpoint_id,
                    target_id=ep_a.endpoint_id,
                    edge_type=APIEdgeType.RESOURCE_DEP,
                    description=f"Both operate on resource: {resource}",
                ))


def _detect_data_deps(graph: APISequenceGraph, nodes: List[CodeNode]) -> None:
    """Endpoints whose handlers call the same data-access function have
    an implicit DATA_DEP (shared data layer)."""
    # Build handler → called functions map
    handler_calls: Dict[str, Set[str]] = {}
    for ep in graph.endpoints.values():
        sn = ep.source_node
        if sn:
            handler_calls[ep.endpoint_id] = set(sn.calls)
        else:
            # Try to find by name
            for node in nodes:
                if node.name == ep.handler:
                    handler_calls[ep.endpoint_id] = set(node.calls)
                    break

    # Find shared data-access calls between endpoints
    ep_ids = list(handler_calls.keys())
    for i, ep_a_id in enumerate(ep_ids):
        for ep_b_id in ep_ids[i + 1:]:
            shared = handler_calls[ep_a_id] & handler_calls[ep_b_id]
            # Filter out generic/common calls
            data_calls = {c for c in shared
                          if any(kw in c.lower() for kw in
                                 ("find", "get", "query", "fetch", "save",
                                  "create", "update", "delete", "insert",
                                  "select", "dao", "repo", "store"))}
            if data_calls:
                graph.add_edge(APIEdge(
                    source_id=ep_a_id,
                    target_id=ep_b_id,
                    edge_type=APIEdgeType.DATA_DEP,
                    description=f"Shared data layer: {', '.join(sorted(data_calls))}",
                ))
                graph.add_edge(APIEdge(
                    source_id=ep_b_id,
                    target_id=ep_a_id,
                    edge_type=APIEdgeType.DATA_DEP,
                    description=f"Shared data layer: {', '.join(sorted(data_calls))}",
                ))


def _detect_state_deps(graph: APISequenceGraph, nodes: List[CodeNode]) -> None:
    """Endpoints where one modifies state that another reads have STATE_DEP.

    Heuristic: POST/PUT/PATCH on a resource creates a STATE_DEP to GET on
    the same resource pattern.
    """
    by_resource: Dict[str, Dict[str, List[APIEndpoint]]] = defaultdict(lambda: defaultdict(list))
    for ep in graph.endpoints.values():
        resource = re.sub(r"\{[^}]+\}|:[^/]+|<[^>]+>", "{id}", ep.path).rstrip("/")
        by_resource[resource][ep.method.value].append(ep)

    mutators = {HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH, HTTPMethod.DELETE}
    readers = {HTTPMethod.GET}

    for resource, by_method in by_resource.items():
        for mut_method in mutators:
            mut_eps = by_method.get(mut_method.value, [])
            for read_method in readers:
                read_eps = by_method.get(read_method.value, [])
                for mut_ep in mut_eps:
                    for read_ep in read_eps:
                        if mut_ep.endpoint_id != read_ep.endpoint_id:
                            graph.add_edge(APIEdge(
                                source_id=mut_ep.endpoint_id,
                                target_id=read_ep.endpoint_id,
                                edge_type=APIEdgeType.STATE_DEP,
                                description=f"{mut_ep.method.value} modifies state read by {read_ep.method.value} on {resource}",
                            ))


def _detect_conflicts(graph: APISequenceGraph) -> None:
    """Detect endpoints that conflict (admin vs user access on same resource)."""
    for ep_id, ep in graph.endpoints.items():
        path_lower = ep.path.lower()
        # Admin paths vs non-admin paths on similar resources
        is_admin = any(kw in path_lower for kw in ("admin", "manage", "internal", "debug"))
        if not is_admin:
            continue

        # Find related non-admin endpoints
        admin_resource = re.sub(r"\{[^}]+\}|:[^/]+|<[^>]+>", "{id}", path_lower)
        admin_resource = admin_resource.replace("/admin", "").replace("/manage", "")
        admin_resource = admin_resource.rstrip("/")

        for other_id, other_ep in graph.endpoints.items():
            if other_ep.endpoint_id == ep.endpoint_id:
                continue
            other_resource = re.sub(r"\{[^}]+\}|:[^/]+|<[^>]+>", "{id}", other_ep.path.lower()).rstrip("/")
            other_resource = other_resource.replace("/admin", "").replace("/manage", "")

            if admin_resource and other_resource and admin_resource == other_resource:
                if not other_ep.auth_required:
                    graph.add_edge(APIEdge(
                        source_id=ep.endpoint_id,
                        target_id=other_ep.endpoint_id,
                        edge_type=APIEdgeType.CONFLICTS_WITH,
                        description=f"Admin endpoint {ep.path} conflicts with non-admin {other_ep.path} (lacks auth)",
                    ))


# ---------------------------------------------------------------------------
# Vulnerability hypothesis detection
# ---------------------------------------------------------------------------

def detect_vuln_hypotheses(graph: APISequenceGraph) -> List[VulnHypothesis]:
    """Detect vulnerability hypotheses from the API sequence graph.

    Detection strategies:
    - BOLA: RESOURCE_DEP edge to an endpoint lacking object-level auth
    - BFLA: Non-admin endpoints that can reach admin functionality
    - State machine bypass: API sequences with skip-step potential
    - TOCTOU: STATE_DEP between check and use operations
    """
    hypotheses: List[VulnHypothesis] = []

    # -- BOLA Detection ----------------------------------------------------
    _detect_bola(graph, hypotheses)

    # -- BFLA Detection ----------------------------------------------------
    _detect_bfla(graph, hypotheses)

    # -- State Machine Bypass Detection ------------------------------------
    _detect_state_bypass(graph, hypotheses)

    # -- TOCTOU Detection --------------------------------------------------
    _detect_toctou(graph, hypotheses)

    # -- Mass Assignment Detection ------------------------------------------
    _detect_mass_assignment(graph, hypotheses)

    return hypotheses


def _detect_bola(graph: APISequenceGraph, hypotheses: List[VulnHypothesis]) -> None:
    """BOLA: Endpoints that operate on resource IDs but lack object-level auth.

    Indicators:
    - Endpoint has RESOURCE_DEP edges (operates on a resource)
    - Path contains a parameter (e.g., {id}, :id, <id>)
    - No object-level authorization in middleware (only role-based)
    """
    for ep_id, ep in graph.endpoints.items():
        # Check if path has a resource parameter
        has_param = bool(re.search(r"\{[^}]+\}|:[^/]+|<[^>]+>", ep.path))
        if not has_param:
            continue

        # Check if auth is present but only role-based (not object-level)
        has_role_auth = any(
            kw in " ".join(ep.auth_middleware).lower()
            for kw in ("role", "admin", "permission", "secured", "preauthorize")
        )
        has_object_auth = any(
            kw in " ".join(ep.auth_middleware).lower()
            for kw in ("owner", "belong", "own", "resource", "objectlevel")
        )

        if has_param and not has_object_auth:
            # Has parameterized resource access but no object-level auth
            related = graph.edges_from(ep_id)
            resource_deps = [e for e in related if e.edge_type == APIEdgeType.RESOURCE_DEP]

            evidence = [f"Path parameterized: {ep.path}"]
            if ep.auth_required:
                evidence.append("Auth present but only role-based (not object-level)")
            else:
                evidence.append("No authentication required at all")

            hypotheses.append(VulnHypothesis(
                vuln_type=VulnType.BOLA,
                endpoint_ids=[ep_id] + [e.target_id for e in resource_deps[:3]],
                description=f"Potential BOLA on {ep.method.value} {ep.path}: "
                            f"resource-level access without object-level authorization",
                confidence=0.7 if not ep.auth_required else 0.5,
                evidence=evidence,
                remediation_hint="Implement object-level authorization check in handler or middleware",
            ))


def _detect_bfla(graph: APISequenceGraph, hypotheses: List[VulnHypothesis]) -> None:
    """BFLA: Non-admin endpoints that can reach admin functionality.

    Uses CONFLICTS_WITH edges and endpoint analysis.
    """
    for edge in graph.edges:
        if edge.edge_type != APIEdgeType.CONFLICTS_WITH:
            continue

        source_ep = graph.endpoints.get(edge.source_id)
        target_ep = graph.endpoints.get(edge.target_id)
        if not source_ep or not target_ep:
            continue

        hypotheses.append(VulnHypothesis(
            vuln_type=VulnType.BFLA,
            endpoint_ids=[edge.source_id, edge.target_id],
            description=f"Potential BFLA: admin endpoint {source_ep.method.value} {source_ep.path} "
                        f"conflicts with non-admin {target_ep.method.value} {target_ep.path}",
            confidence=0.6,
            evidence=[
                f"Admin endpoint: {source_ep.method.value} {source_ep.path}",
                f"Related non-admin endpoint: {target_ep.method.value} {target_ep.path}",
                f"Non-admin endpoint auth_required={target_ep.auth_required}",
            ],
            remediation_hint="Ensure admin endpoints require elevated role authorization; "
                             "separate admin and user API routes",
        ))

    # Also check: any non-authenticated endpoint that reaches admin-function names
    admin_keywords = {"admin", "manage", "delete_all", "bulk", "internal", "debug", "config"}
    admin_eps = [ep for ep in graph.endpoints.values()
                 if any(kw in ep.path.lower() for kw in admin_keywords)]
    non_auth_eps = [ep for ep in graph.endpoints.values() if not ep.auth_required]

    for aep in admin_eps:
        for nep in non_auth_eps:
            # Check if there's a call path from non-auth to admin
            if aep.endpoint_id == nep.endpoint_id:
                continue
            # Check DATA_DEP
            data_deps = [e for e in graph.edges
                         if e.source_id == nep.endpoint_id
                         and e.target_id == aep.endpoint_id
                         and e.edge_type == APIEdgeType.DATA_DEP]
            if data_deps:
                hypotheses.append(VulnHypothesis(
                    vuln_type=VulnType.BFLA,
                    endpoint_ids=[nep.endpoint_id, aep.endpoint_id],
                    description=f"BFLA: unauthenticated {nep.method.value} {nep.path} "
                                f"shares data layer with admin {aep.method.value} {aep.path}",
                    confidence=0.55,
                    evidence=[
                        f"Unauthenticated: {nep.method.value} {nep.path}",
                        f"Admin: {aep.method.value} {aep.path}",
                        f"Shared data dependency",
                    ],
                    remediation_hint="Add authentication and role-based access control to all admin endpoints",
                ))


def _detect_state_bypass(graph: APISequenceGraph, hypotheses: List[VulnHypothesis]) -> None:
    """State machine bypass: API sequences where steps can be skipped.

    E.g., if a resource has create→update→delete, but delete is accessible
    without going through create, that's a bypass.
    """
    # Group by resource and look for state transitions
    by_resource: Dict[str, List[APIEndpoint]] = defaultdict(list)
    for ep in graph.endpoints.values():
        resource = re.sub(r"\{[^}]+\}|:[^/]+|<[^>]+>", "{id}", ep.path).rstrip("/")
        by_resource[resource].append(ep)

    # Expected CRUD lifecycle: POST (create) → GET (read) → PUT/PATCH (update) → DELETE
    lifecycle = {
        "POST": "create",
        "GET": "read",
        "PUT": "update",
        "PATCH": "update",
        "DELETE": "delete",
    }

    for resource, eps in by_resource.items():
        if len(eps) < 2:
            continue

        # Check if a mutating operation exists but a create (POST) does not
        has_create = any(ep.method == HTTPMethod.POST for ep in eps)
        has_delete = any(ep.method == HTTPMethod.DELETE for ep in eps)
        has_update = any(ep.method in (HTTPMethod.PUT, HTTPMethod.PATCH) for ep in eps)

        # State bypass: can update/delete without creating
        bypass_evidence: List[str] = []

        if has_delete and not has_create:
            bypass_evidence.append("DELETE without POST: resource deletion possible without creation step")

        if has_update and not has_create:
            bypass_evidence.append("PUT/PATCH without POST: resource modification possible without creation step")

        if has_delete and not any(ep.auth_required for ep in eps if ep.method == HTTPMethod.DELETE):
            bypass_evidence.append("DELETE endpoint unauthenticated: state can transition to 'deleted' without auth")

        # Check STATE_DEP chains for missing predecessor steps
        state_deps = [e for e in graph.edges if e.edge_type == APIEdgeType.STATE_DEP]
        resource_state_deps = [e for e in state_deps
                               if e.source_id in {ep.endpoint_id for ep in eps}
                               or e.target_id in {ep.endpoint_id for ep in eps}]

        # If there's a read endpoint after a mutate, but the mutate has no auth
        for ep in eps:
            if ep.method in (HTTPMethod.DELETE, HTTPMethod.PUT, HTTPMethod.PATCH):
                if not ep.auth_required:
                    bypass_evidence.append(
                        f"Unauthenticated state modification: {ep.method.value} {ep.path}"
                    )

        if bypass_evidence:
            hypotheses.append(VulnHypothesis(
                vuln_type=VulnType.STATE_BYPASS,
                endpoint_ids=[ep.endpoint_id for ep in eps],
                description=f"Potential state machine bypass on {resource}: "
                            f"API sequence allows skipping required states",
                confidence=0.45,
                evidence=bypass_evidence,
                remediation_hint="Enforce state machine constraints: validate resource state before allowing mutations",
            ))


def _detect_toctou(graph: APISequenceGraph, hypotheses: List[VulnHypothesis]) -> None:
    """TOCTOU: Time-of-check/time-of-use when STATE_DEP exists between
    a check (GET) and a use (POST/PUT) that operates on the same resource.
    """
    by_resource: Dict[str, List[APIEndpoint]] = defaultdict(list)
    for ep in graph.endpoints.values():
        resource = re.sub(r"\{[^}]+\}|:[^/]+|<[^>]+>", "{id}", ep.path).rstrip("/")
        by_resource[resource].append(ep)

    for resource, eps in by_resource.items():
        gets = [ep for ep in eps if ep.method == HTTPMethod.GET]
        mutators = [ep for ep in eps if ep.method in (HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH)]

        for get_ep in gets:
            for mut_ep in mutators:
                # Check if there's a STATE_DEP edge between them
                state_dep = any(
                    e for e in graph.edges
                    if e.source_id == mut_ep.endpoint_id
                    and e.target_id == get_ep.endpoint_id
                    and e.edge_type == APIEdgeType.STATE_DEP
                )
                if state_dep:
                    # TOCTOU: the state can change between the GET (check) and
                    # the mutation (use)
                    hypotheses.append(VulnHypothesis(
                        vuln_type=VulnType.TOCTOU,
                        endpoint_ids=[get_ep.endpoint_id, mut_ep.endpoint_id],
                        description=f"Potential TOCTOU on {resource}: "
                                    f"state read by {get_ep.method.value} can change before "
                                    f"{mut_ep.method.value} acts",
                        confidence=0.4,
                        evidence=[
                            f"Check: {get_ep.method.value} {get_ep.path}",
                            f"Use: {mut_ep.method.value} {mut_ep.path}",
                            "STATE_DEP relationship between check and use",
                        ],
                        remediation_hint="Use atomic operations or optimistic locking to prevent "
                                         "race conditions between check and use",
                    ))


def _detect_mass_assignment(graph: APISequenceGraph, hypotheses: List[VulnHypothesis]) -> None:
    """Mass Assignment: POST/PUT endpoints that accept parameters without
    clear field whitelisting."""
    for ep in graph.endpoints.values():
        if ep.method not in (HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH):
            continue
        if not ep.source_node:
            continue

        source = ep.source_node.source

        # Heuristic: function body directly uses incoming data fields
        if any(kw in source.lower() for kw in ("request.body", "request.json", "req.body",
                                                  "bodyparser", "@requestbody", "**kwargs")):
            # No explicit field whitelist detected
            if not any(kw in source.lower() for kw in ("whitelist", "allowed_fields", "schema",
                                                          "validation", "serialize", "sanitize")):
                hypotheses.append(VulnHypothesis(
                    vuln_type=VulnType.MASS_ASSIGNMENT,
                    endpoint_ids=[ep.endpoint_id],
                    description=f"Potential mass assignment on {ep.method.value} {ep.path}: "
                                f"accepts request body without visible field whitelisting",
                    confidence=0.35,
                    evidence=[
                        f"Endpoint accepts request body: {ep.method.value} {ep.path}",
                        "No field whitelist/validation detected in source",
                    ],
                    remediation_hint="Implement explicit field whitelisting or DTO validation to "
                                     "prevent over-posting of sensitive fields",
                ))