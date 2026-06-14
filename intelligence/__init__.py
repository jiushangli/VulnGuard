"""
VulnGuard Intelligence - Code Intelligence Engine (Phase 0)

This package provides multi-language AST parsing, dependency analysis,
module clustering, and API sequence graph extraction with automatic
vulnerability hypothesis detection.
"""

from .parser import (
    CodeNode,
    CodeNodeType,
    LanguageParser,
    PythonParser,
    JavaParser,
    JavaScriptParser,
    GoParser,
    parse_file,
    parse_repository,
    get_parser,
)

from .dependency import (
    DependencyGraph,
    DependencyEdge,
    EdgeType,
    build_dependency_graph,
)

from .module import (
    Module,
    ModuleTree,
    cluster_modules,
    cluster_by_directory,
    cluster_by_tokens,
)

from .api_sequence import (
    APIEndpoint,
    APIEdge,
    APIEdgeType,
    APISequenceGraph,
    VulnHypothesis,
    VulnType,
    build_api_sequence_graph,
    detect_vuln_hypotheses,
    extract_api_endpoints,
    FlaskRouteExtractor,
    SpringRouteExtractor,
    ExpressRouteExtractor,
    GinRouteExtractor,
    FastAPIRouteExtractor,
)

from .engine import (
    CodeIntelligenceEngine,
    IntelligenceConfig,
    IntelligenceResult,
    Fact,
    FactCategory,
    FactSeverity,
    Intent,
    IntentType,
)

__all__ = [
    # Parser
    "CodeNode",
    "CodeNodeType",
    "LanguageParser",
    "PythonParser",
    "JavaParser",
    "JavaScriptParser",
    "GoParser",
    "parse_file",
    "parse_repository",
    "get_parser",
    # Dependency
    "DependencyGraph",
    "DependencyEdge",
    "EdgeType",
    "build_dependency_graph",
    # Module
    "Module",
    "ModuleTree",
    "cluster_modules",
    "cluster_by_directory",
    "cluster_by_tokens",
    # API Sequence
    "APIEndpoint",
    "APIEdge",
    "APIEdgeType",
    "APISequenceGraph",
    "VulnHypothesis",
    "VulnType",
    "build_api_sequence_graph",
    "detect_vuln_hypotheses",
    "extract_api_endpoints",
    "FlaskRouteExtractor",
    "SpringRouteExtractor",
    "ExpressRouteExtractor",
    "GinRouteExtractor",
    "FastAPIRouteExtractor",
    # Engine
    "CodeIntelligenceEngine",
    "IntelligenceConfig",
    "IntelligenceResult",
    "Fact",
    "FactCategory",
    "FactSeverity",
    "Intent",
    "IntentType",
]

__version__ = "0.1.0"

# Quick self-test (run with: python -m vulnguard.intelligence)
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        engine = CodeIntelligenceEngine(IntelligenceConfig(repo_path=path))
        result = engine.run()
        print(result.to_dict())
    else:
        print("Usage: python -m vulnguard.intelligence <repo_path>")