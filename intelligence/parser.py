"""
VulnGuard Intelligence - Multi-Language AST Parser

Parses source code into CodeNode representations for Python, Java,
JavaScript/TypeScript, and Go.  Python uses the stdlib ast module;
all others use regex-based pattern matching.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

class CodeNodeType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"


@dataclass
class CodeNode:
    """Universal representation of a code unit (function / class / method)."""

    node_id: str
    name: str
    type: CodeNodeType
    source: str                    # full source text of the node
    file_path: str = ""
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    language: str = ""
    metadata: Dict = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------

    def token_estimate(self) -> int:
        """Rough token count: ~4 chars per token."""
        return max(1, len(self.source) // 4)

    def qualified_name(self) -> str:
        if self.file_path:
            return f"{self.file_path}::{self.name}"
        return self.name

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodeNode):
            return NotImplemented
        return self.node_id == other.node_id


# ---------------------------------------------------------------------------
# Abstract parser
# ---------------------------------------------------------------------------

class LanguageParser(ABC):
    """Base class for language-specific parsers."""

    file_extensions: Tuple[str, ...] = ()

    @abstractmethod
    def parse_file(self, path: str) -> List[CodeNode]:
        """Parse *path* and return a flat list of CodeNodes."""
        ...

    # -- shared utilities --------------------------------------------------

    @staticmethod
    def _make_node_id(file_path: str, name: str, node_type: str) -> str:
        raw = f"{file_path}:{name}:{node_type}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _read_source(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except (OSError, IOError):
            return ""


# ---------------------------------------------------------------------------
# Python parser  (uses the ast module)
# ---------------------------------------------------------------------------

class PythonParser(LanguageParser):
    file_extensions = (".py",)

    # -- public API --------------------------------------------------------

    def parse_file(self, path: str) -> List[CodeNode]:
        source = self._read_source(path)
        if not source:
            return []
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return []

        file_imports = self._extract_file_imports(tree)
        nodes: List[CodeNode] = []

        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.ClassDef):
                nodes.append(self._parse_class(ast_node, source, path, file_imports))
            elif isinstance(ast_node, ast.FunctionDef) or isinstance(ast_node, ast.AsyncFunctionDef):
                # Skip methods — they are already captured inside classes.
                if not isinstance(ast_node.parent if hasattr(ast_node, "parent") else None, ast.ClassDef):
                    nodes.append(self._parse_function(ast_node, source, path, file_imports))

        # Also extract top-level module-level nodes (no enclosing class)
        self._attach_parents(tree)
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = getattr(ast_node, "_parent", None)
                if isinstance(parent, ast.ClassDef):
                    # method — will be included in the class node's calls
                    pass
        return nodes

    # -- internal ----------------------------------------------------------

    def _attach_parents(self, tree: ast.AST) -> None:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child._parent = parent  # type: ignore[attr-defined]

    def _extract_file_imports(self, tree: ast.AST) -> List[str]:
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                imports.append(mod)
        return imports

    def _parse_class(self, cls: ast.ClassDef, source: str, path: str,
                     file_imports: List[str]) -> CodeNode:
        start = cls.lineno
        end = getattr(cls, "end_lineno", start) or start
        source_text = "\n".join(source.splitlines()[start - 1:end])
        decorators = [self._decorator_text(d, source) for d in cls.decorator_list]
        calls = self._collect_calls(cls)

        return CodeNode(
            node_id=self._make_node_id(path, cls.name, "class"),
            name=cls.name,
            type=CodeNodeType.CLASS,
            source=source_text,
            file_path=path,
            imports=file_imports,
            calls=calls,
            decorators=decorators,
            start_line=start,
            end_line=end,
            language="python",
            metadata={"bases": [b.id if isinstance(b, ast.Name) else ast.dump(b) for b in cls.bases]},
        )

    def _parse_function(self, func, source: str, path: str,
                        file_imports: List[str]) -> CodeNode:
        # Include decorator lines in source range for FastAPI route extraction
        start = func.lineno
        # Find the first decorator line if any
        if func.decorator_list:
            first_dec_line = min(d.lineno for d in func.decorator_list
                                 if hasattr(d, 'lineno') and d.lineno)
            start = min(start, first_dec_line)
        end = getattr(func, "end_lineno", start) or start
        source_text = "\n".join(source.splitlines()[start - 1:end])
        decorators = [self._decorator_text(d, source) for d in func.decorator_list]
        calls = self._collect_calls(func)

        return CodeNode(
            node_id=self._make_node_id(path, func.name, "function"),
            name=func.name,
            type=CodeNodeType.FUNCTION,
            source=source_text,
            file_path=path,
            imports=file_imports,
            calls=calls,
            decorators=decorators,
            start_line=start,
            end_line=end,
            language="python",
        )

    @staticmethod
    def _decorator_text(dec, source: str) -> str:
        """Convert an AST decorator node back to its source text.

        Tries to extract the original source line first for readability.
        Falls back to ast.dump() for complex expressions.
        """
        # Try to extract from source using line numbers
        if hasattr(dec, 'lineno') and dec.lineno:
            source_lines = source.splitlines()
            idx = dec.lineno - 1
            if 0 <= idx < len(source_lines):
                line = source_lines[idx].strip()
                # Only return as source text if it looks like a decorator
                if line.startswith('@'):
                    return line
        # Fallback: reconstruct from AST
        if isinstance(dec, ast.Name):
            return f"@{dec.id}"
        if isinstance(dec, ast.Attribute):
            return ast.dump(dec)
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                return dec.func.id
            if isinstance(dec.func, ast.Attribute):
                # Try to get source line for calls like @router.get("/path")
                if hasattr(dec, 'lineno') and dec.lineno:
                    source_lines = source.splitlines()
                    idx = dec.lineno - 1
                    if 0 <= idx < len(source_lines):
                        line = source_lines[idx].strip()
                        if line.startswith('@'):
                            # May span multiple lines - collect until closing paren
                            collected = []
                            paren_depth = 0
                            for j in range(idx, min(idx + 10, len(source_lines))):
                                collected.append(source_lines[j])
                                paren_depth += source_lines[j].count('(') - source_lines[j].count(')')
                                if paren_depth <= 0 and j > idx:
                                    break
                            result = ' '.join(l.strip() for l in collected)
                            return result
                return ast.dump(dec.func)
        return ast.dump(dec)

    @staticmethod
    def _collect_calls(node: ast.AST) -> List[str]:
        calls: List[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    calls.append(func.id)
                elif isinstance(func, ast.Attribute):
                    calls.append(func.attr)
        return calls


# ---------------------------------------------------------------------------
# Java parser  (regex-based)
# ---------------------------------------------------------------------------

class JavaParser(LanguageParser):
    file_extensions = (".java",)

    # Patterns
    _RE_IMPORT = re.compile(r"^import\s+(?:static\s+)?([^\s;]+);", re.MULTILINE)
    _RE_CLASS = re.compile(
        r"(?:@\w+(?:\([^)]*\))?\s*)*"                             # annotations
        r"(?:public|protected|private)?\s*"                        # visibility
        r"(?:abstract|final|static)?\s*"                           # modifiers
        r"class\s+(\w+)"                                           # class name
        r"(?:\s+extends\s+(\w+))?"                                 # extends
        r"(?:\s+implements\s+([^{]+))?\s*\{",                      # implements
        re.MULTILINE,
    )
    _RE_METHOD = re.compile(
        r"(?:@\w+(?:\([^)]*\))?\s*)*"                              # annotations
        r"(?:public|protected|private)?\s*"                        # visibility
        r"(?:abstract|final|static|synchronized)?\s*"               # modifiers
        r"(?:<[^>]+>\s+)?"                                         # generics
        r"(\w+(?:\[\])?)\s+"                                        # return type
        r"(\w+)\s*"                                                # method name
        r"\(([^)]*)\)\s*"                                           # params
        r"(?:throws\s+[\w\s,]+)?\s*\{",                            # throws + brace
        re.MULTILINE,
    )
    _RE_ANNOTATION = re.compile(r"@(\w+)(?:\([^)]*\))?")

    def parse_file(self, path: str) -> List[CodeNode]:
        source = self._read_source(path)
        if not source:
            return []

        imports = self._RE_IMPORT.findall(source)
        nodes: List[CodeNode] = []

        # Extract classes
        for m in self._RE_CLASS.finditer(source):
            class_name = m.group(1)
            start = source[:m.start()].count("\n") + 1
            end_line = self._find_block_end(source, m.end()) + 1
            class_source = "\n".join(source.splitlines()[start - 1:end_line])
            annotations = self._RE_ANNOTATION.findall(source[:m.start()])
            bases = []
            if m.group(2):
                bases.append(m.group(2))
            if m.group(3):
                bases.extend(b.strip() for b in m.group(3).split(","))

            calls = self._extract_method_calls(class_source)
            nodes.append(CodeNode(
                node_id=self._make_node_id(path, class_name, "class"),
                name=class_name,
                type=CodeNodeType.CLASS,
                source=class_source,
                file_path=path,
                imports=imports,
                calls=calls,
                decorators=annotations,
                start_line=start,
                end_line=end_line,
                language="java",
                metadata={"bases": bases},
            ))

        # Extract methods (only top-level, not inside classes — those are
        # already captured via class nodes)
        # Note: in Java all methods are inside classes; we still extract
        # references from within the class source.
        return nodes

    @staticmethod
    def _find_block_end(source: str, start_pos: int) -> int:
        depth = 0
        i = start_pos
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return source[:i].count("\n")
            i += 1
        return source.count("\n")

    @staticmethod
    def _extract_method_calls(source: str) -> List[str]:
        calls = re.findall(r"\b(\w+)\s*\(", source)
        # Filter out Java keywords and type names
        keywords = {
            "if", "else", "for", "while", "switch", "case", "return",
            "new", "try", "catch", "throw", "class", "import", "package",
            "public", "private", "protected", "void", "this", "super",
        }
        return list({c for c in calls if c not in keywords and not c[0].isupper()})


# ---------------------------------------------------------------------------
# JavaScript / TypeScript parser  (regex-based)
# ---------------------------------------------------------------------------

class JavaScriptParser(LanguageParser):
    file_extensions = (".js", ".jsx", ".ts", ".tsx")

    _RE_IMPORT = re.compile(
        r"(?:import\s+.*?from\s+['\"]([^'\"]+)['\"]|"
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))",
        re.MULTILINE,
    )
    _RE_EXPORT_CLASS = re.compile(
        r"(?:export\s+)?(?:default\s+)?class\s+(\w+)",
    )
    _RE_EXPORT_FUNC = re.compile(
        r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*(\w+)",
    )
    _RE_ARROW = re.compile(
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
    )
    _RE_DECORATOR = re.compile(r"@(\w+)")

    def parse_file(self, path: str) -> List[CodeNode]:
        source = self._read_source(path)
        if not source:
            return []

        imports = [m.group(1) or m.group(2) for m in self._RE_IMPORT.finditer(source)]
        nodes: List[CodeNode] = []
        lines = source.splitlines()

        for m in self._RE_EXPORT_CLASS.finditer(source):
            name = m.group(1)
            start = source[:m.start()].count("\n") + 1
            end = self._find_block_end(source, m.end()) + 1
            block = "\n".join(lines[start - 1:end])
            annotations = self._RE_DECORATOR.findall(source[:m.start()])
            nodes.append(CodeNode(
                node_id=self._make_node_id(path, name, "class"),
                name=name,
                type=CodeNodeType.CLASS,
                source=block,
                file_path=path,
                imports=imports,
                calls=self._extract_calls(block),
                decorators=annotations,
                start_line=start,
                end_line=end,
                language="javascript",
            ))

        for m in self._RE_EXPORT_FUNC.finditer(source):
            name = m.group(1)
            start = source[:m.start()].count("\n") + 1
            end = self._find_block_end(source, m.end()) + 1
            block = "\n".join(lines[start - 1:end])
            nodes.append(CodeNode(
                node_id=self._make_node_id(path, name, "function"),
                name=name,
                type=CodeNodeType.FUNCTION,
                source=block,
                file_path=path,
                imports=imports,
                calls=self._extract_calls(block),
                decorators=[],
                start_line=start,
                end_line=end,
                language="javascript",
            ))

        for m in self._RE_ARROW.finditer(source):
            name = m.group(1)
            start = source[:m.start()].count("\n") + 1
            end = self._find_block_end(source, m.end()) + 1
            block = "\n".join(lines[start - 1:end])
            nodes.append(CodeNode(
                node_id=self._make_node_id(path, name, "function"),
                name=name,
                type=CodeNodeType.FUNCTION,
                source=block,
                file_path=path,
                imports=imports,
                calls=self._extract_calls(block),
                decorators=[],
                start_line=start,
                end_line=end,
                language="javascript",
            ))

        return nodes

    @staticmethod
    def _find_block_end(source: str, start_pos: int) -> int:
        # Skip to first brace
        i = start_pos
        while i < len(source) and source[i] != "{":
            i += 1
        if i >= len(source):
            return source.count("\n")
        depth = 0
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return source[:i].count("\n")
            i += 1
        return source.count("\n")

    @staticmethod
    def _extract_calls(source: str) -> List[str]:
        calls = re.findall(r"\b(\w+)\s*\(", source)
        keywords = {
            "if", "else", "for", "while", "switch", "case", "return",
            "function", "class", "const", "let", "var", "new", "typeof",
            "import", "export", "async", "await", "try", "catch", "throw",
        }
        return list({c for c in calls if c not in keywords})


# ---------------------------------------------------------------------------
# Go parser  (regex-based)
# ---------------------------------------------------------------------------

class GoParser(LanguageParser):
    file_extensions = (".go",)

    _RE_IMPORT = re.compile(r'import\s+(?:\(\s*(.*?)\s*\)|"([^"]+)")', re.DOTALL)
    _RE_IMPORT_LINE = re.compile(r'"([^"]+)"')
    _RE_FUNC = re.compile(r"func\s+(?:\([^)]+\)\s*)?(\w+)\s*\(", re.MULTILINE)
    _RE_TYPE = re.compile(r"type\s+(\w+)\s+struct", re.MULTILINE)
    _RE_INTERFACE = re.compile(r"type\s+(\w+)\s+interface", re.MULTILINE)
    _RE_METHOD = re.compile(r"func\s+\((\w+)\s+\*?(\w+)\)\s*(\w+)\s*\(", re.MULTILINE)

    def parse_file(self, path: str) -> List[CodeNode]:
        source = self._read_source(path)
        if not source:
            return []
        if source.startswith("// +build") or "//go:build" in source.splitlines()[0] if source else False:
            pass  # still parse

        imports = self._extract_imports(source)
        nodes: List[CodeNode] = []
        lines = source.splitlines()

        # Structs
        for m in self._RE_TYPE.finditer(source):
            name = m.group(1)
            start = source[:m.start()].count("\n") + 1
            end = self._find_block_end(source, m.end()) + 1
            block = "\n".join(lines[start - 1:end])
            nodes.append(CodeNode(
                node_id=self._make_node_id(path, name, "class"),
                name=name,
                type=CodeNodeType.CLASS,
                source=block,
                file_path=path,
                imports=imports,
                calls=[],
                decorators=[],
                start_line=start,
                end_line=end,
                language="go",
            ))

        # Interfaces
        for m in self._RE_INTERFACE.finditer(source):
            name = m.group(1)
            start = source[:m.start()].count("\n") + 1
            end = self._find_block_end(source, m.end()) + 1
            block = "\n".join(lines[start - 1:end])
            nodes.append(CodeNode(
                node_id=self._make_node_id(path, name, "class"),
                name=name,
                type=CodeNodeType.CLASS,
                source=block,
                file_path=path,
                imports=imports,
                calls=[],
                decorators=[],
                start_line=start,
                end_line=end,
                language="go",
            ))

        # Methods (receiver functions)
        for m in self._RE_METHOD.finditer(source):
            receiver_name, receiver_type, method_name = m.group(1), m.group(2), m.group(3)
            start = source[:m.start()].count("\n") + 1
            end = self._find_block_end(source, m.end()) + 1
            block = "\n".join(lines[start - 1:end])
            nodes.append(CodeNode(
                node_id=self._make_node_id(path, f"{receiver_type}.{method_name}", "method"),
                name=f"{receiver_type}.{method_name}",
                type=CodeNodeType.METHOD,
                source=block,
                file_path=path,
                imports=imports,
                calls=self._extract_calls(block),
                decorators=[],
                start_line=start,
                end_line=end,
                language="go",
                metadata={"receiver": receiver_type},
            ))

        # Top-level functions
        for m in self._RE_FUNC.finditer(source):
            # Skip methods (they have receiver — already handled above)
            func_sig = source[m.start():m.end()]
            if func_sig.startswith("func ("):
                continue
            name = m.group(1)
            start = source[:m.start()].count("\n") + 1
            end = self._find_block_end(source, m.end()) + 1
            block = "\n".join(lines[start - 1:end])
            nodes.append(CodeNode(
                node_id=self._make_node_id(path, name, "function"),
                name=name,
                type=CodeNodeType.FUNCTION,
                source=block,
                file_path=path,
                imports=imports,
                calls=self._extract_calls(block),
                decorators=[],
                start_line=start,
                end_line=end,
                language="go",
            ))

        return nodes

    def _extract_imports(self, source: str) -> List[str]:
        imports: List[str] = []
        for m in self._RE_IMPORT.finditer(source):
            if m.group(2):
                imports.append(m.group(2))
            elif m.group(1):
                for inner in self._RE_IMPORT_LINE.finditer(m.group(1)):
                    imports.append(inner.group(1))
        return imports

    @staticmethod
    def _find_block_end(source: str, start_pos: int) -> int:
        i = start_pos
        while i < len(source) and source[i] != "{":
            i += 1
        if i >= len(source):
            return source.count("\n")
        depth = 0
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return source[:i].count("\n")
            i += 1
        return source.count("\n")

    @staticmethod
    def _extract_calls(source: str) -> List[str]:
        calls = re.findall(r"\b(\w+)\s*\(", source)
        keywords = {
            "if", "else", "for", "switch", "case", "return", "func",
            "go", "defer", "var", "const", "type", "package", "import",
            "range", "make", "new", "len", "append", "nil",
        }
        return list({c for c in calls if c not in keywords})


# ---------------------------------------------------------------------------
# Registry / dispatcher
# ---------------------------------------------------------------------------

_PARSERS: List[LanguageParser] = [
    PythonParser(),
    JavaParser(),
    JavaScriptParser(),
    GoParser(),
]

_EXTENSION_MAP: Dict[str, LanguageParser] = {}
for _p in _PARSERS:
    for _ext in _p.file_extensions:
        _EXTENSION_MAP[_ext] = _p


def get_parser(file_path: str) -> Optional[LanguageParser]:
    ext = os.path.splitext(file_path)[1].lower()
    return _EXTENSION_MAP.get(ext)


def parse_file(path: str) -> List[CodeNode]:
    """Parse a single file, auto-detecting the language from its extension."""
    parser = get_parser(path)
    if parser is None:
        return []
    return parser.parse_file(path)


def parse_repository(repo_path: str, max_workers: int = 4) -> List[CodeNode]:
    """Walk *repo_path* and parse every recognised source file."""
    all_nodes: List[CodeNode] = []
    supported = set(_EXTENSION_MAP.keys())

    for root, _dirs, files in os.walk(repo_path):
        # Skip hidden / vendor / node_modules / __pycache__
        skip = False
        for part in root.split(os.sep):
            if part in (".git", "__pycache__", "node_modules", ".venv", "venv", "vendor"):
                skip = True
                break
        if skip:
            continue

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue
            fpath = os.path.join(root, fname)
            nodes = parse_file(fpath)
            all_nodes.extend(nodes)

    return all_nodes