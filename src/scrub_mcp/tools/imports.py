"""Import optimizer. Deterministic-first, DSPy for missing import inference.

Tier 1 (deterministic):
    - Ruff I (isort-compatible sorting)
    - Ruff F401 (unused imports)
    - autoflake (unused imports + variables)

Tier 2 (DSPy, local LLM):
    - Infer missing imports from usage patterns
"""

from __future__ import annotations

import ast
import logging
import subprocess
import tempfile
from pathlib import Path

from scrub_mcp.models import ImportOptResult

logger = logging.getLogger(__name__)


def analyze_imports_deterministic(source: str) -> ImportOptResult:
    """Run deterministic import analysis: find unused, sort, detect missing.

    Uses AST to find all imported names and all used names, then computes
    the diff. This catches most cases without an LLM.

    Args:
        source: Python source code.

    Returns:
        ImportOptResult with unused, missing (AST-inferred), and sorted source.
    """
    tree = ast.parse(source)

    # Collect all imported names
    imported_names: dict[str, str] = {}  # name -> full import statement
    import_nodes: list[ast.AST] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_nodes.append(node)
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                imported_names[name] = ast.unparse(node)
        elif isinstance(node, ast.ImportFrom):
            import_nodes.append(node)
            for alias in node.names:
                name = alias.asname or alias.name
                imported_names[name] = ast.unparse(node)

    # Collect all used names (excluding import statements themselves)
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not _is_import_context(node, import_nodes):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Catch module.function usage
            root = _get_attribute_root(node)
            if root:
                used_names.add(root)

    # Unused: imported but never referenced
    unused = {
        name: stmt for name, stmt in imported_names.items()
        if name not in used_names and name != "__future__"
    }

    # Missing: used but not imported and not a builtin/local definition
    builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    local_defs = _get_local_definitions(tree)
    potentially_missing = used_names - set(imported_names.keys()) - builtins - local_defs

    return ImportOptResult(
        unused_imports=list(unused.values()),
        potentially_missing=sorted(potentially_missing),
        import_count=len(imported_names),
    )


def fix_imports_ruff(source: str) -> tuple[str, int]:
    """Run Ruff to remove unused imports and sort.

    Returns (fixed_source, count_of_fixes).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        # Remove unused imports (F401) and sort (I)
        result = subprocess.run(
            [
                "ruff", "check", "--fix",
                "--select", "F401,I",
                "--line-length", "100",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        fixed = tmp_path.read_text(encoding="utf-8")

        # Count fixes from output
        fix_count = sum(
            1 for line in result.stdout.splitlines()
            if "F401" in line or "I001" in line
        )

        return fixed, fix_count

    finally:
        tmp_path.unlink(missing_ok=True)


def _is_import_context(node: ast.Name, import_nodes: list[ast.AST]) -> bool:
    """Check if a Name node is part of an import statement."""
    for imp in import_nodes:
        if hasattr(imp, 'lineno') and hasattr(node, 'lineno'):
            if imp.lineno == node.lineno:
                return True
    return False


def _get_attribute_root(node: ast.Attribute) -> str | None:
    """Walk an attribute chain to find the root name (e.g., os in os.path.join)."""
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _get_local_definitions(tree: ast.Module) -> set[str]:
    """Get all locally defined names (functions, classes, variables)."""
    defs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defs.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defs.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if isinstance(node.target, ast.Name):
                defs.add(node.target.id)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    defs.add(item.optional_vars.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defs.add(node.name)
        elif isinstance(node, ast.comprehension):
            if isinstance(node.target, ast.Name):
                defs.add(node.target.id)
    return defs
