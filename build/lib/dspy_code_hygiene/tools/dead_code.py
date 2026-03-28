"""Dead code detector. Deterministic AST analysis, no LLM.

Finds:
    - Unreachable code after return/raise/break/continue
    - Unused variables (assigned but never read)
    - Redundant else after return
    - Commented-out code blocks (heuristic)
"""

from __future__ import annotations

import ast
import logging
import re

from dspy_code_hygiene.models import DeadCodeItem

logger = logging.getLogger(__name__)


def find_dead_code(source: str) -> list[DeadCodeItem]:
    """Run all deterministic dead code checks on source.

    Args:
        source: Python source code.

    Returns:
        List of dead code items found.
    """
    items: list[DeadCodeItem] = []
    tree = ast.parse(source)
    lines = source.splitlines()

    items.extend(_find_unreachable(tree))
    items.extend(_find_unused_variables(tree))
    items.extend(_find_redundant_else(tree))
    items.extend(_find_commented_code(lines))

    return sorted(items, key=lambda x: x.line_start)


def _find_unreachable(tree: ast.Module) -> list[DeadCodeItem]:
    """Find code after return/raise/break/continue statements."""
    items = []
    terminal = (ast.Return, ast.Raise, ast.Break, ast.Continue)

    for node in ast.walk(tree):
        if not hasattr(node, "body") or not isinstance(node.body, list):
            continue

        for i, stmt in enumerate(node.body):
            if isinstance(stmt, terminal) and i < len(node.body) - 1:
                next_stmt = node.body[i + 1]
                items.append(
                    DeadCodeItem(
                        kind="unreachable",
                        line_start=next_stmt.lineno,
                        line_end=next_stmt.end_lineno or next_stmt.lineno,
                        name=f"code after {type(stmt).__name__.lower()}",
                        suggestion="remove",
                    )
                )

    return items


def _find_unused_variables(tree: ast.Module) -> list[DeadCodeItem]:
    """Find variables that are assigned but never read.

    Scoped to function bodies. Ignores underscore variables (_),
    __all__, and variables used in augmented assignments.
    """
    items = []

    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        assigned: dict[str, int] = {}  # name -> line number
        read: set[str] = set()

        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned[target.id] = target.lineno
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    assigned[node.target.id] = node.target.lineno
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                read.add(node.id)

        for name, line in assigned.items():
            if name.startswith("_") or name == "__all__":
                continue
            if name not in read:
                items.append(
                    DeadCodeItem(
                        kind="unused_var",
                        line_start=line,
                        line_end=line,
                        name=name,
                        suggestion="remove",
                    )
                )

    return items


def _find_redundant_else(tree: ast.Module) -> list[DeadCodeItem]:
    """Find else blocks after if bodies that always return/raise."""
    items = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not node.orelse:
            continue

        # Check if the if-body always terminates
        body_terminates = (
            node.body
            and isinstance(node.body[-1], (ast.Return, ast.Raise, ast.Break, ast.Continue))
        )

        if body_terminates:
            else_node = node.orelse[0]
            items.append(
                DeadCodeItem(
                    kind="redundant_else",
                    line_start=else_node.lineno,
                    line_end=else_node.end_lineno or else_node.lineno,
                    name="else after return/raise",
                    suggestion="review",
                )
            )

    return items


def _find_commented_code(lines: list[str]) -> list[DeadCodeItem]:
    """Heuristic: find blocks of commented-out code (not regular comments).

    Looks for consecutive comment lines that look like Python code
    (assignments, function calls, imports, control flow).
    """
    items = []
    code_patterns = re.compile(
        r"^\s*#\s*("
        r"import\s|from\s|def\s|class\s|if\s|for\s|while\s|return\s|"
        r"\w+\s*=\s|print\(|raise\s|try:|except\s|with\s|"
        r"\w+\.\w+\(|yield\s"
        r")"
    )

    block_start = None
    block_length = 0

    for i, line in enumerate(lines):
        if code_patterns.match(line):
            if block_start is None:
                block_start = i + 1  # 1-indexed
            block_length += 1
        else:
            if block_start is not None and block_length >= 3:
                items.append(
                    DeadCodeItem(
                        kind="commented_code",
                        line_start=block_start,
                        line_end=block_start + block_length - 1,
                        name=f"{block_length} lines of commented-out code",
                        suggestion="remove",
                    )
                )
            block_start = None
            block_length = 0

    # Handle block at end of file
    if block_start is not None and block_length >= 3:
        items.append(
            DeadCodeItem(
                kind="commented_code",
                line_start=block_start,
                line_end=block_start + block_length - 1,
                name=f"{block_length} lines of commented-out code",
                suggestion="remove",
            )
        )

    return items
