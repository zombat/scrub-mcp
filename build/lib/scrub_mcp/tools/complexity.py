"""Complexity analyzer. Deterministic metrics, DSPy for suggestions.

Deterministic:
    - Cyclomatic complexity (branches + loops)
    - Cognitive complexity (nesting depth weighted)
    - Function length
    - Nesting depth
    - Parameter count

DSPy (local LLM):
    - Suggest concrete simplifications for flagged functions
"""

from __future__ import annotations

import ast
import logging

from scrub_mcp.models import ComplexityReport, FunctionComplexity, FunctionInfo

logger = logging.getLogger(__name__)


def analyze_complexity(func: FunctionInfo) -> FunctionComplexity:
    """Compute complexity metrics for a single function. Pure AST, no LLM.

    Args:
        func: Extracted function metadata.

    Returns:
        FunctionComplexity with all deterministic metrics.

    """
    try:
        tree = ast.parse(func.body)
    except SyntaxError:
        # Body might not parse standalone (indentation issues)
        # Wrap in a function to make it parseable
        wrapped = "def _wrapper():\n" + "\n".join(f"    {line}" for line in func.body.splitlines())
        try:
            tree = ast.parse(wrapped)
        except SyntaxError:
            return FunctionComplexity(
                function_name=func.name,
                parent_class=func.parent_class,
            )

    cyclomatic = _cyclomatic_complexity(tree)
    cognitive = _cognitive_complexity(tree)
    max_depth = _max_nesting_depth(tree)
    param_count = _count_params(func.signature)

    return FunctionComplexity(
        function_name=func.name,
        parent_class=func.parent_class,
        cyclomatic_complexity=cyclomatic,
        cognitive_complexity=cognitive,
        line_count=func.body_line_count,
        max_nesting_depth=max_depth,
        parameter_count=param_count,
        needs_simplification=_should_flag(cyclomatic, cognitive, func.body_line_count, max_depth),
    )


def analyze_file_complexity(
    functions: list[FunctionInfo],
    thresholds: dict[str, int] | None = None,
) -> ComplexityReport:
    """Analyze all functions in a file and produce a complexity report.

    Args:
        functions: List of extracted function metadata.
        thresholds: Override default thresholds.

    Returns:
        ComplexityReport with per-function metrics and file-level summary.

    """
    t = thresholds or {
        "cyclomatic": 10,
        "cognitive": 15,
        "line_count": 50,
        "nesting": 4,
        "params": 5,
    }

    results = []
    for func in functions:
        metrics = analyze_complexity(func)
        results.append(metrics)

    flagged = [r for r in results if r.needs_simplification]

    return ComplexityReport(
        functions=results,
        total_functions=len(results),
        flagged_count=len(flagged),
        avg_cyclomatic=sum(r.cyclomatic_complexity for r in results) / max(len(results), 1),
        avg_cognitive=sum(r.cognitive_complexity for r in results) / max(len(results), 1),
        hotspots=[
            r.function_name
            for r in sorted(flagged, key=lambda r: r.cyclomatic_complexity, reverse=True)[:5]
        ],
    )


def _cyclomatic_complexity(tree: ast.AST) -> int:
    """McCabe cyclomatic complexity."""
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            complexity += 1
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            complexity += 1 + len(node.ifs)
    return complexity


def _cognitive_complexity(tree: ast.AST, depth: int = 0) -> int:
    """Cognitive complexity: penalizes nesting more heavily than McCabe.

    Each nesting level adds a weight to the increment, so deeply nested
    branches cost more than flat ones.
    """
    total = 0

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            total += 1 + depth  # +1 for the branch, +depth for nesting
            total += _cognitive_complexity(node, depth + 1)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            total += 1 + depth
            total += _cognitive_complexity(node, depth + 1)
        elif isinstance(node, ast.ExceptHandler):
            total += 1 + depth
            total += _cognitive_complexity(node, depth + 1)
        elif isinstance(node, ast.BoolOp):
            total += len(node.values) - 1
            total += _cognitive_complexity(node, depth)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Nested function: reset depth but add 1
            total += 1
            total += _cognitive_complexity(node, 0)
        elif isinstance(node, ast.Lambda):
            total += 1 + depth
        else:
            total += _cognitive_complexity(node, depth)

    return total


def _max_nesting_depth(tree: ast.AST, depth: int = 0) -> int:
    """Find the maximum nesting depth in the AST."""
    max_depth = depth

    for node in ast.iter_child_nodes(tree):
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.AsyncFor,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.ExceptHandler,
            ),
        ):
            child_depth = _max_nesting_depth(node, depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _max_nesting_depth(node, depth)
            max_depth = max(max_depth, child_depth)

    return max_depth


def _count_params(signature: str) -> int:
    """Count parameters in a function signature, excluding self/cls."""
    try:
        params_str = signature.split("(", 1)[1].rstrip(")")
    except IndexError:
        return 0

    params = [
        p.strip().split(":")[0].split("=")[0].strip()
        for p in params_str.split(",")
        if p.strip() and p.strip() not in ("self", "cls")
    ]
    return len(params)


def _should_flag(cyclomatic: int, cognitive: int, lines: int, depth: int) -> bool:
    """Determine if a function should be flagged for simplification.

    Uses a weighted score: any single extreme value or a combination
    of moderate values triggers flagging.
    """
    if cyclomatic >= 10:
        return True
    if cognitive >= 15:
        return True
    if lines >= 50:
        return True
    if depth >= 4:
        return True
    # Combined moderate complexity
    score = (cyclomatic / 10) + (cognitive / 15) + (lines / 50) + (depth / 4)
    return score >= 2.0
