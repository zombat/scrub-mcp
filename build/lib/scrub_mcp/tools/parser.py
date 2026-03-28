"""AST-based Python source parser for extracting function, class, and module metadata."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from scrub_mcp.models import ClassInfo, FunctionInfo, ModuleInfo


def _cyclomatic_complexity(node: ast.AST) -> int:
    """Rough cyclomatic complexity: count branches and loops."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.IfExp)):
            complexity += 1
        elif isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, (ast.And, ast.Or)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            complexity += 1 + len(child.ifs)
    return complexity


def extract_module_info(source: str) -> ModuleInfo:
    """Extract module-level metadata."""
    tree = ast.parse(source)
    existing_doc = ast.get_docstring(tree)
    return ModuleInfo(
        existing_docstring=existing_doc,
        imports=[ast.unparse(node) for node in ast.iter_child_nodes(tree)
                 if isinstance(node, (ast.Import, ast.ImportFrom))],
        top_level_names=[
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ],
    )


def extract_classes(source: str) -> list[ClassInfo]:
    """Extract class metadata including bases and existing docstrings."""
    tree = ast.parse(source)
    source_lines = source.splitlines()
    classes: list[ClassInfo] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        bases = [ast.unparse(b) for b in node.bases]
        existing_doc = ast.get_docstring(node)

        method_names = [
            n.name for n in ast.iter_child_nodes(node)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        body_start = node.body[0].lineno - 1
        body_end = node.end_lineno or (body_start + 1)
        body_text = "\n".join(source_lines[body_start:body_end])

        classes.append(
            ClassInfo(
                name=node.name,
                bases=bases,
                body=textwrap.dedent(body_text),
                method_names=method_names,
                existing_docstring=existing_doc,
                decorators=[ast.unparse(d) for d in node.decorator_list],
                line_start=node.lineno,
                line_end=body_end,
            )
        )

    return classes


def extract_functions(
    source: str,
    skip_private: bool = False,
    skip_dunder: bool = False,
) -> list[FunctionInfo]:
    """Parse Python source and extract all function/method metadata.

    Walks the full AST to capture top-level functions AND methods inside
    classes. Computes cyclomatic complexity for comment targeting.
    """
    tree = ast.parse(source)
    source_lines = source.splitlines()
    functions: list[FunctionInfo] = []

    # Build a parent map for class detection
    parent_map: dict[int, str] = {}
    for parent in ast.walk(tree):
        if isinstance(parent, ast.ClassDef):
            for child in ast.iter_child_nodes(parent):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parent_map[id(child)] = parent.name

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        name = node.name

        if skip_dunder and name.startswith("__") and name.endswith("__"):
            continue
        if skip_private and name.startswith("_") and not name.startswith("__"):
            continue

        parent_class = parent_map.get(id(node))

        # Build signature string
        sig_parts = []
        for arg in node.args.args:
            ann = ast.unparse(arg.annotation) if arg.annotation else ""
            part = f"{arg.arg}: {ann}" if ann else arg.arg
            sig_parts.append(part)

        returns = ast.unparse(node.returns) if node.returns else ""
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        signature = f"{async_prefix}def {name}({', '.join(sig_parts)})" + (
            f" -> {returns}" if returns else ""
        )

        # Extract body source
        body_start = node.body[0].lineno - 1
        body_end = node.end_lineno or (body_start + 1)
        body_text = "\n".join(source_lines[body_start:body_end])
        body_line_count = body_end - body_start

        existing_doc = ast.get_docstring(node)

        existing_anns: dict[str, str] = {}
        for arg in node.args.args:
            if arg.annotation:
                existing_anns[arg.arg] = ast.unparse(arg.annotation)
        if node.returns:
            existing_anns["return"] = ast.unparse(node.returns)

        decorators = [ast.unparse(d) for d in node.decorator_list]
        complexity = _cyclomatic_complexity(node)

        functions.append(
            FunctionInfo(
                name=name,
                signature=signature,
                body=textwrap.dedent(body_text),
                decorators=decorators,
                existing_docstring=existing_doc,
                existing_annotations=existing_anns,
                line_start=node.lineno,
                line_end=body_end,
                parent_class=parent_class,
                body_line_count=body_line_count,
                cyclomatic_complexity=complexity,
            )
        )

    return functions


def extract_functions_from_file(file_path: Path, **kwargs) -> list[FunctionInfo]:
    """Convenience wrapper: read file then extract."""
    source = file_path.read_text(encoding="utf-8")
    return extract_functions(source, **kwargs)
