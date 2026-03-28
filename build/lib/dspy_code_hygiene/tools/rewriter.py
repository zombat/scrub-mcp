"""Source rewriter: applies DSPy outputs back into Python source code.

Operates on source strings using AST for precision. Handles:
    - Module-level docstrings (file header)
    - Class docstrings
    - Function/method docstrings
    - Type annotations (all args + return)
"""

from __future__ import annotations

import ast
import re

from dspy_code_hygiene.models import ClassInfo, GeneratedDocstring, TypeAnnotation


def apply_module_docstring(source: str, docstring: str) -> str:
    """Insert a module-level docstring at the top of the file.

    Preserves shebang lines and encoding declarations. Skips if a
    module docstring already exists.
    """
    tree = ast.parse(source)
    if ast.get_docstring(tree):
        return source  # Already has one

    lines = source.splitlines(keepends=True)

    # Find insertion point (after shebang, encoding, and __future__ imports)
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#!") or stripped.startswith("# -*-") or stripped.startswith("# coding"):
            insert_at = i + 1
        elif stripped.startswith("from __future__"):
            insert_at = i + 1
        elif stripped == "" and i < 3:
            continue
        else:
            break

    formatted = f'"""{docstring}"""\n\n'
    lines.insert(insert_at, formatted)
    return "".join(lines)


def apply_class_docstrings(
    source: str,
    class_docs: list[tuple[ClassInfo, GeneratedDocstring]],
) -> str:
    """Insert docstrings into classes that lack them."""
    if not class_docs:
        return source

    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    # Build a map of class name -> docstring
    doc_map = {cls.name: doc.docstring for cls, doc in class_docs}

    insertions: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in doc_map:
            continue
        if ast.get_docstring(node):
            continue

        body_first = node.body[0]
        body_line = lines[body_first.lineno - 1]
        indent = len(body_line) - len(body_line.lstrip())
        indent_str = " " * indent

        formatted = _format_docstring(doc_map[node.name], indent_str)
        insertions.append((body_first.lineno - 1, formatted))

    # Insert in reverse order
    for line_idx, text in sorted(insertions, reverse=True):
        lines.insert(line_idx, text + "\n")

    return "".join(lines)


def apply_docstrings(source: str, docstrings: list[GeneratedDocstring]) -> str:
    """Insert or replace docstrings in functions/methods."""
    # Filter to function-type docstrings only
    func_docs = [d for d in docstrings if d.target_type == "function"]
    if not func_docs:
        return source

    doc_map = {d.function_name: d.docstring for d in func_docs}
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    insertions: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in doc_map:
            continue

        body_first = node.body[0]
        body_line = lines[body_first.lineno - 1]
        indent = len(body_line) - len(body_line.lstrip())
        indent_str = " " * indent

        new_doc = doc_map[node.name]
        formatted = _format_docstring(new_doc, indent_str)

        existing_doc = ast.get_docstring(node)
        if existing_doc:
            doc_node = node.body[0]
            start_line = doc_node.lineno - 1
            end_line = (doc_node.end_lineno or doc_node.lineno) - 1
            insertions.append((start_line, end_line, formatted))
        else:
            insert_at = body_first.lineno - 1
            insertions.append((insert_at, insert_at - 1, formatted))

    for start, end, new_text in sorted(insertions, key=lambda x: x[0], reverse=True):
        if end < start:
            lines.insert(start, new_text + "\n")
        else:
            lines[start : end + 1] = [new_text + "\n"]

    return "".join(lines)


def apply_type_annotations(source: str, annotations: list[TypeAnnotation]) -> str:
    """Apply inferred type annotations to ALL unannotated function params and returns.

    Only annotates parameters that are currently unannotated. Enforces
    return type on every function.
    """
    if not annotations:
        return source

    ann_map = {a.function_name: a for a in annotations}
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    replacements: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in ann_map:
            continue

        ann = ann_map[node.name]
        sig_line_idx = node.lineno - 1
        sig_line = lines[sig_line_idx].rstrip("\n")

        for param_name, param_type in ann.parameters.items():
            if param_name in ("self", "cls"):
                continue
            # Only annotate bare params (no existing annotation)
            pattern = rf"(\b{re.escape(param_name)}\b)(?!\s*:)(\s*[,\)])"
            replacement = rf"\1: {param_type}\2"
            sig_line = re.sub(pattern, replacement, sig_line)

        # Add return type if missing (required for all functions)
        if "->" not in sig_line:
            ret = ann.return_type if ann.return_type else "None"
            sig_line = re.sub(r"\)\s*:", f") -> {ret}:", sig_line)

        replacements.append((sig_line_idx, lines[sig_line_idx], sig_line + "\n"))

    for idx, _old, new in sorted(replacements, key=lambda x: x[0], reverse=True):
        lines[idx] = new

    return "".join(lines)


def _format_docstring(docstring: str, indent: str) -> str:
    """Format a docstring with proper indentation and triple quotes."""
    doc_lines = docstring.splitlines()
    if len(doc_lines) == 1:
        return f'{indent}"""{doc_lines[0]}"""'

    formatted = f'{indent}"""{doc_lines[0]}\n'
    for line in doc_lines[1:]:
        if line.strip():
            formatted += f"{indent}{line}\n"
        else:
            formatted += "\n"
    formatted += f'{indent}"""'
    return formatted
