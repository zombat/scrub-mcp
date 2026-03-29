"""DSPy signatures defining the I/O contract for each code hygiene task.

These are the optimizable units. DSPy tunes the prompts behind these
signatures against your local model, so swapping models just means
re-running the optimizer, not rewriting prompts.
"""

from __future__ import annotations

import dspy


class DocstringSignature(dspy.Signature):
    """Generate a Google-style docstring for a Python function or method.

    Given the function signature and body, produce a clear, concise docstring.
    EVERY parameter must have an Args entry. EVERY return value must have a
    Returns entry. Document raised exceptions under Raises.
    Do NOT include the triple-quote delimiters.
    """

    function_signature: str = dspy.InputField(desc="The function def line with parameters")
    function_body: str = dspy.InputField(desc="The function implementation")
    decorators: str = dspy.InputField(desc="Comma-separated decorator names, if any", default="")
    parent_class: str = dspy.InputField(desc="Enclosing class name, if a method", default="")
    docstring: str = dspy.OutputField(desc="Google-style docstring (no triple quotes)")


class ClassDocstringSignature(dspy.Signature):
    """Generate a Google-style docstring for a Python class.

    Summarize the class purpose. List public attributes under Attributes.
    Reference key methods if they clarify usage. Do NOT document __init__
    params here (those go on __init__'s own docstring).
    Do NOT include the triple-quote delimiters.
    """

    class_name: str = dspy.InputField(desc="The class name")
    bases: str = dspy.InputField(desc="Comma-separated base class names", default="")
    method_names: str = dspy.InputField(desc="Comma-separated method names in the class")
    class_body: str = dspy.InputField(desc="The class body source")
    docstring: str = dspy.OutputField(desc="Google-style class docstring (no triple quotes)")


class ModuleDocstringSignature(dspy.Signature):
    """Generate a Google-style module docstring for a Python file.

    Summarize the module's purpose and key contents. Keep it to 1-3
    sentences plus an optional list of key exports.
    Do NOT include the triple-quote delimiters.
    """

    module_imports: str = dspy.InputField(desc="Import statements in the module")
    top_level_names: str = dspy.InputField(desc="Top-level function/class names")
    existing_docstring: str = dspy.InputField(desc="Current docstring if any", default="")
    docstring: str = dspy.OutputField(desc="Google-style module docstring (no triple quotes)")


class TypeAnnotationSignature(dspy.Signature):
    """Infer type annotations for ALL parameters and return value of a Python function.

    Analyze the function body to determine the most accurate types.
    Use standard library types (str, int, list[str], dict[str, Any], etc.).
    EVERY parameter must get a type. The return type is REQUIRED.
    Output as JSON: {"param_name": "type", ..., "return": "type"}.
    Skip 'self' and 'cls'.
    """

    function_signature: str = dspy.InputField(desc="The function def line")
    function_body: str = dspy.InputField(desc="The function implementation")
    existing_annotations: str = dspy.InputField(
        desc="Already annotated params as JSON", default="{}"
    )
    annotations_json: str = dspy.OutputField(
        desc='JSON: {"param": "type", "return": "ReturnType"} for ALL params'
    )


class CommentSignature(dspy.Signature):
    """Add semantic inline comments to a code block.

    Only comment code that is long, complex, or non-obvious.
    Focus on: branch logic rationale, edge case handling, algorithm steps,
    non-obvious side effects, and workaround explanations.
    Do NOT comment obvious operations like variable assignments or simple returns.
    Output as JSON list:
    [{"line_offset": N, "comment": "text", "category": "explanation|warning|todo|rationale"}]
    """

    code_block: str = dspy.InputField(desc="The code to comment")
    context: str = dspy.InputField(desc="Surrounding context or function purpose", default="")
    complexity: int = dspy.InputField(desc="Cyclomatic complexity score", default=1)
    comments_json: str = dspy.OutputField(
        desc='JSON list of {"line_offset": int, "comment": str, "category": str}'
    )


# ── Batch signatures: N functions in one LLM call ──


class BatchDocstringSignature(dspy.Signature):
    """Generate Google-style docstrings for MULTIPLE Python functions in one call.

    Input is a JSON list of functions. Each has "name", "signature", "body".
    Output a JSON object mapping function name to its docstring.
    EVERY parameter must have an Args entry. EVERY return must have a Returns entry.
    Do NOT include triple-quote delimiters in any docstring.
    Output format: {"func_name": "docstring text", ...}
    """

    functions_json: str = dspy.InputField(
        desc='JSON list of {"name": str, "signature": str, "body": str, "parent_class": str}'
    )
    docstrings_json: str = dspy.OutputField(
        desc='JSON object: {"func_name": "Google-style docstring", ...}'
    )


class BatchTypeAnnotationSignature(dspy.Signature):
    """Infer type annotations for MULTIPLE Python functions in one call.

    Input is a JSON list of functions. Each has "name", "signature", "body".
    Output a JSON object mapping function name to its annotations.
    EVERY parameter must get a type. Return type is REQUIRED. Skip self/cls.
    Output format: {"func_name": {"param": "type", "return": "ReturnType"}, ...}
    """

    functions_json: str = dspy.InputField(
        desc='JSON list of {"name": str, "signature": str, "body": str}'
    )
    annotations_json: str = dspy.OutputField(
        desc='JSON: {"func_name": {"param": "type", "return": "ReturnType"}, ...}'
    )
