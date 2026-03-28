"""DSPy modules for LLM-powered code hygiene tasks."""

from dspy_code_hygiene.modules.hygiene import (
    BatchDocstringGenerator,
    BatchTypeAnnotator,
    ClassDocstringGenerator,
    CommentWriter,
    DocstringGenerator,
    ModuleDocstringGenerator,
    TypeAnnotator,
)

__all__ = [
    "BatchDocstringGenerator",
    "BatchTypeAnnotator",
    "ClassDocstringGenerator",
    "CommentWriter",
    "DocstringGenerator",
    "ModuleDocstringGenerator",
    "TypeAnnotator",
]
