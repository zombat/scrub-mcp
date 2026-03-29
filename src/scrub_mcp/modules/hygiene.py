"""DSPy modules for code hygiene tasks.

Each module wraps a signature with Pydantic validation on the output.
The modules are optimizable: run the optimizer once per model, and
DSPy handles prompt adaptation.

Coverage rules (per Ray's spec):
    - ALL files, classes, methods, functions get docstrings
    - ALL arguments and returns get type annotations
    - Long or complex code gets comments (complexity-gated)
    - Google style everywhere
"""

from __future__ import annotations

import logging

import dspy

from scrub_mcp.config import CommentConfig
from scrub_mcp.models import (
    ClassInfo,
    GeneratedDocstring,
    ModuleInfo,
)
from scrub_mcp.modules.signatures import (
    ClassDocstringSignature,
    CommentSignature,
    DocstringSignature,
    ModuleDocstringSignature,
    TypeAnnotationSignature,
)

logger = logging.getLogger(__name__)


class DocstringGenerator(dspy.Module):
    """Generate Google-style docstrings for functions and methods."""

    def __init__(self) -> None:
        """Initializes the BatchTypeAnnotator class with a BatchTypeAnnotationSignature object."""
        super().__init__()
        self.generate = dspy.ChainOfThought(DocstringSignature)

    def forward(
        self,
        function_signature: str,
        function_body: str,
        decorators: str = "",
        parent_class: str = "",
    ) -> str:
        """Processes a JSON string and returns the annotations in JSON format."""
        result = self.generate(
            function_signature=function_signature,
            function_body=function_body,
            decorators=decorators,
            parent_class=parent_class,
        )
        return result.docstring.strip().strip('"""').strip("'''").strip()


class ClassDocstringGenerator(dspy.Module):
    """Generate Google-style docstrings for classes."""

    def __init__(self) -> None:
        """Initializes the BatchTypeAnnotator class with a BatchTypeAnnotationSignature object."""
        super().__init__()
        self.generate = dspy.ChainOfThought(ClassDocstringSignature)

    def forward(self, cls: ClassInfo) -> GeneratedDocstring:
        """Processes a JSON string and returns the annotations in JSON format."""
        result = self.generate(
            class_name=cls.name,
            bases=", ".join(cls.bases),
            method_names=", ".join(cls.method_names),
            class_body=cls.body[:3000],  # Truncate to keep tokens low
        )
        docstring = result.docstring.strip().strip('"""').strip("'''").strip()
        return GeneratedDocstring(
            function_name=cls.name,
            docstring=docstring,
            style="google",
            target_type="class",
        )


class ModuleDocstringGenerator(dspy.Module):
    """Generate Google-style docstrings for Python modules (files)."""

    def __init__(self) -> None:
        """Initializes the BatchTypeAnnotator class with a BatchTypeAnnotationSignature object."""
        super().__init__()
        self.generate = dspy.ChainOfThought(ModuleDocstringSignature)

    def forward(self, module: ModuleInfo) -> GeneratedDocstring:
        """Processes a JSON string and returns the annotations in JSON format."""
        result = self.generate(
            module_imports="\n".join(module.imports[:20]),  # Cap to keep tokens down
            top_level_names=", ".join(module.top_level_names),
            existing_docstring=module.existing_docstring or "",
        )
        docstring = result.docstring.strip().strip('"""').strip("'''").strip()
        return GeneratedDocstring(
            function_name="__module__",
            docstring=docstring,
            style="google",
            target_type="module",
        )


class TypeAnnotator(dspy.Module):
    """Infer type annotations for ALL params and return values."""

    def __init__(self) -> None:
        """Initializes the BatchTypeAnnotator class with a BatchTypeAnnotationSignature object."""
        super().__init__()
        self.annotate = dspy.ChainOfThought(TypeAnnotationSignature)

    def forward(
        self,
        function_signature: str,
        function_body: str,
        existing_annotations: str = "{}",
    ) -> str:
        """Processes a JSON string and returns the annotations in JSON format."""
        result = self.annotate(
            function_signature=function_signature,
            function_body=function_body,
            existing_annotations=existing_annotations,
        )
        return result.annotations_json


class CommentWriter(dspy.Module):
    """Add semantic comments to long or complex code blocks.

    Complexity-gated: only fires when the function exceeds the configured
    line count OR cyclomatic complexity threshold.
    """

    def __init__(self, config: CommentConfig | None = None) -> None:
        """Initializes the BatchTypeAnnotator class with a BatchTypeAnnotationSignature object."""
        super().__init__()
        self.comment = dspy.ChainOfThought(CommentSignature)
        self.config = config or CommentConfig()

    def should_comment(self, line_count: int, cyclomatic_complexity: int) -> bool:
        """Deterministic gate: is this function complex enough to need comments?"""
        if line_count >= self.config.min_lines:
            return True
        if cyclomatic_complexity >= self.config.min_cyclomatic_complexity:
            return True
        return False

    def forward(self, code_block: str, context: str = "", complexity: int = 1) -> str:
        """Processes a JSON string and returns the annotations in JSON format."""
        result = self.comment(
            code_block=code_block,
            context=context,
            complexity=complexity,
        )
        return result.comments_json


class BatchDocstringGenerator(dspy.Module):
    """Generate docstrings for multiple functions in a single LLM call.

    Cuts round trips from N to ceil(N / batch_size). Each call packs
    multiple function signatures + bodies into one prompt and gets back
    a JSON map of name -> docstring.
    """

    def __init__(self) -> None:
        """Initializes the BatchTypeAnnotator class with a BatchTypeAnnotationSignature object."""
        super().__init__()
        from scrub_mcp.modules.signatures import BatchDocstringSignature

        self.generate = dspy.ChainOfThought(BatchDocstringSignature)

    def forward(self, functions_json: str) -> str:
        """Processes a JSON string and returns the annotations in JSON format."""
        result = self.generate(functions_json=functions_json)
        return result.docstrings_json


class BatchTypeAnnotator(dspy.Module):
    """Infer type annotations for multiple functions in a single LLM call.

    Same batching strategy as BatchDocstringGenerator. Returns one
    TypeAnnotation per function.
    """

    def __init__(self) -> None:
        """Initializes the BatchTypeAnnotator class with a BatchTypeAnnotationSignature object."""
        super().__init__()
        from scrub_mcp.modules.signatures import BatchTypeAnnotationSignature

        self.annotate = dspy.ChainOfThought(BatchTypeAnnotationSignature)

    def forward(self, functions_json: str) -> str:
        """Processes a JSON string and returns the annotations in JSON format."""
        result = self.annotate(functions_json=functions_json)
        return result.annotations_json
