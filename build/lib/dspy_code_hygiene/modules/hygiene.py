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

import json
import logging

import dspy

from dspy_code_hygiene.config import CommentConfig
from dspy_code_hygiene.models import (
    ClassInfo,
    FunctionInfo,
    GeneratedDocstring,
    ModuleInfo,
    SemanticComment,
    TypeAnnotation,
)
from dspy_code_hygiene.modules.signatures import (
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
        super().__init__()
        self.generate = dspy.ChainOfThought(DocstringSignature)

    def forward(self, func: FunctionInfo) -> GeneratedDocstring:
        result = self.generate(
            function_signature=func.signature,
            function_body=func.body,
            decorators=", ".join(func.decorators),
            parent_class=func.parent_class or "",
        )
        docstring = result.docstring.strip().strip('"""').strip("'''").strip()
        return GeneratedDocstring(
            function_name=func.name,
            docstring=docstring,
            style="google",
            target_type="function",
        )


class ClassDocstringGenerator(dspy.Module):
    """Generate Google-style docstrings for classes."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(ClassDocstringSignature)

    def forward(self, cls: ClassInfo) -> GeneratedDocstring:
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
        super().__init__()
        self.generate = dspy.ChainOfThought(ModuleDocstringSignature)

    def forward(self, module: ModuleInfo) -> GeneratedDocstring:
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
        super().__init__()
        self.annotate = dspy.ChainOfThought(TypeAnnotationSignature)

    def forward(self, func: FunctionInfo) -> TypeAnnotation:
        result = self.annotate(
            function_signature=func.signature,
            function_body=func.body,
            existing_annotations=json.dumps(func.existing_annotations),
        )

        try:
            parsed = json.loads(result.annotations_json)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse type annotations for %s, returning empty", func.name
            )
            parsed = {}

        # Strip self/cls, they don't need annotations
        parsed.pop("self", None)
        parsed.pop("cls", None)

        return_type = parsed.pop("return", "None")

        # Validate: every non-self/cls arg should have a type
        sig_params = [
            p.strip().split(":")[0].split("=")[0].strip()
            for p in func.signature.split("(", 1)[1].rstrip(")").split(",")
            if p.strip() and p.strip() not in ("self", "cls")
        ]
        missing = [p for p in sig_params if p and p not in parsed and p not in func.existing_annotations]
        if missing:
            logger.warning(
                "Type annotations missing for params %s in %s", missing, func.name
            )

        return TypeAnnotation(
            function_name=func.name,
            parameters=parsed,
            return_type=return_type,
            confidence=0.8,
        )


class CommentWriter(dspy.Module):
    """Add semantic comments to long or complex code blocks.

    Complexity-gated: only fires when the function exceeds the configured
    line count OR cyclomatic complexity threshold.
    """

    def __init__(self, config: CommentConfig | None = None) -> None:
        super().__init__()
        self.comment = dspy.ChainOfThought(CommentSignature)
        self.config = config or CommentConfig()

    def should_comment(self, func: FunctionInfo) -> bool:
        """Deterministic gate: is this function complex enough to need comments?"""
        if func.body_line_count >= self.config.min_lines:
            return True
        if func.cyclomatic_complexity >= self.config.min_cyclomatic_complexity:
            return True
        return False

    def forward(self, func: FunctionInfo) -> list[SemanticComment]:
        if not self.should_comment(func):
            return []

        result = self.comment(
            code_block=func.body,
            context=func.signature,
            complexity=func.cyclomatic_complexity,
        )

        try:
            parsed = json.loads(result.comments_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse comments for %s, returning empty", func.name)
            return []

        comments = []
        for item in parsed:
            comments.append(
                SemanticComment(
                    line_number=func.line_start + item.get("line_offset", 0),
                    comment=item.get("comment", ""),
                    category=item.get("category", "explanation"),
                )
            )
        return comments


class BatchDocstringGenerator(dspy.Module):
    """Generate docstrings for multiple functions in a single LLM call.

    Cuts round trips from N to ceil(N / batch_size). Each call packs
    multiple function signatures + bodies into one prompt and gets back
    a JSON map of name -> docstring.
    """

    def __init__(self) -> None:
        super().__init__()
        from dspy_code_hygiene.modules.signatures import BatchDocstringSignature

        self.generate = dspy.ChainOfThought(BatchDocstringSignature)

    def forward(self, funcs: list[FunctionInfo]) -> list[GeneratedDocstring]:
        payload = [
            {
                "name": f.name,
                "signature": f.signature,
                "body": f.body[:2000],  # Truncate long bodies to cap tokens
                "parent_class": f.parent_class or "",
            }
            for f in funcs
        ]

        result = self.generate(functions_json=json.dumps(payload))

        try:
            parsed = json.loads(result.docstrings_json)
        except json.JSONDecodeError:
            logger.warning("Batch docstring parse failed, falling back to empty")
            return []

        docs = []
        for func in funcs:
            raw = parsed.get(func.name, "")
            if not raw:
                logger.warning("No docstring returned for %s in batch", func.name)
                continue
            docstring = raw.strip().strip('"""').strip("'''").strip()
            docs.append(
                GeneratedDocstring(
                    function_name=func.name,
                    docstring=docstring,
                    style="google",
                    target_type="function",
                )
            )
        return docs


class BatchTypeAnnotator(dspy.Module):
    """Infer type annotations for multiple functions in a single LLM call.

    Same batching strategy as BatchDocstringGenerator. Returns one
    TypeAnnotation per function.
    """

    def __init__(self) -> None:
        super().__init__()
        from dspy_code_hygiene.modules.signatures import BatchTypeAnnotationSignature

        self.annotate = dspy.ChainOfThought(BatchTypeAnnotationSignature)

    def forward(self, funcs: list[FunctionInfo]) -> list[TypeAnnotation]:
        payload = [
            {
                "name": f.name,
                "signature": f.signature,
                "body": f.body[:2000],
            }
            for f in funcs
        ]

        result = self.annotate(functions_json=json.dumps(payload))

        try:
            parsed = json.loads(result.annotations_json)
        except json.JSONDecodeError:
            logger.warning("Batch type annotation parse failed, falling back to empty")
            return []

        annotations = []
        for func in funcs:
            func_anns = parsed.get(func.name, {})
            if not func_anns:
                logger.warning("No type annotations returned for %s in batch", func.name)
                continue

            # Clean self/cls
            func_anns.pop("self", None)
            func_anns.pop("cls", None)
            return_type = func_anns.pop("return", "None")

            annotations.append(
                TypeAnnotation(
                    function_name=func.name,
                    parameters=func_anns,
                    return_type=return_type,
                    confidence=0.8,
                )
            )
        return annotations
