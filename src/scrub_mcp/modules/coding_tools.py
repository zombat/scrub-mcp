"""DSPy modules for coding tools.

Same pattern as hygiene modules: deterministic first, DSPy fills gaps.
Each module is independently optimizable.

Tools:
    - ComplexityReducer: suggest simplifications for flagged functions
    - TestGenerator / BatchTestGenerator: pytest stub generation
    - RefactoringAdvisor: extract function, rename, dead code (LLM layer)
    - MissingImportInferrer: suggest missing imports from usage patterns
"""

from __future__ import annotations

import json
import logging
import re

import dspy

from scrub_mcp.modules.coding_signatures import (
    BatchTestGenerationSignature,
    ComplexityAnalysisSignature,
    DeadCodeSignature,
    ExtractFunctionSignature,
    MissingImportSignature,
    RenameSuggestionSignature,
    TestGenerationSignature,
)

logger = logging.getLogger(__name__)


# ── Complexity Reducer ──


class ComplexityReducer(dspy.Module):
    """Suggest concrete simplifications for complex functions.

    Only runs on functions that the deterministic analyzer already
    flagged. The LLM provides specific refactoring suggestions with
    example code.
    """

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(ComplexityAnalysisSignature)

    def forward(
        self,
        function_signature: str,
        function_body: str,
        cyclomatic_complexity: int = 1,
        line_count: int = 0,
    ) -> str:
        result = self.analyze(
            function_signature=function_signature,
            function_body=function_body[:3000],
            cyclomatic_complexity=cyclomatic_complexity,
            line_count=line_count,
        )
        return result.suggestions_json


# ── Test Generation ──


class TestGenerator(dspy.Module):
    """Generate pytest test stubs for a single function."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(TestGenerationSignature)

    def forward(
        self,
        function_signature: str,
        function_body: str,
        docstring: str = "",
        parent_class: str = "",
    ) -> str:
        result = self.generate(
            function_signature=function_signature,
            function_body=function_body,
            docstring=docstring,
            parent_class=parent_class,
            module_path="",
        )
        test_code = result.test_code.strip()
        test_code = re.sub(r"^```python\s*\n?", "", test_code)
        test_code = re.sub(r"\n?```\s*$", "", test_code)
        return test_code


class BatchTestGenerator(dspy.Module):
    """Generate pytest stubs for multiple functions in one LLM call."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(BatchTestGenerationSignature)

    def forward(self, functions_json: str) -> str:
        result = self.generate(
            functions_json=functions_json,
            module_path="",
        )
        test_code = result.test_code.strip()
        test_code = re.sub(r"^```python\s*\n?", "", test_code)
        test_code = re.sub(r"\n?```\s*$", "", test_code)
        return test_code


# ── Refactoring ──


class ExtractFunctionAdvisor(dspy.Module):
    """Identify code blocks that should be extracted into separate functions."""

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractFunctionSignature)

    def forward(self, function_signature: str, function_body: str) -> str:
        result = self.extract(
            function_signature=function_signature,
            function_body=function_body[:3000],
        )
        return result.extractions_json


class RenameAdvisor(dspy.Module):
    """Suggest better names for poorly named variables and functions."""

    def __init__(self) -> None:
        super().__init__()
        self.rename = dspy.ChainOfThought(RenameSuggestionSignature)

    def forward(self, code_block: str, context: str = "") -> str:
        result = self.rename(
            code_block=code_block[:3000],
            context=context,
        )
        return result.renames_json


class DeadCodeDetector(dspy.Module):
    """LLM-powered dead code detection for cases AST misses.

    Catches semantic dead code: functions that are defined but never
    called anywhere in the project, parameters that are accepted but
    never used in complex patterns, etc.
    """

    def __init__(self) -> None:
        super().__init__()
        self.detect = dspy.ChainOfThought(DeadCodeSignature)

    def forward(self, source: str) -> list[dict]:
        result = self.detect(source_code=source[:5000])

        try:
            parsed = json.loads(result.dead_code_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse dead code detection results")
            return []

        return parsed


# ── Import Inference ──


class MissingImportInferrer(dspy.Module):
    """Infer missing imports from usage patterns the linter can't catch.

    Only runs after the deterministic import analyzer has already
    identified potentially missing names. The LLM figures out which
    module they likely come from.
    """

    def __init__(self) -> None:
        super().__init__()
        self.infer = dspy.ChainOfThought(MissingImportSignature)

    def forward(self, source_code: str, existing_imports: str = "") -> str:
        result = self.infer(
            source_code=source_code[:5000],
            existing_imports=existing_imports,
        )
        return result.missing_imports_json


# ── Security Remediation ──


class SecurityRemediationAdvisor(dspy.Module):
    """Triage + remediate Bandit security findings.

    Two-step process:
    1. Triage: is this a real vulnerability or a false positive in context?
    2. Act: rewrite the code, add # nosec with justification, or document accepted risk.

    Feeds false-positive hints from a known pattern database so the LLM
    has context about common Bandit over-flags before making a decision.
    """

    def __init__(self) -> None:
        super().__init__()
        from scrub_mcp.modules.coding_signatures import (
            SecurityTriageSignature,
            _COMMON_FALSE_POSITIVES,
        )

        self.triage = dspy.ChainOfThought(SecurityTriageSignature)
        self._fp_hints = _COMMON_FALSE_POSITIVES

    def forward(
        self,
        test_id: str,
        issue_description: str,
        flagged_code: str,
        surrounding_context: str = "",
        file_context: str = "",
        severity: str = "LOW",
        test_name: str = "",
    ) -> "SecurityTriage":
        from scrub_mcp.models import SecurityTriage

        # Look up known false positive hint
        fp_hint = self._fp_hints.get(test_id, "")

        result = self.triage(
            test_id=test_id,
            test_name=test_name,
            issue_description=issue_description,
            severity=severity,
            flagged_code=flagged_code,
            surrounding_context=surrounding_context[:2000],
            file_context=file_context,
            false_positive_hint=fp_hint,
        )

        try:
            parsed = json.loads(result.triage_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse security triage for %s", test_id)
            return SecurityTriage(
                test_id=test_id,
                verdict="rewrite",
                confidence=0.0,
                description=issue_description,
                rationale="Triage parse failed, defaulting to rewrite for safety.",
            )

        verdict = parsed.get("verdict", "rewrite")
        if verdict not in ("rewrite", "nosec", "accept_risk"):
            verdict = "rewrite"  # Default to safe option

        # Build the nosec annotation line if verdict is nosec
        nosec_line = ""
        if verdict == "nosec":
            # Strip the flagged code and append nosec
            code_line = flagged_code.strip().splitlines()[0] if flagged_code.strip() else ""
            justification = parsed.get("nosec_justification", "reviewed, safe in context")
            nosec_line = f"{code_line}  # nosec {test_id} - {justification}"

        return SecurityTriage(
            test_id=test_id,
            verdict=verdict,
            confidence=parsed.get("confidence", 0.5),
            rationale=parsed.get("rationale", ""),
            nosec_justification=parsed.get("nosec_justification", ""),
            nosec_line=nosec_line,
            remediation=parsed.get("remediation", ""),
            fixed_code=parsed.get("fixed_code", ""),
            risk_if_ignored=parsed.get("risk_if_ignored", ""),
            description=issue_description,
        )
