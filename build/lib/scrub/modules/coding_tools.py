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

from scrub.models import (
    ExtractionSuggestion,
    FunctionInfo,
    MissingImport,
    RenameSuggestion,
    SimplificationSuggestion,
    TestGenerationResult,
)
from scrub.modules.coding_signatures import (
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

    def forward(self, func: FunctionInfo) -> list[SimplificationSuggestion]:
        result = self.analyze(
            function_signature=func.signature,
            function_body=func.body[:3000],
            cyclomatic_complexity=func.cyclomatic_complexity,
            line_count=func.body_line_count,
        )

        try:
            parsed = json.loads(result.suggestions_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse complexity suggestions for %s", func.name)
            return []

        suggestions = []
        for item in parsed:
            suggestions.append(
                SimplificationSuggestion(
                    kind=item.get("kind", "decompose"),
                    description=item.get("description", ""),
                    target_lines=item.get("target_lines", []),
                    suggested_code=item.get("suggested_code", ""),
                    impact=item.get("impact", "medium"),
                )
            )

        return suggestions


# ── Test Generation ──


class TestGenerator(dspy.Module):
    """Generate pytest test stubs for a single function."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(TestGenerationSignature)

    def forward(self, func: FunctionInfo, module_path: str = "") -> TestGenerationResult:
        result = self.generate(
            function_signature=func.signature,
            function_body=func.body,
            docstring=func.existing_docstring or "",
            parent_class=func.parent_class or "",
            module_path=module_path,
        )

        test_code = result.test_code.strip()
        # Strip markdown fences if the LLM included them
        test_code = re.sub(r"^```python\s*\n?", "", test_code)
        test_code = re.sub(r"\n?```\s*$", "", test_code)

        # Count test functions
        test_count = len(re.findall(r"^\s*def test_", test_code, re.MULTILINE))

        return TestGenerationResult(
            test_code=test_code,
            function_names_covered=[func.name],
            test_count=test_count,
        )


class BatchTestGenerator(dspy.Module):
    """Generate pytest stubs for multiple functions in one LLM call."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(BatchTestGenerationSignature)

    def forward(
        self, funcs: list[FunctionInfo], module_path: str = ""
    ) -> TestGenerationResult:
        payload = [
            {
                "name": f.name,
                "signature": f.signature,
                "body": f.body[:2000],
                "docstring": f.existing_docstring or "",
                "parent_class": f.parent_class or "",
            }
            for f in funcs
        ]

        result = self.generate(
            functions_json=json.dumps(payload),
            module_path=module_path,
        )

        test_code = result.test_code.strip()
        test_code = re.sub(r"^```python\s*\n?", "", test_code)
        test_code = re.sub(r"\n?```\s*$", "", test_code)

        test_count = len(re.findall(r"^\s*def test_", test_code, re.MULTILINE))

        return TestGenerationResult(
            test_code=test_code,
            function_names_covered=[f.name for f in funcs],
            test_count=test_count,
        )


# ── Refactoring ──


class ExtractFunctionAdvisor(dspy.Module):
    """Identify code blocks that should be extracted into separate functions."""

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractFunctionSignature)

    def forward(self, func: FunctionInfo) -> list[ExtractionSuggestion]:
        result = self.extract(
            function_signature=func.signature,
            function_body=func.body[:3000],
        )

        try:
            parsed = json.loads(result.extractions_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction suggestions for %s", func.name)
            return []

        return [
            ExtractionSuggestion(
                extracted_name=item.get("extracted_name", ""),
                extracted_signature=item.get("extracted_signature", ""),
                extracted_body=item.get("extracted_body", ""),
                call_replacement=item.get("call_replacement", ""),
                target_lines=item.get("target_lines", []),
                rationale=item.get("rationale", ""),
            )
            for item in parsed
        ]


class RenameAdvisor(dspy.Module):
    """Suggest better names for poorly named variables and functions."""

    def __init__(self) -> None:
        super().__init__()
        self.rename = dspy.ChainOfThought(RenameSuggestionSignature)

    def forward(self, func: FunctionInfo) -> list[RenameSuggestion]:
        result = self.rename(
            code_block=func.body[:3000],
            context=func.signature,
        )

        try:
            parsed = json.loads(result.renames_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse rename suggestions for %s", func.name)
            return []

        return [
            RenameSuggestion(
                old_name=item.get("old_name", ""),
                new_name=item.get("new_name", ""),
                kind=item.get("kind", "variable"),
                rationale=item.get("rationale", ""),
            )
            for item in parsed
        ]


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

    def forward(
        self, source: str, existing_imports: list[str]
    ) -> list[MissingImport]:
        result = self.infer(
            source_code=source[:5000],
            existing_imports="\n".join(existing_imports),
        )

        try:
            parsed = json.loads(result.missing_imports_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse missing import suggestions")
            return []

        imports = []
        for item in parsed:
            conf = item.get("confidence", 0.5)
            if conf < 0.7:
                continue  # Only suggest high-confidence imports
            imports.append(
                MissingImport(
                    import_statement=item.get("import_statement", ""),
                    used_name=item.get("used_name", ""),
                    line_used=item.get("line_used", 0),
                    confidence=conf,
                )
            )

        return imports


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
        from scrub.modules.coding_signatures import (
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
        from scrub.models import SecurityTriage

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
