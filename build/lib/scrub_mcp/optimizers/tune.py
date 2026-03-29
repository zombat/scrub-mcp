"""DSPy optimizer with per-module strategy selection.

Strategy assignment:
    BootstrapFewShot (fast, style matching):
        - docstrings, types, comments
    BootstrapFewShotWithRandomSearch (moderate, needs variety):
        - imports, dead_code
    MIPROv2 (thorough, prompt wording matters):
        - complexity, tests, refactoring

Run once per model swap. Optimizer learns YOUR conventions from
annotated examples in your codebase.

Usage:
    python -m scrub_mcp.optimizers.tune --examples-dir ./examples
    python -m scrub_mcp.optimizers.tune --modules docstrings,types
    python -m scrub_mcp.optimizers.tune --strategy mipro  # override all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2

from scrub_mcp.config import OptimizerConfig, PipelineConfig, load_config
from scrub_mcp.examples import bundled_examples_dir
from scrub_mcp.modules.coding_tools import (
    BatchTestGenerator,
    ComplexityReducer,
    ExtractFunctionAdvisor,
    MissingImportInferrer,
    RenameAdvisor,
    TestGenerator,
)
from scrub_mcp.modules.hygiene import (
    BatchDocstringGenerator,
    BatchTypeAnnotator,
    CommentWriter,
    DocstringGenerator,
    TypeAnnotator,
)
from scrub_mcp.tools.parser import extract_functions_from_file

logger = logging.getLogger(__name__)


# ── Metrics ──
#
# Two-tier scoring:
#   Tier 1 (structural): fast, deterministic checks (JSON valid, required
#       fields present, Google style markers). Always runs.
#   Tier 2 (LLM-as-judge): teacher model evaluates semantic quality of
#       student output. Only runs during compilation when use_teacher=True.
#
# Final score = 0.4 * structural + 0.6 * judge (when judge available)
# Final score = structural (when self-teaching, no judge)


# Global reference to the judge LM, set during optimization
_JUDGE_LM: dspy.LM | None = None

# Loaded calibration rubrics, keyed by task_type
_CALIBRATION: dict[str, dict] = {}


def set_judge_lm(lm: dspy.LM | None) -> None:
    """Set the LM used for LLM-as-judge metrics during compilation."""
    global _JUDGE_LM
    _JUDGE_LM = lm


def load_calibration(calibration_path: Path | None = None) -> None:
    """Load judge calibration rubrics from a JSON file.

    The calibration file defines per-task scoring rubrics with grounded
    examples at specific score anchors. This prevents the judge from
    drifting into generic "looks good" scoring and anchors it to the
    project owner's actual quality standards.

    Args:
        calibration_path: Path to calibration JSON. Defaults to
            .dspy_cache/judge_calibration.json.

    """
    global _CALIBRATION

    if calibration_path is None:
        calibration_path = Path(".dspy_cache/judge_calibration.json")

    if calibration_path.exists():
        _CALIBRATION = json.loads(calibration_path.read_text())
        logger.info(
            "[judge] Loaded calibration from %s (%d task types)",
            calibration_path,
            len(_CALIBRATION),
        )
    else:
        _CALIBRATION = {}
        logger.info("[judge] No calibration file found, using default rubrics")


def _build_judge_prompt(task_type: str) -> str:
    """Build a calibrated judge system prompt for a specific task type.

    If calibration data exists for this task, includes:
        - Task-specific rubric with weighted criteria
        - Grounded score anchors (what 0.3, 0.7, 0.95 look like)
        - Style requirements specific to this codebase

    Falls back to a strict default rubric if no calibration is loaded.
    """
    cal = _CALIBRATION.get(task_type, {})

    rubric = cal.get("rubric", _DEFAULT_RUBRICS.get(task_type, _DEFAULT_RUBRICS["_fallback"]))
    anchors = cal.get("anchors", {})
    style_rules = cal.get("style_rules", _DEFAULT_STYLE_RULES)

    prompt_parts = [
        "You are a code quality judge calibrated to a specific engineer's standards.",
        "Score the AI-generated output on a scale of 0.0 to 1.0.",
        "",
        "SCORING RUBRIC:",
        rubric,
        "",
        "STYLE REQUIREMENTS (violations reduce score by 0.1-0.2 each):",
        style_rules,
    ]

    if anchors:
        prompt_parts.extend(
            [
                "",
                "CALIBRATION ANCHORS (use these to ground your scoring):",
            ]
        )
        for score_val, example in sorted(anchors.items(), key=lambda x: float(x[0])):
            prompt_parts.append(f"  Score {score_val}: {example}")

    prompt_parts.extend(
        [
            "",
            "Be strict and consistent. A 0.7+ means production-ready with no changes needed.",
            'Output ONLY a JSON object: {"score": float, "rationale": "brief reason"}',
        ]
    )

    return "\n".join(prompt_parts)


# Default rubrics when no calibration file is present
_DEFAULT_STYLE_RULES = """- Google style docstrings (Args, Returns, Raises sections)
- Oxford commas in all lists
- No em dashes (use commas, parentheses, or separate sentences)
- Type annotations use lowercase generics (list[str] not List[str])
- Guard clauses and early returns over nested if/else
- Deterministic checks before LLM calls (deterministic-first pattern)
- Descriptive variable names (no single letters except loop counters)"""

_DEFAULT_RUBRICS: dict[str, str] = {
    "docstring": """Weighted criteria:
  0.30 - Correctness: accurately describes what the function does
  0.25 - Completeness: every parameter documented, return value described, exceptions listed
  0.20 - Style: Google format (Args/Returns/Raises sections), Oxford commas, no em dashes
  0.15 - Conciseness: no filler, no restating the function name as the summary
  0.10 - Usefulness: would a new developer understand the function from the docstring alone?""",
    "type_annotation": """Weighted criteria:
  0.40 - Correctness: types accurately reflect runtime behavior
  0.25 - Completeness: every parameter and return value annotated
  0.20 - Precision: uses specific types (list[str]) not overly broad (Any)
  0.15 - Style: lowercase generics, union syntax (str | None not Optional[str])""",
    "comments": """Weighted criteria:
  0.35 - Relevance: comments explain non-obvious logic, not obvious operations
  0.25 - Accuracy: comment correctly describes what the code does
  0.20 - Categorization: correct category (explanation/warning/todo/rationale)
  0.20 - Value-add: would removing this comment make the code harder to understand?""",
    "test_generation": """Weighted criteria:
  0.30 - Correctness: tests would actually pass against the implementation
  0.25 - Coverage: happy path, edge cases (empty, None, boundary), error cases
  0.20 - Assertions: meaningful asserts that verify behavior, not just "doesn't crash"
  0.15 - Structure: proper use of parametrize, fixtures, descriptive test names
  0.10 - Independence: tests don't depend on each other or external state""",
    "complexity_reduction": """Weighted criteria:
  0.35 - Correctness: suggested refactoring preserves original behavior
  0.25 - Impact: would this actually reduce complexity (fewer branches, less nesting)?
  0.20 - Practicality: suggestion is specific enough to implement directly
  0.10 - Code quality: suggested code follows style rules
  0.10 - Prioritization: high-impact suggestions ranked first""",
    "refactoring": """Weighted criteria:
  0.35 - Correctness: extraction/rename preserves behavior and semantics
  0.25 - Naming: extracted function names are descriptive verb+noun, renames improve clarity
  0.20 - Boundary: extraction point is a clean seam (clear inputs/outputs)
  0.10 - Rationale: explanation makes sense to another developer
  0.10 - Scope: doesn't over-extract (pulling out 2-line helpers that only run once)""",
    "missing_imports": """Weighted criteria:
  0.40 - Correctness: suggested import actually provides the used name
  0.30 - Confidence: only high-confidence suggestions (no guessing)
  0.20 - Specificity: from-import over bare import where appropriate
  0.10 - Ordering: follows isort conventions""",
    "_fallback": """Weighted criteria:
  0.30 - Correctness: output is factually/syntactically right
  0.25 - Completeness: covers all requirements
  0.25 - Quality: well-written, idiomatic Python
  0.20 - Usefulness: developer would actually benefit from this""",
}


class CalibratedJudgeSignature(dspy.Signature):
    """Evaluate AI-generated code output against calibrated quality standards.

    You are a code quality judge calibrated to a specific engineer's
    standards. Use the provided rubric and calibration anchors to score
    consistently. Your scoring must be reproducible: the same output
    should always get the same score +/- 0.05.

    Output ONLY a JSON object: {"score": float, "rationale": "brief reason"}
    """

    rubric: str = dspy.InputField(desc="Task-specific scoring rubric with weighted criteria")
    task_type: str = dspy.InputField(desc="What was the task (e.g., 'docstring', 'test')")
    original_code: str = dspy.InputField(desc="The original function/code being processed")
    generated_output: str = dspy.InputField(desc="The AI-generated artifact to evaluate")
    ground_truth: str = dspy.InputField(desc="Expected output if available, else empty", default="")
    evaluation_json: str = dspy.OutputField(desc='JSON: {"score": float, "rationale": str}')


def _judge_score(
    task_type: str,
    original_code: str,
    generated_output: str,
    ground_truth: str = "",
) -> float:
    """Run the calibrated judge on the student's output.

    The judge uses task-specific rubrics with weighted criteria and
    grounded score anchors. When calibration data is loaded, the judge
    scores against the project owner's actual examples of what 0.3, 0.7,
    and 0.95 look like for each task type.

    Args:
        task_type: Which task to judge (docstring, test_generation, etc.)
        original_code: The source code being processed.
        generated_output: What the student model produced.
        ground_truth: Expected output for comparison, if available.

    Returns:
        Score 0.0-1.0, or -1.0 if the judge is unavailable.

    """
    if _JUDGE_LM is None:
        return -1.0

    try:
        rubric = _build_judge_prompt(task_type)
        judge = dspy.ChainOfThought(CalibratedJudgeSignature)

        with dspy.context(lm=_JUDGE_LM):
            result = judge(
                rubric=rubric,
                task_type=task_type,
                original_code=original_code[:2000],
                generated_output=generated_output[:2000],
                ground_truth=ground_truth[:1000],
            )

        parsed = json.loads(result.evaluation_json)
        score = float(parsed.get("score", 0.0))
        rationale = parsed.get("rationale", "")

        if rationale:
            logger.debug("[judge:%s] %.2f - %s", task_type, score, rationale[:100])

        return max(0.0, min(score, 1.0))

    except Exception:
        logger.debug("[judge] Failed for %s, falling back to structural only", task_type)
        return -1.0


def _combine_scores(structural: float, judge: float) -> float:
    """Blend structural and judge scores.

    If judge unavailable (-1.0), use structural only.
    Otherwise: 0.4 * structural + 0.6 * judge.
    """
    if judge < 0:
        return structural
    return 0.4 * structural + 0.6 * judge


# ── Structural metrics (Tier 1, always runs) ──


def _structural_docstring(example, prediction) -> float:
    """Structural checks for docstring output."""
    doc = getattr(prediction, "docstring", "") or ""
    if not doc:
        doc = getattr(prediction, "docstrings_json", "") or ""
        if not doc:
            return 0.0

    score = 0.2
    if "Args:" in doc:
        score += 0.25
    if "Returns:" in doc or "Return:" in doc:
        score += 0.2
    if "Raises:" in doc:
        score += 0.1

    sig = getattr(example, "function_signature", "")
    param_count = sig.count(",") + 1 if "," in sig else 1
    doc_lines = len(doc.strip().splitlines())
    if param_count > 1 and doc_lines >= param_count:
        score += 0.15

    func_name = sig.split("(")[0].replace("def ", "").replace("async ", "").strip()
    if func_name and func_name.lower() not in doc.lower().split(".")[0].lower():
        score += 0.1

    return min(score, 1.0)


def _structural_types(example, prediction) -> float:
    """Structural checks for type annotations."""
    expected_raw = getattr(example, "annotations_json", None)
    predicted_raw = getattr(prediction, "annotations_json", None)

    if not expected_raw:
        return 0.5

    try:
        expected = json.loads(expected_raw) if isinstance(expected_raw, str) else expected_raw
        predicted = (
            json.loads(predicted_raw) if isinstance(predicted_raw, str) else predicted_raw or {}
        )
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not expected:
        return 0.5

    correct = sum(1 for k, v in expected.items() if predicted.get(k) == v)
    partial = sum(
        0.5
        for k, v in expected.items()
        if k in predicted and predicted[k] != v and _types_compatible(v, predicted[k])
    )

    return (correct + partial) / len(expected)


def _structural_comments(example, prediction) -> float:
    """Structural checks for comment output."""
    raw = getattr(prediction, "comments_json", "[]")
    try:
        comments = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not comments:
        return 0.2

    score = 0.3
    valid_categories = {"explanation", "warning", "todo", "rationale"}
    categorized = sum(1 for c in comments if c.get("category") in valid_categories)
    score += 0.3 * (categorized / len(comments))

    avg_len = sum(len(c.get("comment", "")) for c in comments) / len(comments)
    if avg_len > 20:
        score += 0.2
    if avg_len > 40:
        score += 0.2

    return min(score, 1.0)


def _structural_tests(example, prediction) -> float:
    """Structural checks for test generation."""
    code = getattr(prediction, "test_code", "") or ""
    if not code:
        return 0.0

    import re

    score = 0.2

    test_count = len(re.findall(r"def test_", code))
    if test_count >= 1:
        score += 0.15
    if test_count >= 3:
        score += 0.15
    if "assert" in code:
        score += 0.15
    if "pytest.raises" in code:
        score += 0.1
    if "@pytest.mark.parametrize" in code:
        score += 0.1
    if "import pytest" in code:
        score += 0.05

    edge_patterns = ["_edge", "_empty", "_none", "_invalid", "_boundary", "_error"]
    if any(p in code for p in edge_patterns):
        score += 0.1

    return min(score, 1.0)


def _structural_complexity(example, prediction) -> float:
    """Structural checks for complexity suggestions."""
    raw = getattr(prediction, "suggestions_json", "[]")
    try:
        suggestions = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not suggestions:
        return 0.1

    score = 0.2
    valid_kinds = {
        "early_return",
        "extract_function",
        "guard_clause",
        "simplify_bool",
        "lookup_table",
        "decompose",
    }

    for s in suggestions:
        if s.get("kind") in valid_kinds:
            score += 0.15
        if s.get("suggested_code", "").strip():
            score += 0.1
        if s.get("description", "").strip() and len(s["description"]) > 15:
            score += 0.1

    return min(score / max(len(suggestions), 1), 1.0)


def _structural_refactoring(example, prediction) -> float:
    """Structural checks for refactoring suggestions."""
    for attr in ("extractions_json", "renames_json"):
        raw = getattr(prediction, attr, None)
        if raw:
            try:
                items = json.loads(raw) if isinstance(raw, str) else raw
                if items:
                    return min(0.3 + 0.15 * len(items), 1.0)
            except (json.JSONDecodeError, TypeError):
                continue
    return 0.2


def _structural_imports(example, prediction) -> float:
    """Structural checks for import suggestions."""
    raw = getattr(prediction, "missing_imports_json", "[]")
    try:
        imports = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not imports:
        return 0.3

    score = 0.3
    for imp in imports:
        stmt = imp.get("import_statement", "")
        if stmt.startswith("from ") or stmt.startswith("import "):
            score += 0.15
        conf = imp.get("confidence", 0)
        if conf >= 0.8:
            score += 0.1

    return min(score, 1.0)


# ── Combined metrics (structural + judge) ──


def docstring_metric(example, prediction, trace=None) -> float:
    """Docstring quality: structural Google style checks + judge."""
    structural = _structural_docstring(example, prediction)
    doc = getattr(prediction, "docstring", "") or ""
    sig = getattr(example, "function_signature", "")
    gt = getattr(example, "docstring", "")
    judge = _judge_score("docstring", sig, doc, gt)
    return _combine_scores(structural, judge)


def type_annotation_metric(example, prediction, trace=None) -> float:
    """Type annotation quality: exact match + judge for edge cases."""
    structural = _structural_types(example, prediction)
    predicted_raw = getattr(prediction, "annotations_json", "") or ""
    sig = getattr(example, "function_signature", "")
    gt = getattr(example, "annotations_json", "")
    judge = _judge_score("type_annotation", sig, predicted_raw, gt)
    return _combine_scores(structural, judge)


def comment_metric(example, prediction, trace=None) -> float:
    """Comment quality: structural checks + judge for relevance."""
    structural = _structural_comments(example, prediction)
    raw = getattr(prediction, "comments_json", "[]")
    code = getattr(example, "code_block", "") or getattr(example, "function_body", "")
    judge = _judge_score("comments", code, raw)
    return _combine_scores(structural, judge)


def test_generation_metric(example: dict, prediction: dict, trace=None) -> float:
    """Test quality: structural pytest checks + judge for correctness.

    This is where the judge matters most. Structural checks can confirm
    'def test_' exists, but only the judge can evaluate whether the test
    assertions are actually meaningful and the edge cases are real.
    """
    structural = _structural_tests(example, prediction)
    code = getattr(prediction, "test_code", "") or ""
    sig = getattr(example, "function_signature", "")
    body = getattr(example, "function_body", "")
    gt = getattr(example, "test_code", "")
    judge = _judge_score("test_generation", f"{sig}\n{body}", code, gt)
    return _combine_scores(structural, judge)


def complexity_metric(example: dict, prediction: dict, trace=None) -> float:
    """Complexity suggestion quality: valid structure + judge for correctness.

    The judge evaluates whether the suggested refactoring would actually
    reduce complexity without breaking functionality.
    """
    structural = _structural_complexity(example, prediction)
    raw = getattr(prediction, "suggestions_json", "[]")
    body = getattr(example, "function_body", "")
    gt = getattr(example, "suggestions_json", "")
    judge = _judge_score("complexity_reduction", body, raw, gt)
    return _combine_scores(structural, judge)


def refactoring_metric(example: dict, prediction: dict, trace=None) -> float:
    """Refactoring quality: valid structure + judge for semantic correctness."""
    structural = _structural_refactoring(example, prediction)
    for attr in ("extractions_json", "renames_json"):
        raw = getattr(prediction, attr, None)
        if raw:
            body = getattr(example, "function_body", "") or getattr(example, "code_block", "")
            judge = _judge_score(
                "refactoring", body, raw if isinstance(raw, str) else json.dumps(raw)
            )
            return _combine_scores(structural, judge)
    return structural


def import_metric(example: dict, prediction: dict, trace=None) -> float:
    """Import inference quality: valid structure + judge for correctness."""
    structural = _structural_imports(example, prediction)
    raw = getattr(prediction, "missing_imports_json", "[]")
    source = getattr(example, "source_code", "")
    judge = _judge_score("missing_imports", source, raw)
    return _combine_scores(structural, judge)


def _types_compatible(expected: str, predicted: str) -> bool:
    """Check if two type annotations are semantically close."""
    normalize = lambda t: (
        t.replace("Optional[", "")
        .replace("]", "")
        .replace("List", "list")
        .replace("Dict", "dict")
        .replace("Tuple", "tuple")
        .replace("Set", "set")
        .replace(" | None", "")
        .strip()
    )
    return normalize(expected) == normalize(predicted)


# ── Optimizer factory ──


def get_optimizer(
    strategy: str,
    metric: Callable,
    opt_config: OptimizerConfig,
) -> BootstrapFewShot | BootstrapFewShotWithRandomSearch | MIPROv2:
    """Create the right optimizer instance for a given strategy.

    Args:
        strategy: One of 'bootstrap', 'bootstrap_rs', 'mipro'.
        metric: Scoring function for this module.
        opt_config: Optimizer hyperparameters.

    Returns:
        Configured DSPy optimizer instance.

    """
    if strategy == "bootstrap":
        return BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=opt_config.bootstrap_max_demos,
            max_labeled_demos=opt_config.bootstrap_max_demos,
        )
    elif strategy == "bootstrap_rs":
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=opt_config.bootstrap_max_demos,
            max_labeled_demos=opt_config.bootstrap_max_demos,
            num_candidate_programs=opt_config.bootstrap_rs_trials,
            num_threads=opt_config.num_threads,
        )
    elif strategy == "mipro":
        return MIPROv2(
            metric=metric,
            auto=opt_config.mipro_auto,
            num_threads=opt_config.num_threads,
        )
    else:
        logger.warning("Unknown strategy '%s', falling back to bootstrap", strategy)
        return BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=opt_config.bootstrap_max_demos,
        )


# ── Example loaders ──


def _empty_examples() -> dict[str, list[dict]]:
    """Returns a dictionary with empty lists for various categories.

    Returns:
        dict[str, list[dict]]: A dictionary containing empty lists for 'docstrings', 'types', 'comments', 'tests', 'complexity', 'refactoring', and 'imports'.
    """
    return {
        "docstrings": [],
        "types": [],
        "comments": [],
        "tests": [],
        "complexity": [],
        "refactoring": [],
        "imports": [],
    }


def _extract_tests_for(test_source: str, func_name: str) -> str:
    """Extract test functions targeting *func_name* from *test_source*.

    Collects any function whose name starts with ``test_{func_name}``,
    prepends the file's import statements, and returns them as one string.
    Returns an empty string if no matching tests are found.
    """
    import ast as _ast

    try:
        tree = _ast.parse(test_source)
    except SyntaxError:
        return ""

    prefix = f"test_{func_name}"
    lines = test_source.splitlines()
    chunks: list[str] = []

    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name.startswith(prefix):
            chunks.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))

    if not chunks:
        return ""

    imports = [
        _ast.unparse(node) for node in tree.body if isinstance(node, (_ast.Import, _ast.ImportFrom))
    ]
    header = "\n".join(imports) + "\n\n" if imports else ""
    return header + "\n\n".join(chunks)


def _strip_imports(source: str) -> tuple[str, list[dict]]:
    """Remove all import statements from *source* and return the stripped source
    plus a list of missing-import dicts suitable for ``missing_imports_json``.
    """
    import ast as _ast

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return source, []

    import_nodes = [
        n for n in _ast.iter_child_nodes(tree) if isinstance(n, (_ast.Import, _ast.ImportFrom))
    ]
    if not import_nodes:
        return source, []

    import_line_nos: set[int] = set()
    for n in import_nodes:
        for ln in range(n.lineno - 1, getattr(n, "end_lineno", n.lineno)):
            import_line_nos.add(ln)

    stripped = "\n".join(
        line for i, line in enumerate(source.splitlines()) if i not in import_line_nos
    )

    missing: list[dict] = []
    seen_stmts: set[str] = set()
    for n in import_nodes:
        stmt = _ast.unparse(n)
        if stmt in seen_stmts:
            continue
        seen_stmts.add(stmt)
        if isinstance(n, _ast.Import):
            for alias in n.names:
                missing.append(
                    {
                        "import_statement": stmt,
                        "used_name": alias.asname or alias.name.split(".")[0],
                        "line_used": 0,
                        "confidence": 1.0,
                    }
                )
        else:
            for alias in n.names:
                missing.append(
                    {
                        "import_statement": stmt,
                        "used_name": alias.asname or alias.name,
                        "line_used": 0,
                        "confidence": 1.0,
                    }
                )

    return stripped, missing


def _load_from_pairs(examples_dir: Path) -> dict[str, list[dict]]:
    """Load training examples from ``*_messy.py`` + ``*_clean.py`` file pairs.

    Docstring and type ground truth are derived by parsing the clean file;
    tests come from an optional ``*_test.py`` companion.
    Complexity ground truth is derived when the clean version has lower CC.
    Import ground truth is derived by stripping imports from the messy file.
    """
    all_examples = _empty_examples()

    for messy_file in sorted(examples_dir.glob("*_messy.py")):
        stem = messy_file.stem[: -len("_messy")]
        clean_file = examples_dir / f"{stem}_clean.py"
        test_file = examples_dir / f"{stem}_test.py"

        if not clean_file.exists():
            logger.warning("No clean file for %s, skipping", messy_file.name)
            continue

        messy_source = messy_file.read_text()
        messy_funcs = extract_functions_from_file(messy_file)
        clean_by_name = {f.name: f for f in extract_functions_from_file(clean_file)}
        test_source = test_file.read_text() if test_file.exists() else ""

        for func in messy_funcs:
            base = {
                "function_signature": func.signature,
                "function_body": func.body,
                "decorators": ", ".join(func.decorators),
                "parent_class": func.parent_class or "",
            }
            clean = clean_by_name.get(func.name)

            if clean and clean.existing_docstring:
                all_examples["docstrings"].append({**base, "docstring": clean.existing_docstring})

            if clean and clean.existing_annotations:
                all_examples["types"].append(
                    {
                        **base,
                        "annotations_json": json.dumps(clean.existing_annotations),
                    }
                )

            if test_source:
                test_code = _extract_tests_for(test_source, func.name)
                if test_code:
                    all_examples["tests"].append(
                        {
                            **base,
                            "docstring": clean.existing_docstring if clean else "",
                            "test_code": test_code,
                        }
                    )

            all_examples["comments"].append(
                {
                    "code_block": func.body,
                    "context": func.signature,
                    "complexity": func.cyclomatic_complexity,
                }
            )
            all_examples["refactoring"].append(base)

            # Complexity: use messy→clean diff as ground truth when clean is simpler
            if (
                clean
                and func.cyclomatic_complexity >= 3
                and clean.cyclomatic_complexity < func.cyclomatic_complexity
            ):
                impact = "high" if func.cyclomatic_complexity >= 6 else "medium"
                all_examples["complexity"].append(
                    {
                        "function_signature": func.signature,
                        "function_body": func.body,
                        "cyclomatic_complexity": func.cyclomatic_complexity,
                        "line_count": func.body_line_count,
                        "suggestions_json": json.dumps(
                            [
                                {
                                    "kind": "decompose",
                                    "description": (
                                        f"Reduce cyclomatic complexity from {func.cyclomatic_complexity} "
                                        f"to {clean.cyclomatic_complexity}"
                                    ),
                                    "target_lines": [func.line_start, func.line_end],
                                    "suggested_code": clean.body,
                                    "impact": impact,
                                }
                            ]
                        ),
                    }
                )

        # Imports: one file-level example per pair — strip imports from messy source
        stripped_source, missing_imports = _strip_imports(messy_source)
        if missing_imports:
            all_examples["imports"].append(
                {
                    "source_code": stripped_source,
                    "existing_imports": "",
                    "missing_imports_json": json.dumps(missing_imports),
                }
            )

    return all_examples


def _load_from_annotated(examples_dir: Path) -> dict[str, list[dict]]:
    """Load training examples from ``.py`` + ``.json`` annotated pairs (legacy format)."""
    all_examples = _empty_examples()

    for py_file in sorted(examples_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        json_file = py_file.with_suffix(".json")
        if not json_file.exists():
            logger.warning("No ground truth for %s, skipping", py_file)
            continue

        functions = extract_functions_from_file(py_file)
        ground_truth = json.loads(json_file.read_text())

        for func in functions:
            base = {
                "function_signature": func.signature,
                "function_body": func.body,
                "decorators": ", ".join(func.decorators),
                "parent_class": func.parent_class or "",
            }

            if func.name in ground_truth.get("docstrings", {}):
                all_examples["docstrings"].append(
                    {
                        **base,
                        "docstring": ground_truth["docstrings"][func.name],
                    }
                )

            if func.name in ground_truth.get("types", {}):
                all_examples["types"].append(
                    {
                        **base,
                        "annotations_json": json.dumps(ground_truth["types"][func.name]),
                    }
                )

            if func.name in ground_truth.get("tests", {}):
                all_examples["tests"].append(
                    {
                        **base,
                        "docstring": func.existing_docstring or "",
                        "test_code": ground_truth["tests"][func.name],
                    }
                )

            if func.name in ground_truth.get("simplifications", {}):
                all_examples["complexity"].append(
                    {
                        **base,
                        "cyclomatic_complexity": func.cyclomatic_complexity,
                        "line_count": func.body_line_count,
                        "suggestions_json": json.dumps(ground_truth["simplifications"][func.name]),
                    }
                )

            all_examples["comments"].append(
                {
                    "code_block": func.body,
                    "context": func.signature,
                    "complexity": func.cyclomatic_complexity,
                }
            )
            all_examples["refactoring"].append(base)

        # Imports: from explicit ground truth key OR fall back to stripping the file
        if "imports" in ground_truth:
            all_examples["imports"].append(
                {
                    "source_code": ground_truth["imports"].get("source_code", ""),
                    "existing_imports": ground_truth["imports"].get("existing_imports", ""),
                    "missing_imports_json": json.dumps(ground_truth["imports"].get("missing", [])),
                }
            )
        else:
            stripped_source, missing_imports = _strip_imports(py_file.read_text())
            if missing_imports:
                all_examples["imports"].append(
                    {
                        "source_code": stripped_source,
                        "existing_imports": "",
                        "missing_imports_json": json.dumps(missing_imports),
                    }
                )

    return all_examples


def load_examples(examples_dir: Path) -> dict[str, list[dict]]:
    """Load training examples from *examples_dir*.

    Supports two formats:
    - **Messy/clean pairs** (preferred): ``*_messy.py`` + ``*_clean.py`` + optional ``*_test.py``
    - **Annotated pairs** (legacy): ``*.py`` + companion ``*.json``

    Returns a dict keyed by module name with lists of raw example dicts.
    """
    if any(examples_dir.glob("*_messy.py")):
        logger.info("[examples] messy/clean pairs in %s", examples_dir)
        all_examples = _load_from_pairs(examples_dir)
    else:
        logger.info("[examples] .py+.json pairs in %s", examples_dir)
        all_examples = _load_from_annotated(examples_dir)

    for key, exs in all_examples.items():
        logger.info("Loaded %d %s examples", len(exs), key)

    return all_examples


# ── Module registry ──

MODULE_REGISTRY: dict[str, dict[str, Any]] = {
    "docstrings": {
        "module_cls": DocstringGenerator,
        "metric": docstring_metric,
        "input_fields": ["function_signature", "function_body", "decorators", "parent_class"],
        "output_field": "docstring",
    },
    "docstrings_batch": {
        "module_cls": BatchDocstringGenerator,
        "metric": docstring_metric,
        "input_fields": ["functions_json"],
        "output_field": "docstrings_json",
        "strategy_key": "docstrings",
    },
    "types": {
        "module_cls": TypeAnnotator,
        "metric": type_annotation_metric,
        "input_fields": ["function_signature", "function_body", "existing_annotations"],
        "output_field": "annotations_json",
    },
    "types_batch": {
        "module_cls": BatchTypeAnnotator,
        "metric": type_annotation_metric,
        "input_fields": ["functions_json"],
        "output_field": "annotations_json",
        "strategy_key": "types",
    },
    "comments": {
        "module_cls": CommentWriter,
        "metric": comment_metric,
        "input_fields": ["code_block", "context", "complexity"],
        "output_field": "comments_json",
    },
    "complexity": {
        "module_cls": ComplexityReducer,
        "metric": complexity_metric,
        "input_fields": [
            "function_signature",
            "function_body",
            "cyclomatic_complexity",
            "line_count",
        ],
        "output_field": "suggestions_json",
    },
    "tests": {
        "module_cls": TestGenerator,
        "metric": test_generation_metric,
        "input_fields": ["function_signature", "function_body", "docstring", "parent_class"],
        "output_field": "test_code",
    },
    "tests_batch": {
        "module_cls": BatchTestGenerator,
        "metric": test_generation_metric,
        "input_fields": ["functions_json"],
        "output_field": "test_code",
        "strategy_key": "tests",
    },
    "refactoring_extract": {
        "module_cls": ExtractFunctionAdvisor,
        "metric": refactoring_metric,
        "input_fields": ["function_signature", "function_body"],
        "output_field": "extractions_json",
        "strategy_key": "refactoring",
    },
    "refactoring_rename": {
        "module_cls": RenameAdvisor,
        "metric": refactoring_metric,
        "input_fields": ["code_block", "context"],
        "output_field": "renames_json",
        "strategy_key": "refactoring",
    },
    "imports": {
        "module_cls": MissingImportInferrer,
        "metric": import_metric,
        "input_fields": ["source_code", "existing_imports"],
        "output_field": "missing_imports_json",
    },
}


# ── Main optimize loop ──


def optimize(
    config: PipelineConfig,
    examples_dir: Path,
    output_dir: Path,
    modules_filter: set[str] | None = None,
    strategy_override: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run per-module optimization with teacher-student strategy dispatch.

    Teacher-student flow:
        1. Configure teacher LM (cloud model) for compilation
        2. Optimizer generates candidate prompts using teacher
        3. Evaluate candidates against student LM (local Qwen Coder)
        4. Save optimized prompts that work best on the student
        5. Runtime pipeline loads saved prompts, runs on student only

    When use_teacher is False, both compilation and evaluation
    run on the student model (same as standard DSPy optimization).

    Args:
        config: Pipeline config with model and optimizer settings.
        examples_dir: Directory with .py + .json training pairs.
        output_dir: Where to save optimized module JSON files.
        modules_filter: Only optimize these modules. None = all.
        strategy_override: Force all modules to use this strategy.

    Returns:
        Dict of module_name -> {"strategy": str, "score": float, "time_s": float}

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_cfg = config.optimizer

    # ── Configure LMs ──
    student_lm = _build_lm(config.model, label="student")

    if opt_cfg.use_teacher:
        teacher_lm = _build_lm(opt_cfg.teacher, label="teacher")
        # Activate LLM-as-judge: teacher evaluates student output quality
        set_judge_lm(teacher_lm)
        # Load calibration rubrics (grounded score anchors)
        calibration_path = output_dir / "judge_calibration.json"
        load_calibration(calibration_path)
        logger.info(
            "[teacher-student] Teacher: %s (%s), Student: %s (%s)",
            opt_cfg.teacher.model,
            opt_cfg.teacher.provider,
            config.model.model,
            config.model.provider,
        )
        logger.info("[judge] Calibrated LLM-as-judge active (0.4 structural + 0.6 judge)")
    else:
        teacher_lm = student_lm
        set_judge_lm(None)  # No judge, structural metrics only
        logger.info(
            "[self-teach] Compiling on student model: %s (%s) (structural metrics only)",
            config.model.model,
            config.model.provider,
        )

    all_examples = load_examples(examples_dir)
    results: dict[str, dict[str, Any]] = {}

    for module_name, reg in MODULE_REGISTRY.items():
        # Filter check
        base_name = reg.get("strategy_key", module_name)
        if modules_filter and base_name not in modules_filter and module_name not in modules_filter:
            continue

        # Get examples for this module
        example_key = base_name.split("_")[0] if "_" in base_name else base_name
        examples = all_examples.get(example_key, [])
        if not examples:
            logger.info("[%s] No training examples, skipping", module_name)
            continue

        # Build trainset
        input_fields = reg["input_fields"]
        output_field = reg["output_field"]

        trainset = []
        for ex in examples:
            if output_field not in ex:
                continue
            example_kwargs = {f: ex.get(f, "") for f in input_fields}
            example_kwargs[output_field] = ex[output_field]
            trainset.append(dspy.Example(**example_kwargs).with_inputs(*input_fields))

        if not trainset:
            logger.info("[%s] No valid training examples with output, skipping", module_name)
            continue

        # Determine strategy
        strategy = strategy_override or getattr(opt_cfg, base_name, "bootstrap")
        metric = reg["metric"]

        # ── Teacher-student LM selection ──
        # For MIPROv2: teacher generates candidate instructions,
        #   student evaluates them (so optimized prompts work on student)
        # For Bootstrap: teacher generates bootstrapped demos,
        #   student validates they transfer
        if opt_cfg.use_teacher and strategy == "mipro":
            # MIPROv2 supports explicit teacher/student via prompt_model
            dspy.configure(lm=student_lm)
            logger.info(
                "[%s] MIPROv2 teacher-student: teacher=%s, student=%s",
                module_name,
                opt_cfg.teacher.model,
                config.model.model,
            )
        elif opt_cfg.use_teacher:
            # Bootstrap strategies: compile with teacher, then validate on student
            dspy.configure(lm=teacher_lm)
            logger.info(
                "[%s] %s with teacher: %s (will validate on student)",
                module_name,
                strategy,
                opt_cfg.teacher.model,
            )
        else:
            dspy.configure(lm=student_lm)

        logger.info(
            "[%s] Optimizing with %s strategy (%d examples)",
            module_name,
            strategy,
            len(trainset),
        )

        optimizer = get_optimizer(strategy, metric, opt_cfg)

        # For MIPROv2 with teacher, pass teacher as prompt_model
        compile_kwargs: dict[str, Any] = {"trainset": trainset}
        if opt_cfg.use_teacher and strategy == "mipro":
            compile_kwargs["teacher"] = teacher_lm

        module_instance = reg["module_cls"]()

        start = time.time()
        try:
            optimized = optimizer.compile(module_instance, **compile_kwargs)
            elapsed = time.time() - start

            # ── Validate on student ──
            # Always evaluate the optimized module on the student LM
            # to confirm the prompts transfer correctly.
            dspy.configure(lm=student_lm)

            scores = []
            for ex in trainset[:10]:
                try:
                    pred = optimized(**{f: getattr(ex, f) for f in input_fields})
                    scores.append(metric(ex, pred))
                except Exception:
                    scores.append(0.0)

            avg_score = sum(scores) / max(len(scores), 1)

            # Save
            save_path = output_dir / f"{module_name}.json"
            optimized.save(str(save_path))

            # Stamp model fingerprint for staleness detection
            from scrub_mcp.optimizers.health import save_fingerprint

            save_fingerprint(output_dir, module_name, config)

            results[module_name] = {
                "strategy": strategy,
                "teacher": opt_cfg.teacher.model if opt_cfg.use_teacher else "self",
                "student": config.model.model,
                "score_on_student": round(avg_score, 3),
                "time_s": round(elapsed, 1),
                "examples": len(trainset),
                "saved_to": str(save_path),
            }

            logger.info(
                "[%s] Done in %.1fs, student score: %.3f, saved to %s",
                module_name,
                elapsed,
                avg_score,
                save_path,
            )

        except Exception:
            elapsed = time.time() - start
            logger.exception("[%s] Optimization failed after %.1fs", module_name, elapsed)
            results[module_name] = {
                "strategy": strategy,
                "teacher": opt_cfg.teacher.model if opt_cfg.use_teacher else "self",
                "student": config.model.model,
                "score_on_student": 0.0,
                "time_s": round(elapsed, 1),
                "error": True,
            }

    # Summary
    logger.info("=" * 70)
    logger.info("Optimization summary:")
    logger.info(
        "  Teacher: %s | Student: %s",
        opt_cfg.teacher.model if opt_cfg.use_teacher else "(self-teach)",
        config.model.model,
    )
    logger.info("-" * 70)
    for name, info in results.items():
        status = "FAIL" if info.get("error") else f"student_score={info['score_on_student']:.3f}"
        logger.info(
            "  %-25s  %-15s  %s  (%.1fs)",
            name,
            info["strategy"],
            status,
            info["time_s"],
        )
    logger.info("=" * 70)

    return results


def _build_lm(model_config: ModelConfig, label: str = "") -> dspy.LM:
    """Build a DSPy LM instance from a ModelConfig.

    Args:
        model_config: Model configuration.
        label: Label for logging.

    Returns:
        Configured dspy.LM instance.

    """
    if model_config.provider == "ollama":
        lm = dspy.LM(
            model=f"ollama_chat/{model_config.model}",
            api_base=model_config.base_url,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )
    elif model_config.provider == "anthropic":
        lm = dspy.LM(
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )
    elif model_config.provider == "apple":
        lm = dspy.LM(
            model=f"apple_local/{model_config.model}",
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )
    else:
        lm = dspy.LM(
            model=model_config.model,
            api_base=model_config.base_url,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )

    if label:
        logger.info("[%s] Configured: %s via %s", label, model_config.model, model_config.provider)

    return lm


def main() -> None:
    """Optimize DSPy code hygiene modules with per-module strategy selection.

    Args:
        --examples-dir (Path, optional): Directory with training examples (default: bundled examples shipped with the package).
        --output-dir (Path, optional): Where to save optimized modules (default: .dspy_cache).
        --config (Path, optional): Path to config.yaml.
        --modules (str, optional): Comma-separated module names to optimize (default: all). E.g.: docstrings,types,tests.
        --strategy (str, optional): Override strategy for all modules (ignores per-module config).
        --teacher (bool, optional): Use teacher model for compilation (cloud), validate on student (local) (default: False).
        --teacher-model (str, optional): Override teacher model (e.g., 'claude-sonnet-4-20250514'). Implies --teacher.
        --teacher-provider (str, optional): Teacher model provider. Defaults to config value.
        --build-examples (Path, optional): Generate training examples into DIR using Claude API before tuning. Requires ANTHROPIC_API_KEY. Uses --teacher-model if set, otherwise defaults to claude-sonnet-4-20250514.
        --build-count (int, optional): Number of example triplets to generate with --build-examples (default: 10).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Optimize DSPy code hygiene modules with per-module strategy selection",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Directory with training examples (default: bundled examples shipped with the package)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".dspy_cache"),
        help="Where to save optimized modules",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Comma-separated module names to optimize (default: all). "
        "E.g.: docstrings,types,tests",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["bootstrap", "bootstrap_rs", "mipro"],
        help="Override strategy for all modules (ignores per-module config)",
    )
    parser.add_argument(
        "--teacher",
        action="store_true",
        default=False,
        help="Use teacher model for compilation (cloud), validate on student (local)",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Override teacher model (e.g., 'claude-sonnet-4-20250514'). Implies --teacher.",
    )
    parser.add_argument(
        "--teacher-provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "ollama"],
        help="Teacher model provider. Defaults to config value.",
    )
    parser.add_argument(
        "--build-examples",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Generate training examples into DIR using Claude API before tuning. "
            "Requires ANTHROPIC_API_KEY. Uses --teacher-model if set, "
            "otherwise defaults to claude-sonnet-4-20250514."
        ),
    )
    parser.add_argument(
        "--build-count",
        type=int,
        default=10,
        help="Number of example triplets to generate with --build-examples (default: 10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.build_examples:
        from scrub_mcp.optimizers.examples_gen import generate_examples

        build_model = args.teacher_model or "claude-sonnet-4-20250514"
        logger.info(
            "Building %d training examples into %s using %s ...",
            args.build_count,
            args.build_examples,
            build_model,
        )
        ok = generate_examples(args.build_examples, model=build_model, count=args.build_count)
        if ok == 0:
            logger.error("No examples generated — aborting. Check ANTHROPIC_API_KEY.")
            raise SystemExit(1)

    examples_dir = args.build_examples or args.examples_dir or bundled_examples_dir()

    config = load_config(args.config)
    modules_filter = set(args.modules.split(",")) if args.modules else None

    # Apply teacher overrides
    if args.teacher or args.teacher_model:
        config.optimizer.use_teacher = True
        if args.teacher_model:
            config.optimizer.teacher.model = args.teacher_model
        if args.teacher_provider:
            config.optimizer.teacher.provider = args.teacher_provider

    results = optimize(
        config,
        examples_dir,
        args.output_dir,
        modules_filter=modules_filter,
        strategy_override=args.strategy,
    )

    # Write results to JSON for CI/reporting
    results_path = args.output_dir / "optimization_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
