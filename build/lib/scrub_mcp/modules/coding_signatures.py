"""DSPy signatures for coding tools: complexity, imports, tests, refactoring.

Same pattern as hygiene signatures: deterministic tools handle what they
can, DSPy fills the gaps. Each signature is independently optimizable.
"""

from __future__ import annotations

import dspy


# ── Complexity Reducer ──


class ComplexityAnalysisSignature(dspy.Signature):
    """Analyze a complex Python function and suggest concrete simplifications.

    Given a function with high cyclomatic complexity or excessive length,
    identify specific refactoring opportunities. Focus on:
    - Early returns to reduce nesting
    - Guard clauses
    - Extract helper functions (name them)
    - Replace conditionals with polymorphism or lookup tables
    - Simplify boolean expressions
    - Break long functions into smaller units

    Output as JSON list of suggestions, ordered by impact (highest first).
    """

    function_signature: str = dspy.InputField(desc="The function def line")
    function_body: str = dspy.InputField(desc="The full function body")
    cyclomatic_complexity: int = dspy.InputField(desc="Current cyclomatic complexity score")
    line_count: int = dspy.InputField(desc="Number of lines in the function body")
    suggestions_json: str = dspy.OutputField(
        desc=(
            'JSON list of {"kind": "early_return|extract_function|guard_clause|'
            'simplify_bool|lookup_table|decompose", "description": str, '
            '"target_lines": [start, end], "suggested_code": str, "impact": "high|medium|low"}'
        )
    )


# ── Test Generation ──


class TestGenerationSignature(dspy.Signature):
    """Generate pytest test stubs for a Python function.

    Given the function signature, body, and docstring, produce a complete
    pytest test module. Include:
    - Happy path test(s)
    - Edge cases (empty input, None, boundary values)
    - Error cases (expected exceptions from Raises docstring)
    - Parametrize decorator for multiple inputs where appropriate
    - Fixtures for complex setup

    Use descriptive test names: test_<function>_<scenario>.
    Use assert statements (not unittest-style).
    Output the full test code as a string (no triple backticks).
    """

    function_signature: str = dspy.InputField(desc="The function def line")
    function_body: str = dspy.InputField(desc="The function implementation")
    docstring: str = dspy.InputField(desc="The function's docstring", default="")
    parent_class: str = dspy.InputField(desc="Enclosing class name, if a method", default="")
    module_path: str = dspy.InputField(desc="Import path for the function under test", default="")
    test_code: str = dspy.OutputField(desc="Complete pytest test code (no markdown fences)")


class BatchTestGenerationSignature(dspy.Signature):
    """Generate pytest test stubs for MULTIPLE functions in one call.

    Input is a JSON list of functions with name, signature, body, docstring.
    Output a single test module covering all functions. Group tests by
    class (TestClassName) for methods, or as top-level test functions.
    Use parametrize where appropriate. Include edge cases.
    Output format: complete Python test file as a string (no markdown fences).
    """

    functions_json: str = dspy.InputField(
        desc='JSON list of {"name": str, "signature": str, "body": str, "docstring": str, "parent_class": str}'
    )
    module_path: str = dspy.InputField(desc="Import path for the module under test", default="")
    test_code: str = dspy.OutputField(desc="Complete pytest test module (no markdown fences)")


# ── Refactoring ──


class ExtractFunctionSignature(dspy.Signature):
    """Identify code blocks that should be extracted into separate functions.

    Analyze the function body for:
    - Repeated patterns (DRY violations)
    - Deeply nested blocks that could be standalone
    - Logically independent sections doing different things
    - Long expressions that deserve a named function

    For each extraction, provide the suggested function name, signature,
    body, and where it would be called in the original.
    """

    function_signature: str = dspy.InputField(desc="The original function def line")
    function_body: str = dspy.InputField(desc="The original function body")
    extractions_json: str = dspy.OutputField(
        desc=(
            'JSON list of {"extracted_name": str, "extracted_signature": str, '
            '"extracted_body": str, "call_replacement": str, '
            '"target_lines": [start, end], "rationale": str}'
        )
    )


class RenameSuggestionSignature(dspy.Signature):
    """Suggest better names for poorly named variables, functions, or parameters.

    Focus on:
    - Single-letter variables (except i, j, k in loops and x, y in math)
    - Abbreviations that hurt readability
    - Names that don't reflect the value's purpose
    - Boolean variables/params that should start with is_/has_/can_/should_
    - Functions that don't describe their action (verb + noun)

    Output as JSON list of rename suggestions.
    """

    code_block: str = dspy.InputField(desc="The code to analyze")
    context: str = dspy.InputField(desc="Function/class context", default="")
    renames_json: str = dspy.OutputField(
        desc=(
            'JSON list of {"old_name": str, "new_name": str, '
            '"kind": "variable|function|parameter|class", "rationale": str}'
        )
    )


class DeadCodeSignature(dspy.Signature):
    """Identify dead or unreachable code in a Python source file.

    Look for:
    - Unreachable code after return/raise/break/continue
    - Unused variables (assigned but never read)
    - Unused imports (already handled by Ruff, but catch what it misses)
    - Unused function parameters
    - Commented-out code blocks that should be removed
    - Redundant else after return

    Output as JSON list.
    """

    source_code: str = dspy.InputField(desc="Full Python source file")
    dead_code_json: str = dspy.OutputField(
        desc=(
            'JSON list of {"kind": "unreachable|unused_var|unused_param|commented_code|redundant_else", '
            '"line_start": int, "line_end": int, "name": str, "suggestion": "remove|review"}'
        )
    )


# ── Import Optimizer ──
# Note: Most import work is deterministic (Ruff I + autoflake).
# The DSPy signature handles the edge case: suggesting imports that
# are missing based on usage patterns the linter can't infer.


class MissingImportSignature(dspy.Signature):
    """Identify missing imports in Python source code.

    Analyze the code for names that are used but not imported or defined
    locally. Common patterns:
    - Using typing constructs without importing them
    - Using stdlib modules referenced by name but not imported
    - Third-party library usage without import

    Only suggest imports you are confident about. Do NOT guess.
    Output as JSON list.
    """

    source_code: str = dspy.InputField(desc="Full Python source file")
    existing_imports: str = dspy.InputField(desc="Current import statements, newline-separated")
    missing_imports_json: str = dspy.OutputField(
        desc='JSON list of {"import_statement": str, "used_name": str, "line_used": int, "confidence": float}'
    )


# ── Security Triage + Remediation ──
# Bandit handles detection (deterministic). DSPy triages each finding
# first (is this a real issue or a false positive in context?), then
# generates the appropriate response: rewrite, nosec, or accept risk.

# Known false-positive-heavy Bandit tests for context
_COMMON_FALSE_POSITIVES = {
    "B101": "assert used in test files or debug-only paths",
    "B311": "random.random() for non-cryptographic purposes (shuffling, sampling, jitter)",
    "B603": "subprocess with static/trusted arguments",
    "B404": "import subprocess (flagged just for importing)",
    "B608": "SQL string building with trusted/validated inputs",
    "B108": "hardcoded /tmp path in controlled environments",
    "B105": "hardcoded password-like strings that are actually config keys or defaults",
    "B607": "subprocess call with partial path in controlled PATH environments",
}


class SecurityTriageSignature(dspy.Signature):
    """Triage a Bandit security finding: real issue or false positive?

    Analyze the flagged code IN CONTEXT to determine if this is:
    1. A real security vulnerability that needs code changes
    2. A false positive that should get a # nosec annotation
    3. An acceptable risk that should be documented

    Common false positive patterns:
    - B101 (assert): safe in test files, pytest, and debug-only paths
    - B311 (random): safe for non-crypto uses (UI, shuffling, jitter, sampling)
    - B603 (subprocess): safe with static arguments or validated input
    - B404 (import subprocess): always a false positive, importing is not a vulnerability
    - B608 (SQL): safe with parameterized queries or ORM-generated SQL
    - B108 (/tmp): safe in containerized/ephemeral environments
    - B105 (hardcoded password): false positive when the string is a config key name, not a secret

    BE CONSERVATIVE: if the code is clearly safe in context, recommend nosec.
    Do NOT recommend rewriting safe code just because Bandit flagged it.

    Output as JSON:
    {
        "verdict": "rewrite|nosec|accept_risk",
        "confidence": float (0.0-1.0),
        "rationale": "why this verdict",
        "nosec_justification": "justification for # nosec comment if verdict is nosec",
        "remediation": "what to change if verdict is rewrite",
        "fixed_code": "corrected code if verdict is rewrite",
        "risk_if_ignored": "impact if this is a real issue and is ignored"
    }
    """

    test_id: str = dspy.InputField(desc="Bandit test ID (e.g., B101, B301)")
    test_name: str = dspy.InputField(desc="Human-readable test name", default="")
    issue_description: str = dspy.InputField(desc="Bandit's description of the issue")
    severity: str = dspy.InputField(desc="LOW, MEDIUM, or HIGH")
    flagged_code: str = dspy.InputField(desc="The exact code snippet Bandit flagged")
    surrounding_context: str = dspy.InputField(
        desc="Surrounding function/class (5-10 lines each direction)", default=""
    )
    file_context: str = dspy.InputField(
        desc="File-level hints: is this a test file? CLI tool? Library? Server?", default=""
    )
    false_positive_hint: str = dspy.InputField(
        desc="Known false positive pattern for this test ID, if any", default=""
    )
    triage_json: str = dspy.OutputField(
        desc=(
            'JSON: {"verdict": "rewrite|nosec|accept_risk", "confidence": float, '
            '"rationale": str, "nosec_justification": str, "remediation": str, '
            '"fixed_code": str, "risk_if_ignored": str}'
        )
    )
