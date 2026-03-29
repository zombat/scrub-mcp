"""Pydantic models anchoring all pipeline I/O."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModuleInfo(BaseModel):
    """Module-level metadata for module docstring generation."""

    existing_docstring: str | None = None
    imports: list[str] = Field(default_factory=list)
    top_level_names: list[str] = Field(default_factory=list)


class ClassInfo(BaseModel):
    """Extracted class metadata for class docstring generation."""

    name: str
    bases: list[str] = Field(default_factory=list)
    body: str = ""
    method_names: list[str] = Field(default_factory=list)
    existing_docstring: str | None = None
    decorators: list[str] = Field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


class FunctionInfo(BaseModel):
    """Extracted function metadata for DSPy module input."""

    name: str
    signature: str
    body: str
    decorators: list[str] = Field(default_factory=list)
    existing_docstring: str | None = None
    existing_annotations: dict[str, str] = Field(default_factory=dict)
    line_start: int = 0
    line_end: int = 0
    parent_class: str | None = None
    body_line_count: int = 0
    cyclomatic_complexity: int = 1


class GeneratedDocstring(BaseModel):
    """DSPy output: a docstring for a function, class, or module."""

    function_name: str
    docstring: str
    style: str = "google"
    target_type: str = Field(
        default="function",
        description="One of: function, class, module",
    )


class TypeAnnotation(BaseModel):
    """DSPy output: inferred type annotations."""

    function_name: str
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of parameter name to type annotation string",
    )
    return_type: str = "None"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SemanticComment(BaseModel):
    """DSPy output: a semantic comment for a code block."""

    line_number: int
    comment: str
    category: str = Field(
        default="explanation",
        description="One of: explanation, warning, todo, rationale",
    )


class LintResult(BaseModel):
    """Ruff lint output (deterministic, no LLM)."""

    file_path: str
    violations_before: int = 0
    violations_after: int = 0
    auto_fixed: int = 0
    remaining_issues: list[str] = Field(default_factory=list)


class HygieneReport(BaseModel):
    """Full pipeline output for a single file."""

    file_path: str
    lint: LintResult | None = None
    docstrings: list[GeneratedDocstring] = Field(default_factory=list)
    type_annotations: list[TypeAnnotation] = Field(default_factory=list)
    comments: list[SemanticComment] = Field(default_factory=list)
    modified_source: str = ""
    skipped_functions: int = Field(
        default=0,
        description="Functions skipped due to cache hits or diff narrowing.",
    )


# ── Coding tool models ──


class FunctionComplexity(BaseModel):
    """Per-function complexity metrics (deterministic)."""

    function_name: str
    parent_class: str | None = None
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    line_count: int = 0
    max_nesting_depth: int = 0
    parameter_count: int = 0
    needs_simplification: bool = False


class ComplexityReport(BaseModel):
    """File-level complexity analysis."""

    functions: list[FunctionComplexity] = Field(default_factory=list)
    total_functions: int = 0
    flagged_count: int = 0
    avg_cyclomatic: float = 0.0
    avg_cognitive: float = 0.0
    hotspots: list[str] = Field(
        default_factory=list,
        description="Top 5 most complex function names",
    )


class SimplificationSuggestion(BaseModel):
    """DSPy output: a concrete simplification for a complex function."""

    kind: str = Field(
        description="early_return, extract_function, guard_clause, simplify_bool, lookup_table, decompose",
    )
    description: str
    target_lines: list[int] = Field(default_factory=list)
    suggested_code: str = ""
    impact: str = "medium"


class ImportOptResult(BaseModel):
    """Import analysis result."""

    unused_imports: list[str] = Field(default_factory=list)
    potentially_missing: list[str] = Field(default_factory=list)
    import_count: int = 0
    fixed_source: str = ""
    ruff_fixes: int = 0


class MissingImport(BaseModel):
    """DSPy output: a missing import suggestion."""

    import_statement: str
    used_name: str
    line_used: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DeadCodeItem(BaseModel):
    """A single dead code finding."""

    kind: str = Field(
        description="unreachable, unused_var, unused_param, commented_code, redundant_else",
    )
    line_start: int
    line_end: int
    name: str
    suggestion: str = Field(
        default="remove",
        description="remove or review",
    )


class ExtractionSuggestion(BaseModel):
    """DSPy output: a suggested function extraction."""

    extracted_name: str
    extracted_signature: str
    extracted_body: str = ""
    call_replacement: str = ""
    target_lines: list[int] = Field(default_factory=list)
    rationale: str = ""


class RenameSuggestion(BaseModel):
    """DSPy output: a variable/function rename suggestion."""

    old_name: str
    new_name: str
    kind: str = Field(
        description="variable, function, parameter, class",
    )
    rationale: str = ""


class TestGenerationResult(BaseModel):
    """Generated test code for a function or file."""

    test_code: str
    function_names_covered: list[str] = Field(default_factory=list)
    test_count: int = 0


# ── Security models ──


class BanditFinding(BaseModel):
    """A single Bandit security finding."""

    test_id: str = Field(description="Bandit test ID (e.g., B101, B301)")
    test_name: str = Field(description="Human-readable test name")
    severity: str = Field(description="LOW, MEDIUM, or HIGH")
    confidence: str = Field(description="LOW, MEDIUM, or HIGH")
    description: str = ""
    line_number: int = 0
    line_range: list[int] = Field(default_factory=list)
    code: str = ""
    cwe: int = Field(default=0, description="CWE ID if available")
    more_info: str = ""
    file_path: str = ""


class BanditReport(BaseModel):
    """Full Bandit scan result."""

    findings: list[BanditFinding] = Field(default_factory=list)
    total: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_confidence: dict[str, int] = Field(default_factory=dict)
    lines_of_code: int = 0
    error: str = ""


class SecurityTriage(BaseModel):
    """DSPy output: triage result for a Bandit finding.

    Three possible verdicts:
        rewrite: real vulnerability, code needs to change
        nosec: false positive in context, annotate with # nosec + justification
        accept_risk: real but low-impact, document and move on
    """

    test_id: str
    verdict: str = Field(
        default="rewrite",
        description="rewrite, nosec, or accept_risk",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    nosec_justification: str = Field(
        default="",
        description="Justification text for the # nosec inline comment",
    )
    nosec_line: str = Field(
        default="",
        description="The original line with # nosec B{ID} appended",
    )
    remediation: str = ""
    fixed_code: str = ""
    risk_if_ignored: str = ""
    description: str = ""


# ── SBOM models ──


class SBOMComponent(BaseModel):
    """A single dependency in the SBOM."""

    name: str
    version: str = "unspecified"
    purl: str = Field(default="", description="Package URL (pkg:pypi/name@version)")
    license_id: str = Field(default="", description="SPDX license identifier")
    sha256: str = ""
    source: str = Field(
        default="",
        description="Where this dependency was found (pip freeze, pyproject.toml, etc.)",
    )
    optional: bool = False


class SBOMReport(BaseModel):
    """Generated SBOM output."""

    format: str = Field(description="cyclonedx or spdx")
    component_count: int = 0
    components: list[SBOMComponent] = Field(default_factory=list)
    output_json: str = Field(default="", description="Formatted SBOM document")


# ── Vulnerability scan models ──


class VulnFinding(BaseModel):
    """A single vulnerability finding from OSV.dev."""

    vuln_id: str = Field(description="OSV/PYSEC/GHSA vulnerability ID")
    cve_id: str = Field(default="", description="CVE identifier if available")
    ghsa_id: str = Field(default="", description="GitHub Security Advisory ID")
    component_name: str = ""
    component_version: str = ""
    component_purl: str = ""
    severity: str = Field(default="UNKNOWN", description="CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN")
    cvss_score: float = Field(default=0.0, ge=0.0, le=10.0)
    summary: str = ""
    details: str = ""
    fixed_version: str = Field(default="", description="Version that fixes this vulnerability")
    references: list[str] = Field(default_factory=list, description="URLs for more info")
    aliases: list[str] = Field(default_factory=list, description="All known IDs for this vuln")


class VulnReport(BaseModel):
    """Full vulnerability scan result."""

    total_scanned: int = 0
    components_scanned: int = Field(default=0, description="Components with exact versions")
    total_vulnerable: int = Field(default=0, description="Unique components with vulnerabilities")
    total_findings: int = Field(default=0, description="Total vulnerability findings")
    by_severity: dict[str, int] = Field(default_factory=dict)
    findings: list[VulnFinding] = Field(default_factory=list)
    message: str = ""
