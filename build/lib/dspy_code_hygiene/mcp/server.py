"""MCP server exposing code hygiene tools.

Works with Claude Code CLI, VS Code, or any MCP client.

Usage:
    python -m dspy_code_hygiene.mcp.server
    # Or in MCP config: {"command": "python", "args": ["-m", "dspy_code_hygiene.mcp.server"]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from dspy_code_hygiene.config import PipelineConfig, load_config
from dspy_code_hygiene.pipeline import run_pipeline, configure_dspy
from dspy_code_hygiene.models import HygieneReport
from dspy_code_hygiene.modules.hygiene import CommentWriter, DocstringGenerator, TypeAnnotator
from dspy_code_hygiene.tools.linter import run_ruff
from dspy_code_hygiene.tools.parser import extract_functions
from dspy_code_hygiene.tools.complexity import analyze_file_complexity
from dspy_code_hygiene.tools.dead_code import find_dead_code
from dspy_code_hygiene.tools.imports import analyze_imports_deterministic, fix_imports_ruff
from dspy_code_hygiene.tools.security import run_bandit
from dspy_code_hygiene.tools.sbom import generate_sbom
from dspy_code_hygiene.tools.vulnscan import scan_components
from dspy_code_hygiene.modules.coding_tools import (
    BatchTestGenerator,
    ComplexityReducer,
    ExtractFunctionAdvisor,
    MissingImportInferrer,
    RenameAdvisor,
    SecurityRemediationAdvisor,
    TestGenerator,
)

logger = logging.getLogger(__name__)

# Load config once at startup
CONFIG = load_config(Path("config.yaml"))

app = Server("code-hygiene")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Expose pipeline tools to MCP clients."""
    return [
        Tool(
            name="lint_file",
            description=(
                "Run Ruff linting and auto-fix on Python source code. "
                "Deterministic, no LLM. Returns violation counts and fixed source."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="generate_docstrings",
            description=(
                "Generate Google-style docstrings for all functions in Python source. "
                "Uses local LLM via DSPy. Returns source with docstrings inserted."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "style": {
                        "type": "string",
                        "description": "Docstring style: google, numpy, sphinx",
                        "default": "google",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="annotate_types",
            description=(
                "Infer and add type annotations to unannotated Python functions. "
                "Uses local LLM via DSPy. Skips already-annotated parameters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="add_comments",
            description=(
                "Add semantic inline comments to Python functions. "
                "Focuses on non-obvious logic, gotchas, and design rationale. "
                "Uses local LLM via DSPy."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="hygiene_full",
            description=(
                "Run the full code hygiene pipeline: lint (Ruff) -> docstrings -> "
                "types -> comments. Deterministic-first, DSPy for what Ruff can't cover. "
                "Returns modified source and a detailed report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Original file path (for reporting)",
                        "default": "<stdin>",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Steps to run: lint, docstrings, types, comments. Default: all.",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Functions per LLM call. Higher = fewer round trips. Default: 5.",
                        "default": 5,
                    },
                },
                "required": ["source"],
            },
        ),
        # ── Coding tools ──
        Tool(
            name="analyze_complexity",
            description=(
                "Analyze cyclomatic and cognitive complexity of all functions in Python source. "
                "Deterministic (no LLM). Returns per-function metrics, hotspots, and flags "
                "functions that need simplification."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="suggest_simplifications",
            description=(
                "Get concrete refactoring suggestions for complex functions. "
                "Uses local LLM via DSPy. Only processes functions above complexity threshold. "
                "Suggests: early returns, guard clauses, extract function, lookup tables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "function_name": {
                        "type": "string",
                        "description": "Target a specific function. Omit to process all flagged functions.",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="optimize_imports",
            description=(
                "Remove unused imports, sort imports (Ruff), and suggest missing imports. "
                "Deterministic cleanup first (Ruff F401 + isort), then DSPy infers missing "
                "imports from usage patterns. Returns fixed source."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "infer_missing": {
                        "type": "boolean",
                        "description": "Use LLM to infer missing imports. Default: true.",
                        "default": True,
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="generate_tests",
            description=(
                "Generate pytest test stubs for functions in Python source. "
                "Uses local LLM via DSPy. Includes happy path, edge cases, error cases, "
                "and parametrize where appropriate. Returns complete test module."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "module_path": {
                        "type": "string",
                        "description": "Import path for the module under test (e.g., 'mypackage.utils').",
                        "default": "",
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Target a specific function. Omit to generate tests for all.",
                        "default": "",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Functions per LLM call for batch generation. Default: 5.",
                        "default": 5,
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="find_dead_code",
            description=(
                "Find unreachable code, unused variables, redundant else blocks, "
                "and commented-out code. Deterministic (AST-based, no LLM). "
                "Returns list of findings with line numbers and suggestions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="suggest_refactoring",
            description=(
                "Get refactoring suggestions: extract function, rename variables/functions, "
                "identify DRY violations. Uses local LLM via DSPy. "
                "Returns extraction suggestions and rename recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "function_name": {
                        "type": "string",
                        "description": "Target a specific function. Omit to analyze all.",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        ),
        # ── Security tools ──
        Tool(
            name="security_scan",
            description=(
                "Run Bandit static security analysis on Python source code. "
                "Deterministic, no LLM. Detects hardcoded passwords, SQL injection, "
                "shell injection, insecure deserialization, weak crypto, and more. "
                "Returns findings with severity, confidence, CWE IDs, and line numbers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "severity": {
                        "type": "string",
                        "description": "Minimum severity: low, medium, high. Default: low.",
                        "default": "low",
                    },
                    "confidence": {
                        "type": "string",
                        "description": "Minimum confidence: low, medium, high. Default: low.",
                        "default": "low",
                    },
                    "skip": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Bandit test IDs to skip (e.g., ['B101'] to skip assert checks).",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="security_remediate",
            description=(
                "Generate remediation suggestions for Bandit security findings. "
                "Uses local LLM via DSPy. Run security_scan first, then pass findings "
                "to get context-aware fix suggestions with corrected code and risk assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "severity": {
                        "type": "string",
                        "description": "Minimum severity to remediate: low, medium, high. Default: medium.",
                        "default": "medium",
                    },
                },
                "required": ["source"],
            },
        ),
        # ── SBOM tools ──
        Tool(
            name="generate_sbom",
            description=(
                "Generate a Software Bill of Materials (SBOM) in CycloneDX 1.5 or SPDX 2.3 format. "
                "Deterministic, no LLM. Extracts dependencies from pip freeze, pyproject.toml, "
                "requirements.txt, and lock files (poetry, pipenv, uv). "
                "Returns the complete SBOM document as JSON."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_dir": {
                        "type": "string",
                        "description": "Path to the Python project root. Default: current directory.",
                        "default": ".",
                    },
                    "format": {
                        "type": "string",
                        "description": "SBOM format: cyclonedx or spdx. Default: cyclonedx.",
                        "default": "cyclonedx",
                        "enum": ["cyclonedx", "spdx"],
                    },
                    "include_pip": {
                        "type": "boolean",
                        "description": "Include pip freeze output for exact installed versions. Default: true.",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="scan_vulnerabilities",
            description=(
                "Scan project dependencies for known vulnerabilities via OSV.dev. "
                "Deterministic, no LLM. Cross-references SBOM PURLs against PyPI advisories, "
                "GitHub Security Advisories, and NVD. Returns CVE/GHSA IDs, severity, "
                "CVSS scores, and fixed versions (upgrade paths). "
                "Generates SBOM first if needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_dir": {
                        "type": "string",
                        "description": "Path to the Python project root. Default: current directory.",
                        "default": ".",
                    },
                    "severity_filter": {
                        "type": "string",
                        "description": "Minimum severity to include: CRITICAL, HIGH, MEDIUM, LOW. Default: all.",
                        "default": "",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch MCP tool calls to pipeline functions."""
    source = arguments.get("source", "")

    try:
        if name == "lint_file":
            fixed_source, result = run_ruff(source, CONFIG.ruff)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "violations_before": result.violations_before,
                            "violations_after": result.violations_after,
                            "auto_fixed": result.auto_fixed,
                            "remaining_issues": result.remaining_issues,
                            "source": fixed_source,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "generate_docstrings":
            report = run_pipeline(
                source,
                config=CONFIG,
                steps={"docstrings"},
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "docstrings_added": len(report.docstrings),
                            "functions": [d.function_name for d in report.docstrings],
                            "source": report.modified_source,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "annotate_types":
            report = run_pipeline(
                source,
                config=CONFIG,
                steps={"types"},
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "annotations_added": len(report.type_annotations),
                            "functions": [a.function_name for a in report.type_annotations],
                            "source": report.modified_source,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "add_comments":
            report = run_pipeline(
                source,
                config=CONFIG,
                steps={"comments"},
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "comments_added": len(report.comments),
                            "source": report.modified_source,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "hygiene_full":
            steps = set(arguments.get("steps", ["lint", "docstrings", "types", "comments"]))
            file_path = arguments.get("file_path", "<stdin>")

            # Allow per-call batch_size override
            run_config = CONFIG.model_copy()
            if "batch_size" in arguments:
                run_config.batch_size = arguments["batch_size"]

            report = run_pipeline(source, file_path, run_config, steps)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "file_path": report.file_path,
                            "lint": report.lint.model_dump() if report.lint else None,
                            "docstrings_added": len(report.docstrings),
                            "type_annotations_added": len(report.type_annotations),
                            "comments_added": len(report.comments),
                            "source": report.modified_source,
                        },
                        indent=2,
                    ),
                )
            ]

        # ── Coding tool handlers ──

        elif name == "analyze_complexity":
            functions = extract_functions(source)
            report = analyze_file_complexity(functions)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "total_functions": report.total_functions,
                            "flagged_count": report.flagged_count,
                            "avg_cyclomatic": round(report.avg_cyclomatic, 1),
                            "avg_cognitive": round(report.avg_cognitive, 1),
                            "hotspots": report.hotspots,
                            "functions": [f.model_dump() for f in report.functions],
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "suggest_simplifications":
            configure_dspy(CONFIG)
            functions = extract_functions(source)
            target = arguments.get("function_name", "")

            reducer = ComplexityReducer()
            results = {}

            for func in functions:
                if target and func.name != target:
                    continue
                from dspy_code_hygiene.tools.complexity import analyze_complexity
                metrics = analyze_complexity(func)
                if not metrics.needs_simplification and not target:
                    continue

                suggestions = reducer(func)
                results[func.name] = [s.model_dump() for s in suggestions]

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"simplifications": results, "functions_analyzed": len(results)},
                        indent=2,
                    ),
                )
            ]

        elif name == "optimize_imports":
            # Tier 1: deterministic (Ruff)
            fixed_source, ruff_fix_count = fix_imports_ruff(source)
            analysis = analyze_imports_deterministic(fixed_source)
            analysis.fixed_source = fixed_source
            analysis.ruff_fixes = ruff_fix_count

            result_data = {
                "ruff_fixes": ruff_fix_count,
                "unused_imports": analysis.unused_imports,
                "potentially_missing": analysis.potentially_missing,
                "source": fixed_source,
            }

            # Tier 2: DSPy missing import inference
            if arguments.get("infer_missing", True) and analysis.potentially_missing:
                configure_dspy(CONFIG)
                inferrer = MissingImportInferrer()
                from dspy_code_hygiene.tools.parser import extract_module_info
                mod_info = extract_module_info(fixed_source)
                missing = inferrer(fixed_source, mod_info.imports)
                result_data["inferred_imports"] = [m.model_dump() for m in missing]

            return [
                TextContent(type="text", text=json.dumps(result_data, indent=2))
            ]

        elif name == "generate_tests":
            configure_dspy(CONFIG)
            functions = extract_functions(source)
            target = arguments.get("function_name", "")
            module_path = arguments.get("module_path", "")
            bs = arguments.get("batch_size", 5)

            if target:
                funcs = [f for f in functions if f.name == target]
            else:
                funcs = functions

            if not funcs:
                return [TextContent(type="text", text=json.dumps({"error": "No functions found"}))]

            if len(funcs) <= bs and bs > 1:
                # Single batch
                gen = BatchTestGenerator()
                result = gen(funcs, module_path)
            elif bs > 1:
                # Multiple batches, combine
                from dspy_code_hygiene.tools.utils import batch as batch_fn
                gen = BatchTestGenerator()
                all_code = []
                all_covered = []
                total_tests = 0
                for func_batch in batch_fn(funcs, bs):
                    r = gen(func_batch, module_path)
                    all_code.append(r.test_code)
                    all_covered.extend(r.function_names_covered)
                    total_tests += r.test_count
                result = type(r)(
                    test_code="\n\n".join(all_code),
                    function_names_covered=all_covered,
                    test_count=total_tests,
                )
            else:
                # Sequential
                gen = TestGenerator()
                all_code = []
                all_covered = []
                total_tests = 0
                for func in funcs:
                    r = gen(func, module_path)
                    all_code.append(r.test_code)
                    all_covered.extend(r.function_names_covered)
                    total_tests += r.test_count
                result = type(r)(
                    test_code="\n\n".join(all_code),
                    function_names_covered=all_covered,
                    test_count=total_tests,
                )

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "test_count": result.test_count,
                            "functions_covered": result.function_names_covered,
                            "test_code": result.test_code,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "find_dead_code":
            items = find_dead_code(source)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "dead_code_count": len(items),
                            "items": [i.model_dump() for i in items],
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "suggest_refactoring":
            configure_dspy(CONFIG)
            functions = extract_functions(source)
            target = arguments.get("function_name", "")

            extract_advisor = ExtractFunctionAdvisor()
            rename_advisor = RenameAdvisor()
            results = {}

            for func in functions:
                if target and func.name != target:
                    continue

                extractions = extract_advisor(func)
                renames = rename_advisor(func)

                if extractions or renames:
                    results[func.name] = {
                        "extractions": [e.model_dump() for e in extractions],
                        "renames": [r.model_dump() for r in renames],
                    }

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"functions_analyzed": len(results), "refactoring": results},
                        indent=2,
                    ),
                )
            ]

        # ── Security tool handlers ──

        elif name == "security_scan":
            severity = arguments.get("severity", "low")
            confidence = arguments.get("confidence", "low")
            skips = arguments.get("skip", None)

            report = run_bandit(source, severity, confidence, skips)

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "total_findings": report.total,
                            "by_severity": report.by_severity,
                            "by_confidence": report.by_confidence,
                            "lines_of_code": report.lines_of_code,
                            "findings": [f.model_dump() for f in report.findings],
                            "error": report.error or None,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "security_remediate":
            severity = arguments.get("severity", "medium")

            # Step 1: Bandit scan (deterministic)
            report = run_bandit(source, severity_threshold=severity)

            if not report.findings:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"message": "No security findings at this severity level.", "total": 0},
                            indent=2,
                        ),
                    )
                ]

            # Step 2: Detect file context for better triage
            source_lines = source.splitlines()
            file_context_parts = []
            if any("import pytest" in line or "def test_" in line for line in source_lines[:50]):
                file_context_parts.append("This is a test file.")
            if any("if __name__" in line for line in source_lines):
                file_context_parts.append("This file has a CLI entrypoint.")
            if any("app = " in line or "Flask(" in line or "FastAPI(" in line for line in source_lines[:30]):
                file_context_parts.append("This is a web application.")
            file_context = " ".join(file_context_parts) if file_context_parts else "General Python module."

            # Step 3: DSPy triage + remediation for each finding (local LLM)
            configure_dspy(CONFIG)
            advisor = SecurityRemediationAdvisor()

            triage_results = {"rewrite": [], "nosec": [], "accept_risk": []}

            for finding in report.findings:
                start = max(0, finding.line_number - 5)
                end = min(len(source_lines), finding.line_number + 5)
                context = "\n".join(source_lines[start:end])

                try:
                    triage = advisor(
                        test_id=finding.test_id,
                        test_name=finding.test_name,
                        issue_description=finding.description,
                        flagged_code=finding.code,
                        surrounding_context=context,
                        file_context=file_context,
                        severity=finding.severity,
                    )
                    triage_results[triage.verdict].append(triage.model_dump())
                except Exception:
                    logger.exception(
                        "Triage failed for %s at line %d", finding.test_id, finding.line_number
                    )
                    triage_results["rewrite"].append({
                        "test_id": finding.test_id,
                        "verdict": "rewrite",
                        "confidence": 0.0,
                        "description": finding.description,
                        "rationale": "Triage failed, defaulting to rewrite for safety.",
                    })

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "total_findings": report.total,
                            "summary": {
                                "rewrite": len(triage_results["rewrite"]),
                                "nosec": len(triage_results["nosec"]),
                                "accept_risk": len(triage_results["accept_risk"]),
                            },
                            "rewrite": triage_results["rewrite"],
                            "nosec": triage_results["nosec"],
                            "accept_risk": triage_results["accept_risk"],
                        },
                        indent=2,
                    ),
                )
            ]

        # ── SBOM handler ──

        elif name == "generate_sbom":
            project_dir = Path(arguments.get("project_dir", "."))
            sbom_format = arguments.get("format", "cyclonedx")
            include_pip = arguments.get("include_pip", True)

            report = generate_sbom(
                project_dir=project_dir,
                format=sbom_format,
                include_pip=include_pip,
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "format": report.format,
                            "component_count": report.component_count,
                            "components": [c.model_dump() for c in report.components],
                            "sbom": json.loads(report.output_json),
                        },
                        indent=2,
                    ),
                )
            ]

        # ── Vulnerability scan handler ──

        elif name == "scan_vulnerabilities":
            project_dir = Path(arguments.get("project_dir", "."))
            severity_filter = arguments.get("severity_filter", "").upper()

            # Step 1: Generate SBOM to get components
            sbom = generate_sbom(project_dir=project_dir, format="cyclonedx")

            if not sbom.components:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"message": "No dependencies found to scan.", "total_scanned": 0},
                            indent=2,
                        ),
                    )
                ]

            # Step 2: Scan against OSV.dev
            vuln_report = scan_components(sbom.components)

            # Step 3: Filter by severity if requested
            findings = vuln_report.findings
            if severity_filter:
                severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
                if severity_filter in severity_order:
                    cutoff = severity_order.index(severity_filter)
                    allowed = set(severity_order[: cutoff + 1])
                    findings = [f for f in findings if f.severity in allowed]

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "total_scanned": vuln_report.total_scanned,
                            "components_scanned": vuln_report.components_scanned,
                            "total_vulnerable": vuln_report.total_vulnerable,
                            "total_findings": len(findings),
                            "by_severity": vuln_report.by_severity,
                            "findings": [f.model_dump() for f in findings],
                        },
                        indent=2,
                    ),
                )
            ]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.exception("Tool %s failed", name)
        return [TextContent(type="text", text=f"Error in {name}: {e}")]


async def main() -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
