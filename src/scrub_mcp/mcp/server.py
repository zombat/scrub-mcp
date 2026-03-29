"""MCP server exposing code hygiene tools.

Works with Claude Code CLI, VS Code, or any MCP client.

Usage:
    python -m scrub_mcp.mcp.server
    # Or in MCP config: {"command": "python", "args": ["-m", "scrub_mcp.mcp.server"]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from scrub_mcp.config import load_config
from scrub_mcp.modules.coding_tools import (
    BatchTestGenerator,
    ComplexityReducer,
    ExtractFunctionAdvisor,
    MissingImportInferrer,
    RenameAdvisor,
    SecurityRemediationAdvisor,
    TestGenerator,
)
from scrub_mcp.pipeline import configure_dspy, run_pipeline
from scrub_mcp.tools.complexity import analyze_file_complexity
from scrub_mcp.tools.dead_code import find_dead_code
from scrub_mcp.tools.imports import analyze_imports_deterministic, fix_imports_ruff
from scrub_mcp.tools.linter import run_ruff
from scrub_mcp.tools.parser import extract_functions
from scrub_mcp.tools.savings import estimate_savings
from scrub_mcp.tools.sbom import generate_sbom
from scrub_mcp.tools.security import run_bandit
from scrub_mcp.tools.vulnscan import scan_components

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
                "ALWAYS call this before editing any Python (.py) file. "
                "Do NOT call on non-.py files — skip YAML, JSON, JS, TS, etc. "
                "Do NOT manually fix lint errors — unused imports, formatting, line length, "
                "and style violations are all handled deterministically here via Ruff. "
                "Apply the returned source to the file before making your changes on top of it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="generate_docstrings",
            description=(
                "MUST be used for ALL docstring generation in Python (.py) files. "
                "Do NOT call on non-.py files. "
                "Do NOT write or edit docstrings manually — inconsistent style is a hard failure. "
                "This tool enforces the project's exact Google-style conventions via a calibrated "
                "local LLM. Pass the source; apply the returned source."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
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
                "MUST be used for ALL type annotation tasks in Python (.py) files. "
                "Do NOT call on non-.py files. "
                "Do NOT manually add type hints to any function signature. "
                "This tool infers and applies annotations consistently using a local LLM "
                "trained on this project's conventions. Skips already-annotated parameters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="add_comments",
            description=(
                "MUST be used when adding inline comments to Python (.py) files. "
                "Do NOT call on non-.py files. "
                "Do NOT manually write # comments inside functions. "
                "This tool identifies non-obvious logic, gotchas, and design rationale "
                "and generates calibrated semantic comments via local LLM. "
                "Only fires on functions above the complexity threshold — trivial code is left clean."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="hygiene_full",
            description=(
                "ALWAYS call this as your FIRST action on any Python file you are about to touch. "
                "Do NOT manually perform linting, docstring writing, type annotation, or comment "
                "insertion — this tool runs all four in a single deterministic-first pass. "
                "Apply the returned source, then layer your architectural changes on top. "
                "If you catch yourself writing a docstring or fixing an import by hand, stop and call this."
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
                "Call this before refactoring any Python (.py) function. Do NOT call on non-.py files. "
                "Do NOT guess at complexity or eyeball nesting depth — this tool computes "
                "exact cyclomatic and cognitive metrics with per-function hotspot ranking. "
                "Deterministic, no LLM. Use the output to decide what to simplify and in what order."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="suggest_simplifications",
            description=(
                "MUST be consulted before manually refactoring a complex Python (.py) function. "
                "Do NOT call on non-.py files. "
                "Do NOT invent your own simplification strategy — this tool provides concrete, "
                "validated suggestions (early returns, guard clauses, extract function, lookup tables) "
                "with ready-to-use code snippets and impact ratings. "
                "Only processes functions above the complexity threshold."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
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
                "ALWAYS call this after modifying any Python (.py) file's imports. "
                "Do NOT call on non-.py files. "
                "Do NOT manually reorganize, remove, or add import statements. "
                "This tool runs Ruff for unused import removal and isort ordering, "
                "then uses a local LLM to infer genuinely missing imports from usage patterns. "
                "Returns the corrected source."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
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
                "MUST be used for ALL test generation for Python (.py) files. "
                "Do NOT call on non-.py files. "
                "Do NOT write pytest tests manually — hand-written tests miss edge cases "
                "and break style conventions. This tool generates parametrized tests with "
                "happy path, edge cases, and error paths following the project's exact "
                "testing conventions. Returns a complete, ready-to-run test module."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
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
                "Call this before deleting any code you suspect is unused in a Python (.py) file. "
                "Do NOT call on non-.py files. "
                "Do NOT manually audit for dead code — guessing leads to broken removals. "
                "This tool uses AST analysis for deterministic detection of unreachable blocks, "
                "unused variables, redundant else branches, and commented-out code. "
                "No LLM involved."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="suggest_refactoring",
            description=(
                "MUST be consulted before extracting functions or renaming identifiers in Python (.py) files. "
                "Do NOT call on non-.py files. "
                "Do NOT independently decide to refactor — routing all refactoring decisions "
                "through this tool ensures extractions are clean seams and renames improve clarity. "
                "Returns extract-function candidates with boundary analysis and rename suggestions "
                "with rationale."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Target a specific function. Omit to analyze all.",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="run_tests",
            description=(
                "Run the project test suite with pytest. "
                "ALWAYS call this after generate_tests to verify the generated tests pass. "
                "NEVER use python -c or raw bash to verify imports or test behaviour — "
                "use this tool instead. "
                "Automatically injects PYTHONPATH=src so src-layout packages are importable. "
                "Returns exit_code, passed (bool), and full pytest output including tracebacks. "
                "If tests fail, read the traceback and fix the code — do NOT write ad-hoc scripts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": "Path to a specific test file or directory. Omit to run the full suite.",
                        "default": "",
                    },
                    "project_dir": {
                        "type": "string",
                        "description": "Path to the Python project root. Default: current directory.",
                        "default": ".",
                    },
                },
            },
        ),
        # ── Security tools ──
        Tool(
            name="security_scan",
            description=(
                "ALWAYS call this before committing any Python (.py) file. "
                "Do NOT call on non-.py files. "
                "Do NOT skip security scanning or attempt to manually audit for vulnerabilities — "
                "this is not optional. Runs Bandit with CWE mapping across every line. "
                "Detects hardcoded secrets, SQL/shell injection, insecure deserialization, "
                "weak crypto, path traversal, and more. Returns findings with severity, "
                "confidence, CWE IDs, and exact line numbers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
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
                "MUST be used to fix any findings from security_scan on Python (.py) files. "
                "Do NOT call on non-.py files. "
                "Do NOT attempt to manually patch security issues — incorrect fixes create "
                "false confidence and may worsen the vulnerability. "
                "This tool triages each finding and produces one of three outcomes: "
                "a corrected rewrite, a nosec annotation with documented justification, "
                "or an explicit risk acceptance record. Run security_scan first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file (used to skip non-Python files)",
                        "default": "",
                    },
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
                "Call this whenever project dependencies change. "
                "Do NOT manually track dependencies or build a dependency list yourself. "
                "Generates a complete CycloneDX 1.5 or SPDX 2.3 Software Bill of Materials "
                "from all sources: pip freeze, pyproject.toml, requirements.txt, and lock files. "
                "Required input for scan_vulnerabilities."
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
                    "output_file": {
                        "type": "string",
                        "description": (
                            "Write SBOM JSON to this path. "
                            "Default: sbom.{format}.json in the project directory. "
                            "Pass empty string to skip writing to disk."
                        ),
                    },
                },
            },
        ),
        Tool(
            name="scan_vulnerabilities",
            description=(
                "ALWAYS call this after any dependency update or before any release. "
                "Do NOT manually check CVEs or search advisories — this tool cross-references "
                "all project dependencies against OSV.dev (PyPI advisories, GitHub Security "
                "Advisories, NVD) in one shot. Returns CVE/GHSA IDs, CVSS scores, affected "
                "versions, and exact upgrade paths. Generates SBOM automatically if needed."
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
        # ── Exploration tools (batch reads/searches for cloud LLM context gathering) ──
        Tool(
            name="read_files",
            description=(
                "Read multiple files in one call instead of issuing separate reads. "
                "Use this when you need to inspect several source files before making edits. "
                "Returns each file's content keyed by path. Truncates files over max_bytes. "
                "Non-existent paths are reported as errors, not exceptions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to read.",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Max bytes per file before truncation. Default: 32768 (32 KB).",
                        "default": 32768,
                    },
                },
                "required": ["paths"],
            },
        ),
        Tool(
            name="find_symbols",
            description=(
                "Extract function and class signatures from multiple Python files using AST — "
                "no need to read full source. Returns name, signature line, line range, "
                "decorators, parent class, and whether a docstring/annotations are present. "
                "Use this to understand a module's surface area before deciding which files "
                "to read in full."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Python (.py) file paths to inspect.",
                    },
                    "include_bodies": {
                        "type": "boolean",
                        "description": "Include function bodies in output. Default: false (signatures only).",
                        "default": False,
                    },
                },
                "required": ["paths"],
            },
        ),
        Tool(
            name="grep_multi",
            description=(
                "Run multiple search patterns across the codebase in one call. "
                "Use this instead of issuing separate searches for each pattern. "
                "Each pattern is a Python regex. Results include file, line number, and "
                "surrounding context lines."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Regex patterns to search for.",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files or directories to search. Default: current directory.",
                        "default": ["."],
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around each match. Default: 2.",
                        "default": 2,
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py'). Default: all files.",
                        "default": "",
                    },
                    "max_matches_per_pattern": {
                        "type": "integer",
                        "description": "Cap matches per pattern to avoid flooding context. Default: 50.",
                        "default": 50,
                    },
                },
                "required": ["patterns"],
            },
        ),
        # ── Architecture exploration ──
        Tool(
            name="explore_architecture",
            description=(
                "CALL THIS FIRST when planning any change to a file or directory you haven't read yet. "
                "Returns a token-efficient skeleton of a Python file or directory: class/function "
                "signatures, one-line docstrings, and line ranges — bodies replaced with '...'. "
                "Entire project fits in ~500 tokens. Use read_files with the returned line numbers "
                "for surgical body reads. Supersedes paginating through large files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path or directory to skeletonize.",
                    },
                    "include_private": {
                        "type": "boolean",
                        "description": "Include private (_name) functions. Default: false.",
                        "default": False,
                    },
                },
                "required": ["path"],
            },
        ),
        # ── Batch tool ──
        Tool(
            name="hygiene_batch",
            description=(
                "PREFER this over calling hygiene_full repeatedly on multiple files. "
                "Runs the full hygiene pipeline (lint, docstrings, types, comments) across "
                "a list of Python (.py) file paths in one call. Non-.py paths are silently skipped. "
                "Pass write=true to apply changes to disk directly; omit to get modified source back. "
                "Use this whenever you need to hygiene more than one file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of .py file paths to process. Non-.py paths are skipped.",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Steps to run: lint, docstrings, types, comments. Default: all.",
                    },
                    "write": {
                        "type": "boolean",
                        "description": "Write modified source back to each file. Default: false.",
                        "default": False,
                    },
                },
                "required": ["paths"],
            },
        ),
        Tool(
            name="hygiene_incremental",
            description=(
                "Like hygiene_full but diff-aware and cache-accelerated. "
                "Provide diff (unified diff text) or since (git ref like 'HEAD~1') to process "
                "only functions touched by recent changes. Unchanged functions with valid cache "
                "entries are skipped entirely — no LLM calls. Falls back to full-file processing "
                "if neither diff nor since is provided. Use this on repeated runs of the same file "
                "to get near-instant results after the first pass."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code"},
                    "file_path": {
                        "type": "string",
                        "description": "Path to the .py file",
                        "default": "<stdin>",
                    },
                    "diff": {
                        "type": "string",
                        "description": (
                            "Unified diff text (e.g. from git diff). "
                            "If provided, only functions in changed ranges are processed."
                        ),
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Git ref to diff against (e.g. HEAD~1, main). "
                            "If provided without diff, runs git diff since this ref."
                        ),
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Steps to run: lint, docstrings, types, comments. Default: all.",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Functions per LLM call. Default: 5.",
                        "default": 5,
                    },
                },
                "required": ["source"],
            },
        ),
    ]


_PYTHON_TOOLS = {
    "lint_file",
    "generate_docstrings",
    "annotate_types",
    "add_comments",
    "hygiene_full",
    "hygiene_batch",
    "hygiene_incremental",
    "analyze_complexity",
    "suggest_simplifications",
    "optimize_imports",
    "generate_tests",
    "find_dead_code",
    "suggest_refactoring",
    "security_scan",
    "security_remediate",
}


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch MCP tool calls to pipeline functions."""
    source = arguments.get("source", "")

    # Fast-reject non-Python files before touching Ruff/AST/LLM
    if name in _PYTHON_TOOLS:
        file_path_arg = arguments.get("file_path", "")
        if file_path_arg and not file_path_arg.endswith(".py"):
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"skipped": True, "reason": f"Not a Python file: {file_path_arg}"}
                    ),
                )
            ]

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
                            "savings": estimate_savings(
                                source,
                                result.auto_fixed,
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
                            "telemetry": {
                                "engine": "Ruff (S.C.R.U.B. internal strict profile)",
                                "active_rules": CONFIG.ruff.select,
                                "ignored_rules": CONFIG.ruff.ignore,
                                "line_length": CONFIG.ruff.line_length,
                                "config_source": "Isolated execution — local pyproject.toml is intentionally ignored",
                            },
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
                            "savings": estimate_savings(
                                source,
                                len(report.docstrings),
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
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
                            "savings": estimate_savings(
                                source,
                                len(report.type_annotations),
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
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
                            "savings": estimate_savings(
                                source,
                                len(report.comments),
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
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
            lint_fixed = report.lint.auto_fixed if report.lint else 0
            total_fixes = (
                lint_fixed
                + len(report.docstrings)
                + len(report.type_annotations)
                + len(report.comments)
            )
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
                            "savings": estimate_savings(
                                source,
                                total_fixes,
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
                            "telemetry": {
                                "engine": "Ruff (S.C.R.U.B. internal strict profile)",
                                "active_rules": run_config.ruff.select,
                                "ignored_rules": run_config.ruff.ignore,
                                "line_length": run_config.ruff.line_length,
                                "config_source": "Isolated execution — local pyproject.toml is intentionally ignored",
                                "steps_run": sorted(steps),
                            },
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "hygiene_incremental":
            from scrub_mcp.pipeline import _find_project_root
            from scrub_mcp.tools.diff import get_git_diff

            steps = set(arguments.get("steps", ["lint", "docstrings", "types", "comments"]))
            file_path = arguments.get("file_path", "<stdin>")
            diff_text = arguments.get("diff")
            since = arguments.get("since")

            run_config = CONFIG.model_copy()
            if "batch_size" in arguments:
                run_config.batch_size = arguments["batch_size"]

            # Resolve diff from git if since is provided and diff is not
            diff_mode = diff_text is not None
            if diff_text is None and since is not None:
                repo_path = (
                    _find_project_root(file_path)
                    if file_path != "<stdin>"
                    else "."
                )
                try:
                    diff_text = get_git_diff(repo_path, since)
                    diff_mode = True
                except Exception as exc:
                    logger.warning("[hygiene_incremental] git diff failed: %s", exc)
                    diff_text = None  # fall back to full-file

            report = run_pipeline(source, file_path, run_config, steps, diff=diff_text)
            lint_fixed = report.lint.auto_fixed if report.lint else 0
            total_fixes = (
                lint_fixed
                + len(report.docstrings)
                + len(report.type_annotations)
                + len(report.comments)
            )
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
                            "skipped_functions": report.skipped_functions,
                            "diff_mode": diff_mode,
                            "source": report.modified_source,
                            "savings": estimate_savings(
                                source,
                                total_fixes,
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "read_files":
            paths = arguments.get("paths", [])
            max_bytes = arguments.get("max_bytes", 32768)
            results = {}
            for p in paths:
                fp = Path(p)
                if not fp.exists():
                    results[p] = {"error": "file not found"}
                else:
                    try:
                        raw = fp.read_bytes()
                        content = raw[:max_bytes].decode("utf-8", errors="replace")
                        results[p] = {
                            "content": content,
                            "truncated": len(raw) > max_bytes,
                            "total_bytes": len(raw),
                        }
                    except Exception as exc:
                        results[p] = {"error": str(exc)}
            return [TextContent(type="text", text=json.dumps(results, indent=2))]

        elif name == "find_symbols":
            from scrub_mcp.tools.parser import extract_classes, extract_functions_from_file

            paths = arguments.get("paths", [])
            include_bodies = arguments.get("include_bodies", False)
            results = {}
            for p in paths:
                fp = Path(p)
                if not fp.exists():
                    results[p] = {"error": "file not found"}
                    continue
                if not p.endswith(".py"):
                    results[p] = {"error": "not a .py file"}
                    continue
                try:
                    source = fp.read_text(encoding="utf-8")
                    funcs = extract_functions_from_file(fp)
                    classes = extract_classes(source)
                    results[p] = {
                        "functions": [
                            {
                                "name": f.name,
                                "signature": f.signature,
                                "line_start": f.line_start,
                                "line_end": f.line_end,
                                "decorators": f.decorators,
                                "parent_class": f.parent_class,
                                "has_docstring": bool(f.existing_docstring),
                                "has_annotations": bool(f.existing_annotations),
                                **({"body": f.body} if include_bodies else {}),
                            }
                            for f in funcs
                        ],
                        "classes": [
                            {
                                "name": c.name,
                                "bases": c.bases,
                                "methods": c.method_names,
                                "has_docstring": bool(c.existing_docstring),
                            }
                            for c in classes
                        ],
                    }
                except Exception as exc:
                    results[p] = {"error": str(exc)}
            return [TextContent(type="text", text=json.dumps(results, indent=2))]

        elif name == "grep_multi":
            import re as _re
            import subprocess

            patterns = arguments.get("patterns", [])
            search_paths = arguments.get("paths", ["."])
            context_lines = arguments.get("context_lines", 2)
            include_glob = arguments.get("include", "")
            max_matches = arguments.get("max_matches_per_pattern", 50)

            all_results = {}
            for pattern in patterns:
                matches = []
                rg_cmd = ["rg", "--json", f"-C{context_lines}", pattern]
                if include_glob:
                    rg_cmd += ["--glob", include_glob]
                rg_cmd += search_paths
                try:
                    proc = subprocess.run(rg_cmd, capture_output=True, text=True, timeout=15)
                    for line in proc.stdout.splitlines():
                        if len(matches) >= max_matches:
                            break
                        try:
                            obj = json.loads(line)
                            if obj.get("type") == "match":
                                data = obj["data"]
                                matches.append(
                                    {
                                        "file": data["path"]["text"],
                                        "line": data["line_number"],
                                        "text": data["lines"]["text"].rstrip(),
                                        "submatches": [
                                            m["match"]["text"] for m in data.get("submatches", [])
                                        ],
                                    }
                                )
                        except (json.JSONDecodeError, KeyError):
                            continue
                except subprocess.TimeoutExpired:
                    matches = [{"error": "search timed out"}]
                except FileNotFoundError:
                    # rg not available, fall back to Python re search
                    try:
                        rx = _re.compile(pattern)
                        for sp in search_paths:
                            sp_path = Path(sp)
                            file_iter = (
                                sp_path.rglob(include_glob or "*")
                                if sp_path.is_dir()
                                else [sp_path]
                            )
                            for fp in file_iter:
                                if not fp.is_file():
                                    continue
                                try:
                                    lines = fp.read_text(errors="replace").splitlines()
                                    for i, text in enumerate(lines):
                                        if len(matches) >= max_matches:
                                            break
                                        if rx.search(text):
                                            matches.append(
                                                {
                                                    "file": str(fp),
                                                    "line": i + 1,
                                                    "text": text.rstrip(),
                                                }
                                            )
                                except OSError:
                                    continue
                    except _re.error as e:
                        matches = [{"error": f"invalid regex: {e}"}]
                all_results[pattern] = {"matches": matches, "count": len(matches)}
            return [TextContent(type="text", text=json.dumps(all_results, indent=2))]

        elif name == "hygiene_batch":
            from scrub_mcp.pipeline import run_pipeline_batch_parallel

            paths = arguments.get("paths", [])
            steps_arg = arguments.get("steps")
            steps_set = set(steps_arg) if steps_arg else None
            write = arguments.get("write", False)

            # Separate skipped paths from valid ones up front
            skipped = [
                {"file_path": p, "skipped": True, "reason": "not a .py file"}
                for p in paths if not p.endswith(".py")
            ]
            skipped += [
                {"file_path": p, "skipped": True, "reason": "file not found"}
                for p in paths if p.endswith(".py") and not Path(p).exists()
            ]

            valid_paths = [p for p in paths if p.endswith(".py") and Path(p).exists()]
            pipeline_results = run_pipeline_batch_parallel(
                valid_paths, config=CONFIG, steps=steps_set, write=write
            )

            results: list[dict] = list(skipped)
            for p, rpt_or_exc in zip(valid_paths, pipeline_results):
                if rpt_or_exc is None:
                    results.append({"file_path": p, "skipped": True, "reason": "unknown"})
                    continue
                if isinstance(rpt_or_exc, Exception):
                    results.append({"file_path": p, "error": str(rpt_or_exc)})
                    continue
                rpt = rpt_or_exc
                try:
                    file_source = Path(p).read_text(encoding="utf-8") if write else (rpt.modified_source or "")
                except Exception:
                    file_source = ""
                lint_fixed = rpt.lint.auto_fixed if rpt.lint else 0
                file_fixes = lint_fixed + len(rpt.docstrings) + len(rpt.type_annotations) + len(rpt.comments)
                entry: dict = {
                    "file_path": p,
                    "docstrings_added": len(rpt.docstrings),
                    "types_added": len(rpt.type_annotations),
                    "comments_added": len(rpt.comments),
                    "savings": estimate_savings(
                        file_source,
                        file_fixes,
                        CONFIG.savings.price_per_mtoken,
                        CONFIG.savings.currency_unit,
                    ),
                }
                if rpt.lint:
                    entry["lint_violations_fixed"] = rpt.lint.auto_fixed
                if not write:
                    entry["source"] = rpt.modified_source
                results.append(entry)

            total_tokens_saved = sum(
                r.get("savings", {}).get("tokens_saved", 0) for r in results if "savings" in r
            )
            total_est_cost = sum(
                r.get("savings", {}).get("est_cost", 0.0) for r in results if "savings" in r
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "files_processed": len(results),
                            "aggregate_savings": {
                                "tokens_saved": total_tokens_saved,
                                "est_cost": round(total_est_cost, 4),
                                "currency": CONFIG.savings.currency_unit,
                            },
                            "results": results,
                        },
                        indent=2,
                    ),
                )
            ]

        # ── Architecture exploration handler ──

        elif name == "explore_architecture":
            from scrub_mcp.tools.fs import get_tracked_files
            from scrub_mcp.tools.parser import skeletonize
            target = Path(arguments["path"])
            include_private = arguments.get("include_private", False)
            if target.is_dir():
                parts: list[str] = []
                for py_file in get_tracked_files(target, CONFIG.exclude_paths):
                    try:
                        src = py_file.read_text(encoding="utf-8")
                        parts.append(f"\n## {py_file}\n")
                        parts.append(skeletonize(src, str(py_file)))
                    except Exception:
                        logger.debug("[explore_architecture] Skipping %s", py_file)
                result = "\n".join(parts)
            elif target.is_file() and target.suffix == ".py":
                src = target.read_text(encoding="utf-8")
                result = skeletonize(src, str(target))
            else:
                result = f"# {arguments['path']}: not found or not a .py file / directory"
            return [TextContent(type="text", text=result)]

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
                from scrub_mcp.tools.complexity import analyze_complexity

                metrics = analyze_complexity(func)
                if not metrics.needs_simplification and not target:
                    continue

                raw = reducer(
                    function_signature=func.signature,
                    function_body=func.body,
                    cyclomatic_complexity=func.cyclomatic_complexity,
                    line_count=func.body_line_count,
                )
                try:
                    results[func.name] = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    results[func.name] = []

            fixes = sum(len(v) for v in results.values())
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "simplifications": results,
                            "functions_analyzed": len(results),
                            "savings": estimate_savings(
                                source,
                                fixes,
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
                        },
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
                from scrub_mcp.tools.parser import extract_module_info

                mod_info = extract_module_info(fixed_source)
                raw = inferrer(
                    source_code=fixed_source,
                    existing_imports="\n".join(mod_info.imports),
                )
                try:
                    result_data["inferred_imports"] = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    result_data["inferred_imports"] = []

            return [TextContent(type="text", text=json.dumps(result_data, indent=2))]

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

            import re as _re

            all_code_parts: list[str] = []
            covered: list[str] = []

            if len(funcs) <= bs and bs > 1:
                gen = BatchTestGenerator()
                payload = json.dumps(
                    [
                        {
                            "name": f.name,
                            "signature": f.signature,
                            "body": f.body[:2000],
                            "docstring": f.existing_docstring or "",
                            "parent_class": f.parent_class or "",
                        }
                        for f in funcs
                    ]
                )
                all_code_parts.append(gen(functions_json=payload))
                covered = [f.name for f in funcs]
            elif bs > 1:
                from scrub_mcp.tools.utils import batch as batch_fn

                gen = BatchTestGenerator()
                for func_batch in batch_fn(funcs, bs):
                    payload = json.dumps(
                        [
                            {
                                "name": f.name,
                                "signature": f.signature,
                                "body": f.body[:2000],
                                "docstring": f.existing_docstring or "",
                                "parent_class": f.parent_class or "",
                            }
                            for f in func_batch
                        ]
                    )
                    all_code_parts.append(gen(functions_json=payload))
                    covered.extend(f.name for f in func_batch)
            else:
                gen_single = TestGenerator()
                for func in funcs:
                    code = gen_single(
                        function_signature=func.signature,
                        function_body=func.body,
                        docstring=func.existing_docstring or "",
                        parent_class=func.parent_class or "",
                    )
                    all_code_parts.append(code)
                    covered.append(func.name)

            combined_code = "\n\n".join(p for p in all_code_parts if p)
            test_count = len(_re.findall(r"^\s*def test_", combined_code, _re.MULTILINE))

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "test_count": test_count,
                            "functions_covered": covered,
                            "test_code": combined_code,
                            "savings": estimate_savings(
                                source,
                                test_count,
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
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

                try:
                    extractions = json.loads(
                        extract_advisor(
                            function_signature=func.signature,
                            function_body=func.body,
                        )
                        or "[]"
                    )
                except json.JSONDecodeError:
                    extractions = []
                try:
                    renames = json.loads(
                        rename_advisor(
                            code_block=func.body,
                            context=func.signature,
                        )
                        or "[]"
                    )
                except json.JSONDecodeError:
                    renames = []

                if extractions or renames:
                    results[func.name] = {
                        "extractions": extractions,
                        "renames": renames,
                    }

            fixes = sum(len(v["extractions"]) + len(v["renames"]) for v in results.values())
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "functions_analyzed": len(results),
                            "refactoring": results,
                            "savings": estimate_savings(
                                source,
                                fixes,
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
                        },
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
            if any(
                "app = " in line or "Flask(" in line or "FastAPI(" in line
                for line in source_lines[:30]
            ):
                file_context_parts.append("This is a web application.")
            file_context = (
                " ".join(file_context_parts) if file_context_parts else "General Python module."
            )

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
                    triage_results["rewrite"].append(
                        {
                            "test_id": finding.test_id,
                            "verdict": "rewrite",
                            "confidence": 0.0,
                            "description": finding.description,
                            "rationale": "Triage failed, defaulting to rewrite for safety.",
                        }
                    )

            fixes = len(triage_results["rewrite"]) + len(triage_results["nosec"])
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
                            "savings": estimate_savings(
                                source,
                                fixes,
                                CONFIG.savings.price_per_mtoken,
                                CONFIG.savings.currency_unit,
                            ),
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
            output_file = arguments.get("output_file", None)

            report = generate_sbom(
                project_dir=project_dir,
                format=sbom_format,
                include_pip=include_pip,
                output_file=output_file,
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

        elif name == "run_tests":
            import os
            import subprocess

            test_path = arguments.get("test_path", "")
            project_dir = Path(arguments.get("project_dir", ".")).resolve()

            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_dir / "src")

            cmd = ["pytest", "--tb=short", "-q"]
            if test_path:
                cmd.append(test_path)

            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(project_dir),
            )
            output = proc.stdout
            if proc.stderr:
                output += "\n--- stderr ---\n" + proc.stderr

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "exit_code": proc.returncode,
                            "passed": proc.returncode == 0,
                            "output": output,
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


_AGENTS_MD = """\
# S.C.R.U.B. Agent Instructions

You are the **Cloud Orchestrator**. Your role is to plan, reason, and delegate.
You have a local execution server (S.C.R.U.B.) connected via MCP. It handles all
code quality work: linting, documentation, type annotations, tests, security,
refactoring, and supply chain analysis.

## Prime Directive

**Do not write boilerplate code yourself.** If a task can be routed through a
S.C.R.U.B. tool, you MUST route it. You are the architect and reviewer.
S.C.R.U.B. is the compiler.

## Prohibited Actions (The Bash Ban)

You likely have access to a native `bash` or `shell` tool. You are **STRICTLY FORBIDDEN**
from using it for any task that overlaps with S.C.R.U.B. capabilities.

- **NO SHELL LINTING:** Never run `ruff`, `flake8`, `mypy`, `bandit`, or `black` via
    bash. S.C.R.U.B. uses an internal strict configuration that supersedes any local
    `pyproject.toml` — the project's own ruff config is irrelevant and intentionally
    ignored.
- **NO FILE SYSTEM TRAVERSAL:** Never use `find`, `ls -R`, or `grep -r` to discover project
    files. These commands ignore `.gitignore` and poison your context window. You MUST use
    `explore_architecture` to discover files.
- **NO SHELL FILE READING:** Never use `cat`, `grep`, `head`, or `less` to read code.
    Use `explore_architecture` and your native **Read** tool to stay within the token budget.
- **NO SMOKE TESTS:** Never use `python -c`, raw `bash`, or ad-hoc subprocess calls to
    verify imports or test behaviour. Use `run_tests` instead — it handles `src/`-layout
    `PYTHONPATH` injection automatically.
- **TRUST THE COMPILER:** If S.C.R.U.B. returns a clean pass (zero violations, zero
    fixes), trust it. Do not attempt to verify its configuration, inspect `pyproject.toml`
    for ruff settings, or run alternative checks. The `telemetry` key in every lint
    response documents exactly which rules were applied.

## Division of Labor

| You do | S.C.R.U.B. does |
|--------|-----------------|
| Plan architecture | Lint and auto-fix (Ruff) |
| Write business logic | Generate docstrings |
| Make design decisions | Add type annotations |
| Review tool output | Write inline comments |
| Identify what to refactor | Suggest how to refactor |
| Decide on test strategy | Generate test code |
| Assess security posture | Scan and remediate findings |
| Choose dependency versions | Generate SBOM + scan CVEs |

## Context Gathering Rules (Map-Then-Navigate)

When investigating a codebase or planning a new feature, you must optimize for token
efficiency. You are **FORBIDDEN** from reading full files to understand the architecture.

1. **Call `explore_architecture`** on the target directory or files to understand the
     classes, signatures, and docstrings without the implementation bodies.
2. **Identify exact line ranges** of the functions you need to modify based on the skeleton
     map.
3. **Call your native Read tool** using the explicit offset and limit arguments to pull
     only those specific lines.
4. **Never paginate** through a file using sequential reads or read a file in its entirety.

## Mandatory Workflows

### File Modifications

1. Call `hygiene_full` with the current source.
2. Apply the returned source.
3. Make your changes on top of the clean baseline.

### Pre-Commit Checks

1. Call `security_scan` on every modified file.
2. If findings exist, call `security_remediate`.
3. Call `optimize_imports` to finalize the import block.

### When Refactoring

1. Call `analyze_complexity` to identify hotspots — do not guess.
2. Call `suggest_simplifications` for any function with CC ≥ 4.
3. Call `suggest_refactoring` before extracting functions or renaming.

### When Adding Tests

1. Call `generate_tests` to create the test module.
2. Call `run_tests` with the generated test file path to verify it passes.
3. Fix any failures before committing.

### When Dependencies Change

1. Call `generate_sbom` to rebuild the bill of materials.
2. Call `scan_vulnerabilities` to check for new CVEs.

## The Bouncer Rule

If you are about to write code for any of the following, **stop and call the
corresponding tool instead**:

- Writing or editing a docstring → `generate_docstrings` or `hygiene_full`
- Adding a type hint → `annotate_types` or `hygiene_full`
- Fixing a lint error → `lint_file` or `hygiene_full`
- Writing a `# comment` inside a function → `add_comments`
- Deleting code you think is unused → `find_dead_code` first
- Extracting a helper function → `suggest_refactoring` first
- Writing a `test_` function → `generate_tests`
- Auditing for security issues → `security_scan`
- Fixing a security issue → `security_remediate`

If a S.C.R.U.B. tool returns an error or asks you to use a different tool,
respect that and follow the redirect.

## Pre-Processing Gate

These tools **MUST** be called before the Cloud Orchestrator takes any action on a file.
No edits, writes, or plans may proceed without passing through this gate.

| Trigger | Required tool |
|---------|--------------|
| Any file or directory not yet read | `explore_architecture` |
| Before touching a single Python file | `hygiene_full` |
| Before touching multiple Python files | `hygiene_batch` |
| Repeated runs on a file already processed | `hygiene_incremental` |
| Before any commit | `security_scan` → `security_remediate` (if findings) → `optimize_imports` |

## Tool Quick Reference

### Exploration (use these before editing)

| Tool | When to call |
|------|-------------|
| `explore_architecture` | First step to map AST skeletons across files |
| `read_files` | Batch-read multiple files after reviewing the skeleton |
| `find_symbols` | Extract function/class signatures without reading bodies |
| **Read** (native) | Targeted single-file line-range reads |
| `grep_multi` | Find specific string patterns across the codebase |

### Code Hygiene

| Tool | When to call |
|------|-------------|
| `hygiene_full` | First action on any Python file |
| `hygiene_batch` | First action when touching multiple Python files |
| `hygiene_incremental` | Repeated runs on the same file (diff/cache-aware) |
| `lint_file` | Targeted lint-only pass |
| `generate_docstrings` | Docstrings only |
| `annotate_types` | Types only |
| `add_comments` | Comments only |

### Coding Tools

| Tool | When to call |
|------|-------------|
| `analyze_complexity` | Before any refactor |
| `suggest_simplifications` | Before simplifying complex functions |
| `suggest_refactoring` | Before extract/rename decisions |
| `optimize_imports` | After any import change |
| `generate_tests` | Any test generation |
| `run_tests` | After test generation or refactoring |
| `find_dead_code` | Before deleting suspected dead code |

### Security & Supply Chain

| Tool | When to call |
|------|-------------|
| `security_scan` | Before every commit |
| `security_remediate` | After security_scan finds issues |
| `generate_sbom` | After dependency changes |
| `scan_vulnerabilities` | After generate_sbom or before release |
"""


def write_agent_instructions(target_dir: Path) -> Path:
    """Write AGENTS.md into *target_dir* and return the path."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "AGENTS.md"
    out.write_text(_AGENTS_MD)
    return out


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(description="S.C.R.U.B. MCP server")
    parser.add_argument(
        "--agent-instructions",
        metavar="DIR",
        nargs="?",
        const=".",
        default=None,
        help="Write AGENTS.md into DIR (default: current directory) and exit.",
    )
    args = parser.parse_args()

    if args.agent_instructions is not None:
        out = write_agent_instructions(Path(args.agent_instructions))
        print(f"Written: {out}")
        sys.exit(0)

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
