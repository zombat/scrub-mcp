"""S.C.R.U.B. CLI — cache management, quality gates, and project tools.

Commands:
    scrub check   — detect violations, exit 1 if any (CI gate)
    scrub fix     — run pipeline and write hygiene fixes
    scrub diff    — show what fix would change without touching files
    scrub audit   — security scan + supply-chain report
    scrub cache   — cache management (stats, clear, warm)
"""

from __future__ import annotations

import difflib
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="scrub", help="S.C.R.U.B. code hygiene CLI", no_args_is_help=True)
cache_app = typer.Typer(help="Cache management commands", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


# ─────────────────────────────────────────────────────────────────────────────
# Existing cache subcommands (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


@cache_app.command("stats")
def cache_stats(
    cache_dir: str = typer.Option(
        ".scrub_cache/artifacts",
        "--cache-dir",
        help="Cache directory to inspect",
    ),
) -> None:
    """Show cache statistics: size, entry counts, oldest entry, stale fingerprints."""
    from scrub_mcp.config import load_config
    from scrub_mcp.tools.cache import cache_stats as _stats

    cfg = load_config()
    effective_dir = cache_dir if cache_dir != ".scrub_cache/artifacts" else cfg.cache.cache_dir
    stats = _stats(effective_dir)
    current_fp = f"{cfg.model.provider}/{cfg.model.model}"

    stale = sum(
        1 for e in stats["entries"]
        if e.get("model_fingerprint") != current_fp
    )

    typer.echo(f"Cache directory : {effective_dir}")
    typer.echo(f"Total entries   : {stats['total_entries']}")
    typer.echo(f"Total size      : {stats['total_size_mb']} MB")
    typer.echo(f"Current model   : {current_fp}")
    typer.echo(f"Stale entries   : {stale} (mismatched model fingerprint)")
    typer.echo(f"Oldest entry    : {stats['oldest_timestamp'] or 'N/A'}")
    typer.echo("")
    typer.echo("Entries by artifact type:")
    for atype, count in sorted(stats["by_artifact_type"].items()):
        typer.echo(f"  {atype:12s} : {count}")


@cache_app.command("clear")
def cache_clear(
    cache_dir: str = typer.Option(
        ".scrub_cache/artifacts",
        "--cache-dir",
        help="Cache directory to clear",
    ),
    artifact_type: str | None = typer.Option(
        None,
        "--type",
        help="Only remove entries of this artifact type (docstring, type, comment, test)",
    ),
    stale: bool = typer.Option(
        False,
        "--stale",
        help="Only remove entries with a mismatched model fingerprint",
    ),
) -> None:
    """Remove cache entries. With no flags, clears everything (prompts for confirmation)."""
    from scrub_mcp.config import load_config
    from scrub_mcp.tools.cache import CacheEntry, cache_stats as _stats

    cfg = load_config()
    effective_dir = cache_dir if cache_dir != ".scrub_cache/artifacts" else cfg.cache.cache_dir
    cache_path = Path(effective_dir)

    if not cache_path.exists():
        typer.echo("Cache directory does not exist. Nothing to clear.")
        return

    if not artifact_type and not stale:
        # Full clear with confirmation
        typer.confirm(
            f"Delete entire cache at {effective_dir}?",
            abort=True,
        )
        shutil.rmtree(effective_dir, ignore_errors=True)
        typer.echo("Cache cleared.")
        return

    current_fp = f"{cfg.model.provider}/{cfg.model.model}"
    removed = 0

    for f in cache_path.rglob("*.json"):
        try:
            entry = CacheEntry.model_validate_json(f.read_text(encoding="utf-8"))
            remove = False
            if artifact_type and entry.artifact_type == artifact_type:
                remove = True
            if stale and entry.model_fingerprint != current_fp:
                remove = True
            if remove:
                f.unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass

    typer.echo(f"Removed {removed} entries.")


@cache_app.command("warm")
def cache_warm(
    path: str = typer.Argument(..., help="Project path or directory to warm"),
    steps: list[str] = typer.Option(
        ["docstrings", "types"],
        "--step",
        help="Steps to run (repeat flag for multiple: --step docstrings --step types)",
    ),
    workers: int = typer.Option(4, "--workers", help="Parallel workers"),
) -> None:
    """Pre-populate the cache by running the pipeline on the entire project."""
    from scrub_mcp.config import load_config
    from scrub_mcp.pipeline import run_pipeline_batch_parallel
    from scrub_mcp.tools.fs import get_tracked_files

    cfg = load_config()
    project_path = Path(path).resolve()

    if not project_path.exists():
        typer.echo(f"Path not found: {project_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Warming cache for {project_path} ...")
    typer.echo(f"Steps   : {', '.join(steps)}")
    typer.echo(f"Workers : {workers}")

    py_files = get_tracked_files(str(project_path), cfg.exclude_paths)
    file_paths = [str(f) for f in py_files if f.suffix == ".py"]

    if not file_paths:
        typer.echo("No .py files found.")
        return

    typer.echo(f"Files   : {len(file_paths)}")

    results = run_pipeline_batch_parallel(
        file_paths, cfg, set(steps), write=False, max_workers=workers
    )

    success = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    errors = sum(1 for r in results if isinstance(r, Exception))

    typer.echo(f"\nDone. {success} files processed, {errors} errors.")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_ORDER: dict[str, int] = {
    "low": 0, "medium": 1, "high": 2, "critical": 3
}


def _parse_fail_on(fail_on_str: str) -> dict:
    """Parse --fail-on CSV into a structured dict.

    Returns a dict with any of these keys set:
        "missing-docstrings": True
        "missing-types": True
        "complexity": int  (cyclomatic threshold)
        "security": str    (min severity: low/medium/high)
        "vulns": str       (min severity: low/medium/high/critical)
    """
    result: dict = {}
    if not fail_on_str.strip():
        return result
    for token in fail_on_str.split(","):
        token = token.strip()
        if not token:
            continue
        if token in ("missing-docstrings", "missing-types"):
            result[token] = True
        elif token.startswith("complexity:"):
            try:
                result["complexity"] = int(token.split(":", 1)[1])
            except ValueError:
                typer.echo(f"Invalid complexity threshold: {token}", err=True)
        elif token.startswith("security:"):
            result["security"] = token.split(":", 1)[1].lower()
        elif token.startswith("vulns:"):
            result["vulns"] = token.split(":", 1)[1].lower()
        else:
            typer.echo(f"Unknown --fail-on token: {token}", err=True)
    return result


def _make_relative(file_path: str | Path) -> str:
    """Return *file_path* relative to cwd when possible, else as-is."""
    try:
        return str(Path(file_path).relative_to(Path.cwd()))
    except ValueError:
        return str(file_path)


def _resolve_py_files(
    path: str,
    since: str | None,
    cfg,
    quiet: bool,
) -> list[Path]:
    """Return Python files to process, optionally filtered by --since."""
    from scrub_mcp.tools.diff import get_git_diff, parse_diff
    from scrub_mcp.tools.fs import get_tracked_files

    p = Path(path).resolve()
    if p.is_file():
        return [p] if p.suffix == ".py" else []

    all_py = [f for f in get_tracked_files(str(p), cfg.exclude_paths) if f.suffix == ".py"]

    if not since:
        return all_py

    diff_text = get_git_diff(str(p), since)
    if not diff_text:
        if not quiet:
            typer.echo(f"No diff found for --since={since}; checking all files.", err=True)
        return all_py

    changed_ranges = parse_diff(diff_text)
    changed_rel: set[str] = {cr.file_path for cr in changed_ranges}

    filtered = [
        f for f in all_py
        if any(
            str(f).endswith(cp) or _make_relative(f) == cp
            for cp in changed_rel
        )
    ]
    if not quiet:
        typer.echo(
            f"--since={since}: {len(filtered)} of {len(all_py)} files changed.",
            err=True,
        )
    return filtered


def _collect_violations(
    file_path: str,
    source: str,
    fail_on: dict,
) -> list[dict]:
    """Collect violations from a single Python file based on *fail_on* flags.

    All checks are deterministic (no LLM).
    """
    from scrub_mcp.tools.complexity import analyze_file_complexity
    from scrub_mcp.tools.parser import extract_classes, extract_functions, extract_module_info, parse_source
    from scrub_mcp.tools.security import run_bandit

    violations: list[dict] = []
    rel = _make_relative(file_path)

    # Parse AST once; reuse for all checks.
    try:
        tree = parse_source(source)
    except SyntaxError as exc:
        violations.append({
            "rule_id": "SCRUB-PARSE-001",
            "file": rel,
            "line_start": exc.lineno or 1,
            "line_end": exc.lineno or 1,
            "message": f"Syntax error: {exc.msg}",
            "level": "error",
        })
        return violations

    # Cache parsed structures — only extract what is needed.
    _functions: list | None = None
    _classes: list | None = None

    def _get_functions():
        nonlocal _functions
        if _functions is None:
            _functions = extract_functions(source, tree=tree)
        return _functions

    def _get_classes():
        nonlocal _classes
        if _classes is None:
            _classes = extract_classes(source, tree=tree)
        return _classes

    # ── missing-docstrings ────────────────────────────────────────────────────
    if fail_on.get("missing-docstrings"):
        module_info = extract_module_info(source, tree=tree)
        if not module_info.existing_docstring:
            stem = Path(file_path).stem
            violations.append({
                "rule_id": "SCRUB-DOC-003",
                "file": rel,
                "line_start": 1,
                "line_end": 1,
                "message": f"Module '{stem}' is missing a docstring",
                "level": "note",
            })

        for cls in _get_classes():
            if not cls.existing_docstring:
                violations.append({
                    "rule_id": "SCRUB-DOC-002",
                    "file": rel,
                    "line_start": cls.line_start,
                    "line_end": cls.line_end,
                    "message": f"Class '{cls.name}' is missing a docstring",
                    "level": "warning",
                    "function_name": cls.name,
                })

        for func in _get_functions():
            if not func.existing_docstring:
                violations.append({
                    "rule_id": "SCRUB-DOC-001",
                    "file": rel,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "message": f"Function '{func.name}' is missing a docstring",
                    "level": "warning",
                    "function_name": func.name,
                })

    # ── missing-types ─────────────────────────────────────────────────────────
    if fail_on.get("missing-types"):
        for func in _get_functions():
            if not func.existing_annotations:
                violations.append({
                    "rule_id": "SCRUB-TYPE-001",
                    "file": rel,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "message": f"Function '{func.name}' is missing type annotations",
                    "level": "warning",
                    "function_name": func.name,
                })

    # ── complexity:N ──────────────────────────────────────────────────────────
    if "complexity" in fail_on:
        threshold = fail_on["complexity"]
        report = analyze_file_complexity(_get_functions())
        func_map = {f.name: f for f in _get_functions()}
        for fc in report.functions:
            if fc.cyclomatic_complexity >= threshold:
                func = func_map.get(fc.function_name)
                violations.append({
                    "rule_id": "SCRUB-CMPLX-001",
                    "file": rel,
                    "line_start": func.line_start if func else 1,
                    "line_end": func.line_end if func else 1,
                    "message": (
                        f"Function '{fc.function_name}' has cyclomatic complexity "
                        f"{fc.cyclomatic_complexity} (threshold: {threshold})"
                    ),
                    "level": "warning",
                    "function_name": fc.function_name,
                })

    # ── security:SEVERITY ─────────────────────────────────────────────────────
    if "security" in fail_on:
        min_sev = fail_on["security"].lower()
        threshold_val = _SEVERITY_ORDER.get(min_sev, 0)
        bandit_report = run_bandit(source, severity_threshold=min_sev)
        for finding in bandit_report.findings:
            if _SEVERITY_ORDER.get(finding.severity.lower(), 0) >= threshold_val:
                end_line = (
                    finding.line_range[-1]
                    if finding.line_range
                    else finding.line_number
                )
                sev_level = {
                    "HIGH": "error", "MEDIUM": "warning", "LOW": "note"
                }.get(finding.severity.upper(), "warning")
                violations.append({
                    "rule_id": f"SCRUB-SEC-{finding.test_id}",
                    "file": rel,
                    "line_start": finding.line_number,
                    "line_end": end_line,
                    "message": f"[{finding.test_id}] {finding.description}",
                    "level": sev_level,
                })

    return violations


def _format_violations_text(violations: list[dict]) -> str:
    """Format violations as human-readable text for stdout."""
    if not violations:
        return "No violations found.\n"
    lines = []
    for v in violations:
        loc = f"{v['file']}:{v['line_start']}"
        fn = f" ({v['function_name']})" if v.get("function_name") else ""
        lines.append(f"{loc}{fn}  [{v['level'].upper()}]  {v['rule_id']}  {v['message']}")
    lines.append(f"\n{len(violations)} violation(s) found.")
    return "\n".join(lines) + "\n"


def _write_output(content: str, output: str | None) -> None:
    """Write *content* to *output* file, or print to stdout."""
    if output:
        Path(output).write_text(content, encoding="utf-8")
    else:
        typer.echo(content, nl=False)


# ─────────────────────────────────────────────────────────────────────────────
# scrub check
# ─────────────────────────────────────────────────────────────────────────────


@app.command("check")
def check_cmd(
    path: str = typer.Argument(".", help="File or directory to check"),
    config: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
    steps: Optional[list[str]] = typer.Option(None, "--steps", help="Steps (unused for check; use --fail-on)"),
    since: Optional[str] = typer.Option(None, "--since", help="Only check files changed since this git ref"),
    fail_on: str = typer.Option(
        "",
        "--fail-on",
        help=(
            "Comma-separated violations that trigger exit 1. "
            "Options: missing-docstrings, missing-types, complexity:N, "
            "security:SEVERITY, vulns:SEVERITY"
        ),
    ),
    fmt: str = typer.Option("text", "--format", help="Output format: text, json, or sarif"),
    output: Optional[str] = typer.Option(None, "--output", help="Write output to this file instead of stdout"),
) -> None:
    """Detect violations without modifying files.

    Exit 0 if clean (or --fail-on is empty), exit 1 if violations found.
    Entirely deterministic — no LLM required.
    """
    from scrub_mcp.config import load_config

    cfg = load_config(Path(config) if Path(config).exists() else None)
    fail_on_parsed = _parse_fail_on(fail_on)

    # Default checks when --fail-on is empty (informational mode).
    if not fail_on_parsed:
        fail_on_parsed = {"missing-docstrings": True, "missing-types": True}

    py_files = _resolve_py_files(path, since, cfg, quiet)
    if not py_files:
        if not quiet:
            typer.echo("No Python files found.", err=True)
        raise typer.Exit(0)

    if not quiet:
        typer.echo(f"Checking {len(py_files)} file(s)...", err=True)

    all_violations: list[dict] = []

    for fpath in py_files:
        try:
            source = fpath.read_text(encoding="utf-8")
        except OSError as exc:
            typer.echo(f"Cannot read {fpath}: {exc}", err=True)
            continue
        all_violations.extend(_collect_violations(str(fpath), source, fail_on_parsed))

    # ── vulns:SEVERITY is project-level, runs once ────────────────────────────
    if "vulns" in fail_on_parsed:
        from scrub_mcp.tools.sbom import generate_sbom
        from scrub_mcp.tools.vulnscan import scan_components

        min_sev = fail_on_parsed["vulns"].upper()
        threshold_val = _SEVERITY_ORDER.get(min_sev.lower(), 0)
        if not quiet:
            typer.echo("Scanning for vulnerabilities (OSV.dev)...", err=True)
        project_dir = Path(path).resolve() if Path(path).is_dir() else Path(path).resolve().parent
        sbom = generate_sbom(project_dir)
        vuln_report = scan_components(sbom.components)
        for finding in vuln_report.findings:
            if _SEVERITY_ORDER.get(finding.severity.lower(), 0) >= threshold_val:
                vuln_id = finding.cve_id or finding.vuln_id
                all_violations.append({
                    "rule_id": f"SCRUB-VULN-{vuln_id}",
                    "file": "pyproject.toml",
                    "line_start": 1,
                    "line_end": 1,
                    "message": (
                        f"[{finding.severity}] {finding.component_name}@"
                        f"{finding.component_version}: {finding.summary}"
                        + (f" (fix: {finding.fixed_version})" if finding.fixed_version else "")
                    ),
                    "level": {"CRITICAL": "error", "HIGH": "error"}.get(
                        finding.severity.upper(), "warning"
                    ),
                })

    # ── Format and emit ───────────────────────────────────────────────────────
    if fmt == "json":
        content = json.dumps(all_violations, indent=2) + "\n"
    elif fmt == "sarif":
        from scrub_mcp.tools.sarif import to_sarif
        content = json.dumps(to_sarif(all_violations), indent=2) + "\n"
    else:
        content = _format_violations_text(all_violations)

    _write_output(content, output)

    # Gate: exit 1 only when --fail-on was explicitly set and violations exist.
    if fail_on.strip() and all_violations:
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# scrub fix
# ─────────────────────────────────────────────────────────────────────────────


@app.command("fix")
def fix_cmd(
    path: str = typer.Argument(".", help="File or directory to fix"),
    config: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
    steps: Optional[list[str]] = typer.Option(
        None,
        "--steps",
        help="Pipeline steps to run (lint, docstrings, types, comments). Default: all.",
    ),
    since: Optional[str] = typer.Option(None, "--since", help="Only fix files changed since this git ref"),
    commit: bool = typer.Option(False, "--commit", help="Create a git commit after fixing"),
    branch: Optional[str] = typer.Option(None, "--branch", help="Create this branch before committing"),
) -> None:
    """Run the hygiene pipeline and write fixes to disk.

    Requires a local LLM (configured in config.yaml).
    """
    import subprocess

    from scrub_mcp.config import load_config
    from scrub_mcp.pipeline import configure_dspy, run_pipeline

    cfg = load_config(Path(config) if Path(config).exists() else None)
    configure_dspy(cfg)

    step_set = set(steps) if steps else None  # None = all steps

    py_files = _resolve_py_files(path, since, cfg, quiet)
    if not py_files:
        if not quiet:
            typer.echo("No Python files found.", err=True)
        raise typer.Exit(0)

    if not quiet:
        typer.echo(f"Fixing {len(py_files)} file(s)...", err=True)

    modified: list[str] = []
    for fpath in py_files:
        try:
            source = fpath.read_text(encoding="utf-8")
        except OSError as exc:
            typer.echo(f"Cannot read {fpath}: {exc}", err=True)
            continue

        report = run_pipeline(source, str(fpath), cfg, step_set)
        if report.modified_source and report.modified_source != source:
            fpath.write_text(report.modified_source, encoding="utf-8")
            modified.append(str(fpath))
            if not quiet:
                typer.echo(f"  fixed: {_make_relative(fpath)}", err=True)

    if not quiet:
        typer.echo(f"\n{len(modified)} file(s) modified.", err=True)

    if not modified:
        raise typer.Exit(0)

    if branch:
        try:
            subprocess.run(["git", "checkout", "-b", branch], check=True)
            if not quiet:
                typer.echo(f"Created branch: {branch}", err=True)
        except subprocess.CalledProcessError as exc:
            typer.echo(f"Could not create branch {branch!r}: {exc}", err=True)

    if commit:
        try:
            subprocess.run(["git", "add"] + modified, check=True)
            subprocess.run(
                ["git", "commit", "-m", "chore(scrub): auto-apply hygiene fixes"],
                check=True,
            )
            if not quiet:
                typer.echo("Committed hygiene fixes.", err=True)
        except subprocess.CalledProcessError as exc:
            typer.echo(f"git commit failed: {exc}", err=True)
            raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# scrub diff
# ─────────────────────────────────────────────────────────────────────────────


@app.command("diff")
def diff_cmd(
    path: str = typer.Argument(".", help="File or directory to diff"),
    config: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
    steps: Optional[list[str]] = typer.Option(None, "--steps", help="Pipeline steps to run"),
    since: Optional[str] = typer.Option(None, "--since", help="Only diff files changed since this git ref"),
    fmt: str = typer.Option("unified", "--format", help="Output format: unified or json"),
) -> None:
    """Show what 'scrub fix' would change without touching any files.

    Requires a local LLM (configured in config.yaml).
    Outputs to stdout so the diff can be piped to git apply or gh pr comment.
    """
    from scrub_mcp.config import load_config
    from scrub_mcp.pipeline import configure_dspy, run_pipeline

    cfg = load_config(Path(config) if Path(config).exists() else None)
    configure_dspy(cfg)

    step_set = set(steps) if steps else None

    py_files = _resolve_py_files(path, since, cfg, quiet)
    if not py_files:
        raise typer.Exit(0)

    if not quiet:
        typer.echo(f"Diffing {len(py_files)} file(s)...", err=True)

    diffs: list[dict] = []
    for fpath in py_files:
        try:
            source = fpath.read_text(encoding="utf-8")
        except OSError as exc:
            typer.echo(f"Cannot read {fpath}: {exc}", err=True)
            continue

        report = run_pipeline(source, str(fpath), cfg, step_set)
        modified = report.modified_source or source

        if modified == source:
            continue

        rel = _make_relative(fpath)
        unified = "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
        diffs.append({"file": rel, "diff": unified})

    if fmt == "json":
        typer.echo(json.dumps(diffs, indent=2))
    else:
        for d in diffs:
            typer.echo(d["diff"], nl=False)

    if not diffs and not quiet:
        typer.echo("No changes.", err=True)


# ─────────────────────────────────────────────────────────────────────────────
# scrub audit
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT_SEV_ORDER: dict[str, int] = {
    "low": 0, "medium": 1, "high": 2, "critical": 3, "unknown": -1
}


@app.command("audit")
def audit_cmd(
    path: str = typer.Argument(".", help="Project root to audit"),
    config: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
    since: Optional[str] = typer.Option(None, "--since", help="Only audit files changed since this git ref"),
    sbom_format: str = typer.Option("cyclonedx", "--sbom-format", help="SBOM format: cyclonedx or spdx"),
    fail_on_severity: Optional[str] = typer.Option(
        None,
        "--fail-on-severity",
        help="Exit 1 if any finding is at or above this severity (low, medium, high, critical)",
    ),
    output_sbom: Optional[str] = typer.Option(None, "--output-sbom", help="Write SBOM to this file"),
    fmt: str = typer.Option("text", "--format", help="Report format: text, json, or sarif"),
    output: Optional[str] = typer.Option(None, "--output", help="Write report to this file instead of stdout"),
) -> None:
    """Run security scan + supply-chain audit. No LLM required.

    Combines Bandit (static analysis) and OSV.dev (vulnerability database)
    into a single compliance report. Optionally writes an SBOM.
    """
    from scrub_mcp.config import load_config
    from scrub_mcp.tools.sbom import generate_sbom
    from scrub_mcp.tools.security import run_bandit
    from scrub_mcp.tools.vulnscan import scan_components

    cfg = load_config(Path(config) if Path(config).exists() else None)
    project_dir = Path(path).resolve() if Path(path).is_dir() else Path(path).resolve().parent

    py_files = _resolve_py_files(path, since, cfg, quiet)

    # ── Security scan (Bandit) ────────────────────────────────────────────────
    if not quiet:
        typer.echo(f"Security scan: {len(py_files)} file(s)...", err=True)

    sec_violations: list[dict] = []
    for fpath in py_files:
        try:
            source = fpath.read_text(encoding="utf-8")
        except OSError:
            continue
        bandit_report = run_bandit(source)
        rel = _make_relative(fpath)
        for finding in bandit_report.findings:
            end_line = finding.line_range[-1] if finding.line_range else finding.line_number
            sev_level = {"HIGH": "error", "MEDIUM": "warning", "LOW": "note"}.get(
                finding.severity.upper(), "warning"
            )
            sec_violations.append({
                "rule_id": f"SCRUB-SEC-{finding.test_id}",
                "file": rel,
                "line_start": finding.line_number,
                "line_end": end_line,
                "message": f"[{finding.test_id}] {finding.description}",
                "level": sev_level,
                "severity": finding.severity.upper(),
            })

    # ── Supply chain (SBOM + OSV) ─────────────────────────────────────────────
    if not quiet:
        typer.echo("Generating SBOM...", err=True)

    sbom = generate_sbom(project_dir, format=sbom_format)

    if output_sbom:
        Path(output_sbom).write_text(sbom.output_json, encoding="utf-8")
        if not quiet:
            typer.echo(f"SBOM written to {output_sbom}", err=True)

    if not quiet:
        typer.echo(f"Scanning {sbom.component_count} component(s) against OSV.dev...", err=True)

    vuln_report = scan_components(sbom.components)
    vuln_violations: list[dict] = []
    for finding in vuln_report.findings:
        vuln_id = finding.cve_id or finding.vuln_id
        sev_level = {"CRITICAL": "error", "HIGH": "error", "MEDIUM": "warning", "LOW": "note"}.get(
            finding.severity.upper(), "note"
        )
        vuln_violations.append({
            "rule_id": f"SCRUB-VULN-{vuln_id}",
            "file": "pyproject.toml",
            "line_start": 1,
            "line_end": 1,
            "message": (
                f"[{finding.severity}] {finding.component_name}@"
                f"{finding.component_version}: {finding.summary}"
                + (f" (fix: {finding.fixed_version})" if finding.fixed_version else "")
            ),
            "level": sev_level,
            "severity": finding.severity.upper(),
        })

    all_violations = sec_violations + vuln_violations

    # ── Format report ─────────────────────────────────────────────────────────
    if fmt == "json":
        report_doc = {
            "security": sec_violations,
            "vulnerabilities": vuln_violations,
            "sbom_components": sbom.component_count,
            "total_findings": len(all_violations),
        }
        content = json.dumps(report_doc, indent=2) + "\n"
    elif fmt == "sarif":
        from scrub_mcp.tools.sarif import to_sarif
        content = json.dumps(to_sarif(all_violations), indent=2) + "\n"
    else:
        lines = [
            f"S.C.R.U.B. Audit Report",
            f"{'=' * 40}",
            f"Files scanned : {len(py_files)}",
            f"Components    : {sbom.component_count}",
            f"",
            f"Security findings ({len(sec_violations)}):",
        ]
        for v in sec_violations:
            lines.append(f"  {v['file']}:{v['line_start']}  [{v['severity']}]  {v['message']}")
        if not sec_violations:
            lines.append("  (none)")
        lines += [
            f"",
            f"Vulnerability findings ({len(vuln_violations)}):",
        ]
        for v in vuln_violations:
            lines.append(f"  [{v['severity']}]  {v['message']}")
        if not vuln_violations:
            lines.append("  (none)")
        lines.append(f"\n{len(all_violations)} total finding(s).")
        content = "\n".join(lines) + "\n"

    _write_output(content, output)

    # ── Exit code ─────────────────────────────────────────────────────────────
    if fail_on_severity:
        threshold = _AUDIT_SEV_ORDER.get(fail_on_severity.lower(), 0)
        if any(
            _AUDIT_SEV_ORDER.get(v.get("severity", "unknown").lower(), -1) >= threshold
            for v in all_violations
        ):
            raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the scrub CLI."""
    app()


if __name__ == "__main__":
    main()
