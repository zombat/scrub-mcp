"""Ruff linter wrapper. Deterministic-first: no LLM needed."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from scrub_mcp.config import RuffConfig
from scrub_mcp.models import LintResult


def run_ruff(source: str, config: RuffConfig | None = None) -> tuple[str, LintResult]:
    """Run Ruff on source code. Returns (fixed_source, lint_result).

    Writes to a temp file because Ruff operates on files, not stdin for fix mode.
    Three subprocess calls (down from four): check-JSON, fix, format.
    The JSON check captures all violation metadata before fixing, eliminating
    the redundant check-after-fix call.
    """
    cfg = config or RuffConfig()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        common_flags = [
            "--select", ",".join(cfg.select),
            "--ignore", ",".join(cfg.ignore),
            "--line-length", str(cfg.line_length),
        ]

        # Call 1: JSON check (no --fix). Reports all violations with their fix eligibility.
        # violations with a "fix" field are auto-fixable; fix=null are not.
        check_result = subprocess.run(
            ["ruff", "check", "--output-format", "json"] + common_flags + [str(tmp_path)],
            capture_output=True,
            text=True,
        )

        try:
            diagnostics: list[dict] = json.loads(check_result.stdout) if check_result.stdout.strip() else []
        except json.JSONDecodeError:
            diagnostics = []

        violations_before = len(diagnostics)
        # Diagnostics with fix=null cannot be auto-fixed; they remain after --fix
        remaining_diags = [d for d in diagnostics if d.get("fix") is None]
        violations_after = len(remaining_diags)
        remaining = [
            f"{d.get('filename', '')}:{d.get('location', {}).get('row', 0)}: "
            f"[{d.get('code', '')}] {d.get('message', '')}"
            for d in remaining_diags
        ]

        # Call 2: Apply fixes in-place
        if cfg.fix and violations_before > violations_after:
            subprocess.run(
                ["ruff", "check", "--fix"] + common_flags + [str(tmp_path)],
                capture_output=True,
                text=True,
            )

        # Call 3: Format
        subprocess.run(
            ["ruff", "format", "--line-length", str(cfg.line_length), str(tmp_path)],
            capture_output=True,
            text=True,
        )

        fixed_source = tmp_path.read_text(encoding="utf-8")

        return fixed_source, LintResult(
            file_path=str(tmp_path),
            violations_before=violations_before,
            violations_after=violations_after,
            auto_fixed=violations_before - violations_after,
            remaining_issues=remaining,
        )

    finally:
        tmp_path.unlink(missing_ok=True)
