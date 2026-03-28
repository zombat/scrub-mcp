"""Ruff linter wrapper. Deterministic-first: no LLM needed."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from scrub.config import RuffConfig
from scrub.models import LintResult


def run_ruff(source: str, config: RuffConfig | None = None) -> tuple[str, LintResult]:
    """Run Ruff on source code. Returns (fixed_source, lint_result).

    Writes to a temp file because Ruff operates on files, not stdin for fix mode.
    """
    cfg = config or RuffConfig()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        # Count violations before fix
        check_before = subprocess.run(
            [
                "ruff", "check",
                "--select", ",".join(cfg.select),
                "--ignore", ",".join(cfg.ignore),
                "--line-length", str(cfg.line_length),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        violations_before = _count_violations(check_before.stdout)

        # Apply fixes
        if cfg.fix:
            subprocess.run(
                [
                    "ruff", "check", "--fix",
                    "--select", ",".join(cfg.select),
                    "--ignore", ",".join(cfg.ignore),
                    "--line-length", str(cfg.line_length),
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
            )

        # Format
        subprocess.run(
            ["ruff", "format", "--line-length", str(cfg.line_length), str(tmp_path)],
            capture_output=True,
            text=True,
        )

        # Count remaining violations
        check_after = subprocess.run(
            [
                "ruff", "check",
                "--select", ",".join(cfg.select),
                "--ignore", ",".join(cfg.ignore),
                "--line-length", str(cfg.line_length),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        violations_after = _count_violations(check_after.stdout)
        remaining = [
            line.strip()
            for line in check_after.stdout.splitlines()
            if line.strip() and not line.startswith("Found")
        ]

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


def _count_violations(ruff_output: str) -> int:
    """Parse Ruff output to count violation lines."""
    return sum(
        1
        for line in ruff_output.splitlines()
        if line.strip() and not line.startswith("Found") and not line.startswith("All checks")
    )
