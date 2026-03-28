"""Bandit security scanner. Deterministic, no LLM.

Wraps the Bandit static analysis tool to find common security
issues in Python code: hardcoded passwords, SQL injection,
shell injection, insecure deserialization, weak crypto, etc.

Deterministic-first: Bandit handles the detection. DSPy can
optionally generate remediation suggestions for flagged issues
via the security_remediation MCP tool.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from scrub.models import BanditFinding, BanditReport

logger = logging.getLogger(__name__)


def run_bandit(
    source: str,
    severity_threshold: str = "low",
    confidence_threshold: str = "low",
    skips: list[str] | None = None,
) -> BanditReport:
    """Run Bandit on Python source code.

    Args:
        source: Python source code string.
        severity_threshold: Minimum severity to report (low, medium, high).
        confidence_threshold: Minimum confidence to report (low, medium, high).
        skips: List of Bandit test IDs to skip (e.g., ["B101"] to skip assert checks).

    Returns:
        BanditReport with findings and summary.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "bandit",
            "--format", "json",
            "--severity-level", severity_threshold.upper()[0],
            "--confidence-level", confidence_threshold.upper()[0],
        ]

        if skips:
            cmd.extend(["--skip", ",".join(skips)])

        cmd.append(str(tmp_path))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Bandit returns exit code 1 when findings exist, 0 when clean
        try:
            report_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("[bandit] Failed to parse output: %s", result.stderr[:200])
            return BanditReport(
                findings=[],
                total=0,
                error=result.stderr[:500] if result.stderr else "Failed to parse Bandit output",
            )

        findings = []
        for item in report_data.get("results", []):
            findings.append(
                BanditFinding(
                    test_id=item.get("test_id", ""),
                    test_name=item.get("test_name", ""),
                    severity=item.get("issue_severity", "LOW"),
                    confidence=item.get("issue_confidence", "LOW"),
                    description=item.get("issue_text", ""),
                    line_number=item.get("line_number", 0),
                    line_range=item.get("line_range", []),
                    code=item.get("code", ""),
                    cwe=item.get("issue_cwe", {}).get("id", 0),
                    more_info=item.get("more_info", ""),
                )
            )

        metrics = report_data.get("metrics", {}).get(str(tmp_path), {})

        return BanditReport(
            findings=findings,
            total=len(findings),
            by_severity={
                "high": metrics.get("SEVERITY.HIGH", 0),
                "medium": metrics.get("SEVERITY.MEDIUM", 0),
                "low": metrics.get("SEVERITY.LOW", 0),
            },
            by_confidence={
                "high": metrics.get("CONFIDENCE.HIGH", 0),
                "medium": metrics.get("CONFIDENCE.MEDIUM", 0),
                "low": metrics.get("CONFIDENCE.LOW", 0),
            },
            lines_of_code=metrics.get("loc", 0),
        )

    except FileNotFoundError:
        logger.error("[bandit] Not installed. Run: pip install bandit")
        return BanditReport(
            findings=[],
            total=0,
            error="Bandit not installed. Run: pip install bandit",
        )
    except subprocess.TimeoutExpired:
        logger.warning("[bandit] Timed out after 30s")
        return BanditReport(
            findings=[],
            total=0,
            error="Bandit timed out after 30 seconds",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def run_bandit_on_file(
    file_path: Path,
    severity_threshold: str = "low",
    confidence_threshold: str = "low",
    skips: list[str] | None = None,
) -> BanditReport:
    """Convenience wrapper: run Bandit on a file.

    Args:
        file_path: Path to the Python file.
        severity_threshold: Minimum severity to report.
        confidence_threshold: Minimum confidence to report.
        skips: Bandit test IDs to skip.

    Returns:
        BanditReport with findings and summary.
    """
    source = file_path.read_text(encoding="utf-8")
    report = run_bandit(source, severity_threshold, confidence_threshold, skips)
    # Patch file path into findings for reporting
    for finding in report.findings:
        finding.file_path = str(file_path)
    return report
