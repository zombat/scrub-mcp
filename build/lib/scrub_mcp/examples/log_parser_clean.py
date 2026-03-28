"""Utilities for parsing and summarizing structured log files."""

import re
from collections import Counter
from pathlib import Path
from typing import Any


class LogParser:
    """Parse log files against named regex patterns and cache matched entries."""

    def __init__(self, log_dir: str | Path, patterns: dict[str, str] | None = None) -> None:
        """Initialize the LogParser.

        Args:
            log_dir: Path to the directory containing log files.
            patterns: Mapping of pattern names to regex strings.
                Defaults to an empty dict if not provided.
        """
        self.log_dir = Path(log_dir)
        self.patterns = patterns or {}
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def parse_file(self, filename: str) -> list[dict[str, Any]]:
        """Parse a log file and extract entries matching configured patterns.

        Args:
            filename: Name of the log file to parse, relative to log_dir.

        Returns:
            List of dicts, each containing line number, pattern name,
            match text, severity, and surrounding context lines.

        Raises:
            FileNotFoundError: If the specified log file does not exist.
        """
        path = self.log_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        lines = path.read_text().splitlines()
        results: list[dict[str, Any]] = []

        for i, line in enumerate(lines):
            for name, pattern in self.patterns.items():
                match = re.search(pattern, line)
                if match:
                    severity = (
                        "error" if "error" in line.lower()
                        else "warning" if "warn" in line.lower()
                        else "info"
                    )
                    entry: dict[str, Any] = {
                        "line": i + 1,
                        "pattern": name,
                        "match": match.group(0),
                        "severity": severity,
                        "context": lines[max(0, i - 2):i + 3],
                    }
                    results.append(entry)

                    if name not in self._cache:
                        self._cache[name] = []
                    self._cache[name].append(entry)

        return results

    def get_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize parsed results by severity and pattern.

        Args:
            results: List of parsed log entries from parse_file.

        Returns:
            Dict with total count, by_severity counts, and by_pattern counts.
        """
        summary: dict[str, Any] = {"total": len(results), "by_severity": {}, "by_pattern": {}}

        for entry in results:
            sev = entry["severity"]
            pat = entry["pattern"]
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            summary["by_pattern"][pat] = summary["by_pattern"].get(pat, 0) + 1

        return summary

    def __repr__(self) -> str:
        return f"LogParser(log_dir={self.log_dir}, patterns={len(self.patterns)})"


def find_anomalies(entries: list[dict[str, Any]], threshold: int = 3) -> list[dict[str, Any]]:
    """Detect anomalous patterns using z-score analysis.

    Args:
        entries: List of parsed log entries, each containing a "pattern" key.
        threshold: Number of standard deviations above the mean to flag as
            anomalous. Defaults to 3.

    Returns:
        List of anomaly dicts sorted by z_score descending, each containing
        pattern name, count, and z_score.
    """
    if not entries:
        return []

    pattern_counts = Counter(e["pattern"] for e in entries)
    mean = sum(pattern_counts.values()) / len(pattern_counts)
    std = (sum((v - mean) ** 2 for v in pattern_counts.values()) / len(pattern_counts)) ** 0.5

    anomalies = []
    for pattern, count in pattern_counts.items():
        if count > mean + threshold * std:
            anomalies.append({
                "pattern": pattern,
                "count": count,
                "z_score": (count - mean) / std if std > 0 else 0,
            })

    return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)
