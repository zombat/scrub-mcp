import re
from pathlib import Path


class LogParser:
    def __init__(self, log_dir, patterns=None):
        self.log_dir = Path(log_dir)
        self.patterns = patterns or {}
        self._cache = {}

    def parse_file(self, filename):
        path = self.log_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        lines = path.read_text().splitlines()
        results = []

        for i, line in enumerate(lines):
            for name, pattern in self.patterns.items():
                match = re.search(pattern, line)
                if match:
                    severity = "error" if "error" in line.lower() else "warning" if "warn" in line.lower() else "info"
                    entry = {
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

    def get_summary(self, results):
        summary = {"total": len(results), "by_severity": {}, "by_pattern": {}}

        for entry in results:
            sev = entry["severity"]
            pat = entry["pattern"]
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            summary["by_pattern"][pat] = summary["by_pattern"].get(pat, 0) + 1

        return summary

    def __repr__(self):
        return f"LogParser(log_dir={self.log_dir}, patterns={len(self.patterns)})"


def find_anomalies(entries, threshold=3):
    if not entries:
        return []

    from collections import Counter
    pattern_counts = Counter(e["pattern"] for e in entries)
    mean = sum(pattern_counts.values()) / len(pattern_counts)
    std = (sum((v - mean) ** 2 for v in pattern_counts.values()) / len(pattern_counts)) ** 0.5

    anomalies = []
    for pattern, count in pattern_counts.items():
        if count > mean + threshold * std:
            anomalies.append({"pattern": pattern, "count": count, "z_score": (count - mean) / std if std > 0 else 0})

    return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)
