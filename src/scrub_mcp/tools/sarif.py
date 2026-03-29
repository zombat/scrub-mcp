"""SARIF 2.1.0 serializer for S.C.R.U.B. findings.

Maps violation dicts produced by the CLI commands to the SARIF schema used
by GitHub Code Scanning and other SAST integrations.

Rule ID scheme:
    SCRUB-DOC-001   Missing function docstring
    SCRUB-DOC-002   Missing class docstring
    SCRUB-DOC-003   Missing module docstring
    SCRUB-TYPE-001  Missing type annotation
    SCRUB-CMPLX-001 High cyclomatic complexity
    SCRUB-CMPLX-002 High cognitive complexity
    SCRUB-SEC-{ID}  Bandit finding  (e.g. SCRUB-SEC-B101)
    SCRUB-VULN-{ID} OSV vulnerability (e.g. SCRUB-VULN-CVE-2024-1234)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/"
    "Documents/CommitteeSpecifications/2.1.0/sarif-schema-2.1.0.json"
)

_STATIC_RULES: dict[str, dict] = {
    "SCRUB-DOC-001": {
        "name": "MissingFunctionDocstring",
        "shortDescription": "Function is missing a docstring.",
        "helpUri": "https://peps.python.org/pep-0257/",
        "defaultLevel": "warning",
    },
    "SCRUB-DOC-002": {
        "name": "MissingClassDocstring",
        "shortDescription": "Class is missing a docstring.",
        "helpUri": "https://peps.python.org/pep-0257/",
        "defaultLevel": "warning",
    },
    "SCRUB-DOC-003": {
        "name": "MissingModuleDocstring",
        "shortDescription": "Module is missing a docstring.",
        "helpUri": "https://peps.python.org/pep-0257/",
        "defaultLevel": "note",
    },
    "SCRUB-TYPE-001": {
        "name": "MissingTypeAnnotation",
        "shortDescription": "Function is missing type annotations.",
        "helpUri": "https://peps.python.org/pep-0484/",
        "defaultLevel": "warning",
    },
    "SCRUB-CMPLX-001": {
        "name": "HighCyclomaticComplexity",
        "shortDescription": "Function has high cyclomatic complexity.",
        "helpUri": "https://en.wikipedia.org/wiki/Cyclomatic_complexity",
        "defaultLevel": "warning",
    },
    "SCRUB-CMPLX-002": {
        "name": "HighCognitiveComplexity",
        "shortDescription": "Function has high cognitive complexity.",
        "helpUri": "https://www.sonarsource.com/docs/CognitiveComplexity.pdf",
        "defaultLevel": "note",
    },
}

_SEVERITY_TO_LEVEL: dict[str, str] = {
    "HIGH": "error",
    "MEDIUM": "warning",
    "LOW": "note",
    "CRITICAL": "error",
    "UNKNOWN": "note",
}


def _tool_version() -> str:
    try:
        return pkg_version("scrub-mcp")
    except PackageNotFoundError:
        return "0.1.0"


def _rule_descriptor(rule_id: str) -> dict:
    """Return a SARIF reportingDescriptor for *rule_id*."""
    if rule_id in _STATIC_RULES:
        r = _STATIC_RULES[rule_id]
        return {
            "id": rule_id,
            "name": r["name"],
            "shortDescription": {"text": r["shortDescription"]},
            "defaultConfiguration": {"level": r["defaultLevel"]},
            "helpUri": r["helpUri"],
        }
    if rule_id.startswith("SCRUB-SEC-"):
        test_id = rule_id[len("SCRUB-SEC-") :]
        return {
            "id": rule_id,
            "name": f"BanditFinding{test_id}",
            "shortDescription": {"text": f"Bandit security finding: {test_id}"},
            "defaultConfiguration": {"level": "warning"},
            "helpUri": (f"https://bandit.readthedocs.io/en/latest/plugins/{test_id.lower()}.html"),
        }
    if rule_id.startswith("SCRUB-VULN-"):
        vuln_id = rule_id[len("SCRUB-VULN-") :]
        return {
            "id": rule_id,
            "name": f"Vulnerability{vuln_id.replace('-', '_')}",
            "shortDescription": {"text": f"Known vulnerability: {vuln_id}"},
            "defaultConfiguration": {"level": "error"},
            "helpUri": f"https://osv.dev/vulnerability/{vuln_id}",
        }
    return {
        "id": rule_id,
        "name": rule_id,
        "shortDescription": {"text": rule_id},
        "defaultConfiguration": {"level": "warning"},
    }


def to_sarif(violations: list[dict], tool_version: str | None = None) -> dict:
    """Convert S.C.R.U.B. violations to a SARIF 2.1.0 document.

    Args:
        violations: List of violation dicts.  Each dict must contain:
            - rule_id   (str)  e.g. "SCRUB-DOC-001"
            - file      (str)  relative path to the source file
            - line_start (int) first line of the finding (1-based)
            - line_end   (int) last line (inclusive)
            - message   (str)  human-readable description
            - level     (str)  "error", "warning", or "note"
            - function_name (str, optional)  logical name for the location
        tool_version: Override the version string embedded in the driver.

    Returns:
        SARIF 2.1.0 document as a plain dict (serialize with json.dumps).

    """
    version = tool_version or _tool_version()

    # Build deduplicated rule list in encounter order.
    seen_ids: set[str] = set()
    rules: list[dict] = []
    for v in violations:
        rid = v.get("rule_id", "SCRUB-UNKNOWN")
        if rid not in seen_ids:
            seen_ids.add(rid)
            rules.append(_rule_descriptor(rid))

    results: list[dict] = []
    for v in violations:
        start = max(v.get("line_start", 1), 1)
        end = max(v.get("line_end", start), start)

        physical: dict = {
            "artifactLocation": {
                "uri": v.get("file", ""),
                "uriBaseId": "%SRCROOT%",
            },
            "region": {
                "startLine": start,
                "endLine": end,
            },
        }

        location: dict = {"physicalLocation": physical}
        fn = v.get("function_name")
        if fn:
            location["logicalLocations"] = [{"name": fn, "kind": "function"}]

        results.append(
            {
                "ruleId": v.get("rule_id", "SCRUB-UNKNOWN"),
                "level": v.get("level", "warning"),
                "message": {"text": v.get("message", "")},
                "locations": [location],
            }
        )

    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "S.C.R.U.B.",
                        "version": version,
                        "informationUri": "https://github.com/zombat/scrub-mcp",
                        "rules": rules,
                    }
                },
                "results": results,
                "columnKind": "unicodeCodePoints",
                "originalUriBaseIds": {
                    "%SRCROOT%": {"uri": ""},
                },
            }
        ],
    }
