"""Vulnerability scanner. Cross-references SBOM PURLs against OSV.dev.

Deterministic, no LLM. Queries the OSV.dev API (Google's open
vulnerability database) which aggregates:
    - PyPI Advisory Database
    - GitHub Security Advisories (GHSA)
    - NVD (National Vulnerability Database)
    - OSS-Fuzz

Flow:
    1. Generate SBOM (or accept components directly)
    2. Batch-query OSV.dev with PURLs
    3. Map findings to CVE/GHSA IDs, severity (CVSS), and affected version ranges
    4. Return actionable report with upgrade paths

OSV.dev API is free, no API key required, rate-limited to 100 req/min.
Batch endpoint handles up to 1000 queries per call.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any

from scrub.models import (
    SBOMComponent,
    VulnFinding,
    VulnReport,
)

logger = logging.getLogger(__name__)

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_SINGLE_URL = "https://api.osv.dev/v1/query"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns"


def scan_components(
    components: list[SBOMComponent],
    batch_size: int = 100,
) -> VulnReport:
    """Scan SBOM components for known vulnerabilities via OSV.dev.

    Args:
        components: List of SBOM components with PURLs.
        batch_size: Components per API call. OSV supports up to 1000.

    Returns:
        VulnReport with findings grouped by severity.
    """
    # Filter to components with valid PURLs and real versions
    scannable = [
        c for c in components
        if c.purl
        and c.version
        and c.version not in ("unspecified", "")
        and "+" not in c.version  # Skip >= ranges
    ]

    if not scannable:
        return VulnReport(
            total_scanned=len(components),
            total_vulnerable=0,
            components_scanned=len(scannable),
            message="No components with exact versions to scan.",
        )

    logger.info("[vuln] Scanning %d components against OSV.dev", len(scannable))

    all_findings: list[VulnFinding] = []

    # Batch query
    for i in range(0, len(scannable), batch_size):
        batch = scannable[i : i + batch_size]
        findings = _query_osv_batch(batch)
        all_findings.extend(findings)

    # Deduplicate by (component, vuln_id)
    seen: set[tuple[str, str]] = set()
    unique_findings: list[VulnFinding] = []
    for f in all_findings:
        key = (f.component_name, f.vuln_id)
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    # Group by severity
    by_severity: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    for f in unique_findings:
        sev = f.severity.upper() if f.severity else "UNKNOWN"
        if sev in by_severity:
            by_severity[sev] += 1
        else:
            by_severity["UNKNOWN"] += 1

    vulnerable_components = len({f.component_name for f in unique_findings})

    return VulnReport(
        total_scanned=len(components),
        components_scanned=len(scannable),
        total_vulnerable=vulnerable_components,
        total_findings=len(unique_findings),
        by_severity=by_severity,
        findings=unique_findings,
    )


def _query_osv_batch(components: list[SBOMComponent]) -> list[VulnFinding]:
    """Query OSV.dev batch endpoint for vulnerabilities.

    Args:
        components: Batch of components to query.

    Returns:
        List of vulnerability findings.
    """
    queries = []
    for comp in components:
        queries.append({
            "package": {"purl": comp.purl},
            "version": comp.version,
        })

    payload = json.dumps({"queries": queries}).encode("utf-8")

    try:
        req = urllib.request.Request(
            OSV_BATCH_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        logger.error("[vuln] OSV.dev API error: %s", e)
        return []
    except Exception:
        logger.exception("[vuln] Failed to query OSV.dev")
        return []

    findings: list[VulnFinding] = []
    results = data.get("results", [])

    for comp, result in zip(components, results):
        vulns = result.get("vulns", [])
        if not vulns:
            continue

        for vuln in vulns:
            finding = _parse_osv_vuln(vuln, comp)
            if finding:
                findings.append(finding)

    return findings


def _parse_osv_vuln(vuln: dict[str, Any], comp: SBOMComponent) -> VulnFinding | None:
    """Parse a single OSV vulnerability response into a VulnFinding.

    Args:
        vuln: Raw OSV vulnerability object.
        comp: The component this vulnerability applies to.

    Returns:
        VulnFinding or None if parsing fails.
    """
    vuln_id = vuln.get("id", "")
    aliases = vuln.get("aliases", [])

    # Extract CVE ID from aliases if available
    cve_id = ""
    ghsa_id = ""
    for alias in aliases:
        if alias.startswith("CVE-"):
            cve_id = alias
        elif alias.startswith("GHSA-"):
            ghsa_id = alias

    # Summary
    summary = vuln.get("summary", "")
    details = vuln.get("details", "")

    # Severity from CVSS
    severity = "UNKNOWN"
    cvss_score = 0.0
    for sev_entry in vuln.get("severity", []):
        if sev_entry.get("type") == "CVSS_V3":
            cvss_vector = sev_entry.get("score", "")
            cvss_score = _extract_cvss_base_score(cvss_vector)
            severity = _cvss_to_severity(cvss_score)
            break

    # If no CVSS, try database_specific severity
    if severity == "UNKNOWN":
        db_sev = vuln.get("database_specific", {}).get("severity")
        if db_sev:
            severity = db_sev.upper()

    # Fixed version (upgrade path)
    fixed_version = ""
    for affected in vuln.get("affected", []):
        for rng in affected.get("ranges", []):
            for event in rng.get("events", []):
                if "fixed" in event:
                    fixed_version = event["fixed"]
                    break

    # References
    references = [
        ref.get("url", "")
        for ref in vuln.get("references", [])
        if ref.get("url")
    ][:5]  # Cap at 5 URLs

    return VulnFinding(
        vuln_id=vuln_id,
        cve_id=cve_id,
        ghsa_id=ghsa_id,
        component_name=comp.name,
        component_version=comp.version,
        component_purl=comp.purl,
        severity=severity,
        cvss_score=cvss_score,
        summary=summary,
        details=details[:500],
        fixed_version=fixed_version,
        references=references,
        aliases=aliases,
    )


def _extract_cvss_base_score(cvss_vector: str) -> float:
    """Extract the base score from a CVSS v3 vector string.

    OSV sometimes returns the full vector, sometimes just the score.
    """
    if not cvss_vector:
        return 0.0

    # If it's just a number
    try:
        return float(cvss_vector)
    except ValueError:
        pass

    # CVSS vector format: CVSS:3.1/AV:N/AC:L/... doesn't include score directly
    # OSV typically provides the score as a float, but handle both
    return 0.0


def _cvss_to_severity(score: float) -> str:
    """Map CVSS base score to severity label."""
    if score >= 9.0:
        return "CRITICAL"
    elif score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    elif score > 0.0:
        return "LOW"
    return "UNKNOWN"
