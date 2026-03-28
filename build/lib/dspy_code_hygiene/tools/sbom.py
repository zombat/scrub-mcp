"""SBOM generator. CycloneDX and SPDX from Python project metadata.

Deterministic: parses pyproject.toml, requirements.txt, pip freeze,
and lock files to build a complete Software Bill of Materials.

Supports two formats:
    - CycloneDX 1.5 (JSON): industry standard for vulnerability tracking
    - SPDX 2.3 (JSON): ISO/IEC 5962:2021, required by many compliance frameworks

No LLM needed. This is pure metadata extraction and formatting.

Sources (checked in priority order):
    1. pip freeze (installed packages, exact versions)
    2. pyproject.toml [project.dependencies]
    3. requirements.txt / requirements/*.txt
    4. poetry.lock / Pipfile.lock / uv.lock
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dspy_code_hygiene.models import SBOMComponent, SBOMReport

logger = logging.getLogger(__name__)


# ── Dependency extraction ──


def extract_pip_freeze() -> list[SBOMComponent]:
    """Get installed packages from pip freeze. Most accurate source."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return _parse_freeze_output(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("[sbom] pip freeze failed, skipping")
        return []


def extract_from_pyproject(pyproject_path: Path) -> list[SBOMComponent]:
    """Parse dependencies from pyproject.toml."""
    if not pyproject_path.exists():
        return []

    try:
        # Use tomllib (3.11+) or tomli
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        data = tomllib.loads(pyproject_path.read_text())
    except Exception:
        logger.warning("[sbom] Failed to parse %s", pyproject_path)
        return []

    components = []

    # [project.dependencies]
    for dep in data.get("project", {}).get("dependencies", []):
        comp = _parse_requirement_string(dep)
        if comp:
            comp.source = "pyproject.toml"
            components.append(comp)

    # [project.optional-dependencies]
    for group, deps in data.get("project", {}).get("optional-dependencies", {}).items():
        for dep in deps:
            comp = _parse_requirement_string(dep)
            if comp:
                comp.source = f"pyproject.toml[{group}]"
                comp.optional = True
                components.append(comp)

    return components


def extract_from_requirements(req_path: Path) -> list[SBOMComponent]:
    """Parse dependencies from requirements.txt."""
    if not req_path.exists():
        return []

    components = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        comp = _parse_requirement_string(line)
        if comp:
            comp.source = str(req_path)
            components.append(comp)

    return components


def extract_from_lockfile(lock_path: Path) -> list[SBOMComponent]:
    """Parse dependencies from poetry.lock, Pipfile.lock, or uv.lock."""
    if not lock_path.exists():
        return []

    name = lock_path.name
    try:
        if name == "poetry.lock":
            return _parse_poetry_lock(lock_path)
        elif name == "Pipfile.lock":
            return _parse_pipfile_lock(lock_path)
        elif name == "uv.lock":
            return _parse_uv_lock(lock_path)
    except Exception:
        logger.warning("[sbom] Failed to parse %s", lock_path)

    return []


# ── SBOM generation ──


def generate_sbom(
    project_dir: Path | None = None,
    format: str = "cyclonedx",
    include_pip: bool = True,
    tool_name: str = "dspy-code-hygiene",
    tool_version: str = "0.1.0",
) -> SBOMReport:
    """Generate a complete SBOM from all available sources.

    Args:
        project_dir: Root of the Python project. Defaults to cwd.
        format: Output format: "cyclonedx" or "spdx".
        include_pip: Include pip freeze output (most accurate).
        tool_name: Name of the generating tool.
        tool_version: Version of the generating tool.

    Returns:
        SBOMReport with components and formatted output.
    """
    project = project_dir or Path.cwd()
    all_components: dict[str, SBOMComponent] = {}

    # Priority 1: pip freeze (exact installed versions)
    if include_pip:
        for comp in extract_pip_freeze():
            all_components[comp.name.lower()] = comp

    # Priority 2: pyproject.toml
    pyproject = project / "pyproject.toml"
    for comp in extract_from_pyproject(pyproject):
        key = comp.name.lower()
        if key not in all_components:
            all_components[key] = comp

    # Priority 3: requirements files
    for req_file in [
        project / "requirements.txt",
        project / "requirements" / "base.txt",
        project / "requirements" / "prod.txt",
    ]:
        for comp in extract_from_requirements(req_file):
            key = comp.name.lower()
            if key not in all_components:
                all_components[key] = comp

    # Priority 4: lock files
    for lock_file in [
        project / "poetry.lock",
        project / "Pipfile.lock",
        project / "uv.lock",
    ]:
        for comp in extract_from_lockfile(lock_file):
            key = comp.name.lower()
            if key not in all_components:
                all_components[key] = comp

    components = sorted(all_components.values(), key=lambda c: c.name.lower())

    # Generate formatted output
    if format == "spdx":
        output = _format_spdx(components, tool_name, tool_version)
    else:
        output = _format_cyclonedx(components, tool_name, tool_version)

    return SBOMReport(
        format=format,
        component_count=len(components),
        components=components,
        output_json=output,
    )


# ── CycloneDX 1.5 formatter ──


def _format_cyclonedx(
    components: list[SBOMComponent],
    tool_name: str,
    tool_version: str,
) -> str:
    """Format components as CycloneDX 1.5 JSON."""
    serial = f"urn:uuid:{uuid4()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    bom: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": serial,
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": tool_name,
                        "version": tool_version,
                    }
                ]
            },
        },
        "components": [],
    }

    for comp in components:
        cdx_comp: dict[str, Any] = {
            "type": "library",
            "name": comp.name,
            "version": comp.version,
            "purl": comp.purl,
        }

        if comp.license_id:
            cdx_comp["licenses"] = [{"license": {"id": comp.license_id}}]

        if comp.sha256:
            cdx_comp["hashes"] = [{"alg": "SHA-256", "content": comp.sha256}]

        if comp.source:
            cdx_comp["properties"] = [
                {"name": "source", "value": comp.source},
            ]
            if comp.optional:
                cdx_comp["properties"].append(
                    {"name": "optional", "value": "true"}
                )

        bom["components"].append(cdx_comp)

    return json.dumps(bom, indent=2)


# ── SPDX 2.3 formatter ──


def _format_spdx(
    components: list[SBOMComponent],
    tool_name: str,
    tool_version: str,
) -> str:
    """Format components as SPDX 2.3 JSON."""
    doc_namespace = f"https://spdx.org/spdxdocs/{tool_name}-{uuid4()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    spdx: dict[str, Any] = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{tool_name}-sbom",
        "documentNamespace": doc_namespace,
        "creationInfo": {
            "created": timestamp,
            "creators": [f"Tool: {tool_name}-{tool_version}"],
            "licenseListVersion": "3.22",
        },
        "packages": [],
        "relationships": [],
    }

    for i, comp in enumerate(components):
        spdx_id = f"SPDXRef-Package-{i + 1}"
        pkg: dict[str, Any] = {
            "SPDXID": spdx_id,
            "name": comp.name,
            "versionInfo": comp.version,
            "downloadLocation": f"https://pypi.org/project/{comp.name}/{comp.version}/",
            "filesAnalyzed": False,
            "supplier": "NOASSERTION",
        }

        if comp.purl:
            pkg["externalRefs"] = [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": comp.purl,
                }
            ]

        if comp.license_id:
            pkg["licenseConcluded"] = comp.license_id
            pkg["licenseDeclared"] = comp.license_id
        else:
            pkg["licenseConcluded"] = "NOASSERTION"
            pkg["licenseDeclared"] = "NOASSERTION"

        if comp.sha256:
            pkg["checksums"] = [{"algorithm": "SHA256", "checksumValue": comp.sha256}]

        spdx["packages"].append(pkg)
        spdx["relationships"].append({
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relatedSpdxElement": spdx_id,
            "relationshipType": "DESCRIBES",
        })

    return json.dumps(spdx, indent=2)


# ── Parsers ──


def _parse_freeze_output(output: str) -> list[SBOMComponent]:
    """Parse pip freeze output into components."""
    components = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle editable installs
        if line.startswith("-e "):
            continue

        match = re.match(r"^([a-zA-Z0-9_.-]+)==([^\s;]+)", line)
        if match:
            name, version = match.group(1), match.group(2)
            components.append(
                SBOMComponent(
                    name=name,
                    version=version,
                    purl=f"pkg:pypi/{name.lower()}@{version}",
                    source="pip freeze",
                )
            )

    return components


def _parse_requirement_string(req: str) -> SBOMComponent | None:
    """Parse a PEP 508 requirement string."""
    req = req.strip()
    if not req or req.startswith("#"):
        return None

    # Strip extras, markers, and environment markers
    req = re.split(r"\s*;", req)[0].strip()
    req = re.sub(r"\[.*?\]", "", req)

    # Match name + version specifier
    match = re.match(r"^([a-zA-Z0-9_.-]+)\s*(?:([><=!~]+)\s*([^\s,]+))?", req)
    if not match:
        return None

    name = match.group(1)
    version = match.group(3) or "unspecified"
    op = match.group(2) or ""

    # For >= or ~=, note that this is a minimum, not exact
    if op in (">=", "~=", ">"):
        version = f"{version}+"

    return SBOMComponent(
        name=name,
        version=version,
        purl=f"pkg:pypi/{name.lower()}@{version.rstrip('+')}",
    )


def _parse_poetry_lock(lock_path: Path) -> list[SBOMComponent]:
    """Parse poetry.lock (TOML format)."""
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        data = tomllib.loads(lock_path.read_text())
    except Exception:
        return []

    components = []
    for pkg in data.get("package", []):
        name = pkg.get("name", "")
        version = pkg.get("version", "")
        if name and version:
            components.append(
                SBOMComponent(
                    name=name,
                    version=version,
                    purl=f"pkg:pypi/{name.lower()}@{version}",
                    source="poetry.lock",
                )
            )

    return components


def _parse_pipfile_lock(lock_path: Path) -> list[SBOMComponent]:
    """Parse Pipfile.lock (JSON format)."""
    data = json.loads(lock_path.read_text())
    components = []

    for section in ("default", "develop"):
        for name, info in data.get(section, {}).items():
            version = info.get("version", "").lstrip("=")
            sha = ""
            for h in info.get("hashes", []):
                if h.startswith("sha256:"):
                    sha = h[7:]
                    break

            components.append(
                SBOMComponent(
                    name=name,
                    version=version,
                    purl=f"pkg:pypi/{name.lower()}@{version}",
                    sha256=sha,
                    source=f"Pipfile.lock[{section}]",
                    optional=section == "develop",
                )
            )

    return components


def _parse_uv_lock(lock_path: Path) -> list[SBOMComponent]:
    """Parse uv.lock (TOML format)."""
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        data = tomllib.loads(lock_path.read_text())
    except Exception:
        return []

    components = []
    for pkg in data.get("package", []):
        name = pkg.get("name", "")
        version = pkg.get("version", "")
        if name and version:
            components.append(
                SBOMComponent(
                    name=name,
                    version=version,
                    purl=f"pkg:pypi/{name.lower()}@{version}",
                    source="uv.lock",
                )
            )

    return components
