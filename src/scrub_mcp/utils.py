"""Utilities: batching, deterministic pre-filters, and helpers.

Pre-filter chain (deterministic, no LLM):
1. AST check: does the function have a docstring / annotations at all?
2. pydocstyle: does the existing docstring pass Google style validation?
3. pyright: do the existing type annotations actually typecheck?

If any check fails, the function gets sent to the DSPy module.
If all pass, the function is skipped entirely (zero LLM tokens).

File-level checks (1.3 optimization):
  _pydocstyle_file_check and _pyright_file_check run the tool ONCE on the
  real source file and map diagnostics to function line ranges. This replaces
  the old per-function stub approach (N subprocesses → 1 subprocess).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TypeVar

from scrub_mcp.models import ClassInfo, FunctionInfo

logger = logging.getLogger(__name__)

T = TypeVar("T")


def batch(items: list[T], size: int) -> list[list[T]]:
    """Split a list into batches of the given size."""
    return [items[i : i + size] for i in range(0, len(items), size)]


# ── Docstring pre-filters ──


def needs_docstring(func: FunctionInfo, use_pydocstyle: bool = True) -> bool:
    """Deterministic check: does this function need a (new or better) docstring?

    Two-tier check:
        1. AST: is a docstring present at all?
        2. pydocstyle: does the existing docstring pass Google convention?

    If pydocstyle is unavailable or the check errors out, falls back to
    AST-only (presence check).

    Args:
        func: Extracted function metadata.
        use_pydocstyle: If True, run pydocstyle on existing docstrings.

    Returns:
        True if the function needs a docstring generated or rewritten.

    """
    # Tier 1: no docstring at all
    if func.existing_docstring is None:
        return True

    # Tier 2: docstring exists but may be malformed
    if use_pydocstyle:
        return _pydocstyle_fails(func)

    return False


def _pydocstyle_fails(func: FunctionInfo) -> bool:
    """Run pydocstyle on a single function stub (legacy per-function path).

    Prefer _pydocstyle_file_check when processing a full source file.
    Returns True if the docstring has violations.
    """
    stub = f"{func.signature}:\n"
    if func.existing_docstring:
        stub += f'    """{func.existing_docstring}"""\n'
    stub += "    pass\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(stub)
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [
                "pydocstyle",
                "--convention=google",
                "--add-ignore=D100",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        tmp_path.unlink(missing_ok=True)

        if result.returncode == 0:
            return False

        violations = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.strip().startswith(str(tmp_path))
        ]
        if violations:
            logger.debug(
                "[pydocstyle] %s has %d violations: %s",
                func.name,
                len(violations),
                "; ".join(violations[:3]),
            )
            return True

        return False

    except FileNotFoundError:
        logger.debug("[pydocstyle] Not installed, falling back to AST-only check")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("[pydocstyle] Timed out checking %s", func.name)
        return False
    except Exception:
        logger.debug("[pydocstyle] Error checking %s, falling back to AST-only", func.name)
        return False


def _pydocstyle_file_check(
    source: str,
    functions: list[FunctionInfo],
    file_path: str | None = None,
) -> set[str]:
    """Run pydocstyle once on the full source file. Return names of functions with violations.

    One subprocess regardless of function count. Full module context preserved
    so import-dependent type hints don't cause false positives.
    """
    if not functions:
        return set()

    tmp_path: Path | None = None
    try:
        if file_path and Path(file_path).is_file():
            target = file_path
            cleanup = False
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(source)
                tmp_path = Path(tmp.name)
            target = str(tmp_path)
            cleanup = True

        result = subprocess.run(
            ["pydocstyle", "--convention=google", "--add-ignore=D100", target],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if cleanup and tmp_path:
            tmp_path.unlink(missing_ok=True)

        if result.returncode == 0:
            return set()

        # pydocstyle output lines alternate: "path:line ..." then "    Dxxx: message"
        # Extract line numbers from the path lines.
        failing: set[str] = set()
        _LINE_RE = re.compile(r":(\d+)\s+")
        for line in result.stdout.splitlines():
            m = _LINE_RE.search(line)
            if not m:
                continue
            diag_line = int(m.group(1))
            for func in functions:
                if func.line_start <= diag_line <= func.line_end:
                    failing.add(func.name)
                    break

        return failing

    except FileNotFoundError:
        logger.debug("[pydocstyle] Not installed, skipping file-level check")
        return set()
    except subprocess.TimeoutExpired:
        logger.warning("[pydocstyle] Timed out on file-level check")
        return set()
    except Exception:
        logger.debug("[pydocstyle] File-level check failed, falling back to empty set")
        return set()


# ── Type annotation pre-filters ──


def needs_type_annotations(func: FunctionInfo, use_pyright: bool = True) -> bool:
    """Deterministic check: does this function need type annotations?

    Two-tier check:
        1. AST: are all params + return annotated?
        2. pyright: do the existing annotations actually typecheck?

    Args:
        func: Extracted function metadata.
        use_pyright: If True, run pyright on already-annotated functions.

    Returns:
        True if the function needs type annotations generated or fixed.

    """
    sig_params = _extract_param_names(func.signature)

    # Tier 1: any unannotated params?
    for p in sig_params:
        if p not in func.existing_annotations:
            return True

    # Tier 1: missing return type?
    if "return" not in func.existing_annotations:
        return True

    # Tier 2: all annotated, but do they typecheck?
    if use_pyright:
        return _pyright_fails(func)

    return False


def _pyright_fails(func: FunctionInfo) -> bool:
    """Run pyright on a single function stub (legacy per-function path).

    Prefer _pyright_file_check when processing a full source file.
    Returns True if pyright reports type errors.
    """
    stub = "from __future__ import annotations\nfrom typing import Any\n\n"
    stub += f"{func.signature}:\n"
    if func.existing_docstring:
        stub += f'    """{func.existing_docstring}"""\n'
    stub += f"    {func.body}\n" if func.body.strip() else "    pass\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(stub)
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            ["pyright", "--outputjson", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        tmp_path.unlink(missing_ok=True)

        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            return False

        diagnostics = report.get("generalDiagnostics", [])
        type_errors = [d for d in diagnostics if d.get("severity", "") == "error"]

        if type_errors:
            logger.debug(
                "[pyright] %s has %d type errors: %s",
                func.name,
                len(type_errors),
                "; ".join(d.get("message", "")[:80] for d in type_errors[:3]),
            )
            return True

        return False

    except FileNotFoundError:
        logger.debug("[pyright] Not installed, falling back to AST-only check")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("[pyright] Timed out checking %s", func.name)
        return False
    except Exception:
        logger.debug("[pyright] Error checking %s, falling back to AST-only", func.name)
        return False


def _pyright_file_check(
    source: str,
    functions: list[FunctionInfo],
    file_path: str | None = None,
) -> set[str]:
    """Run pyright once on the full source file. Return names of functions with type errors.

    One subprocess regardless of function count. Full module context preserved
    so import-dependent type hints don't cause false positives.
    """
    if not functions:
        return set()

    tmp_path: Path | None = None
    try:
        if file_path and Path(file_path).is_file():
            target = file_path
            cleanup = False
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(source)
                tmp_path = Path(tmp.name)
            target = str(tmp_path)
            cleanup = True

        result = subprocess.run(
            ["pyright", "--outputjson", target],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if cleanup and tmp_path:
            tmp_path.unlink(missing_ok=True)

        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            return set()

        diagnostics = report.get("generalDiagnostics", [])
        failing: set[str] = set()

        for diag in diagnostics:
            if diag.get("severity", "") != "error":
                continue
            # pyright line numbers are 0-based in the JSON range
            diag_line = diag.get("range", {}).get("start", {}).get("line", -1) + 1
            if diag_line < 1:
                continue
            for func in functions:
                if func.line_start <= diag_line <= func.line_end:
                    failing.add(func.name)
                    logger.debug(
                        "[pyright] %s (L%d-%d) has error on L%d: %s",
                        func.name,
                        func.line_start,
                        func.line_end,
                        diag_line,
                        diag.get("message", "")[:80],
                    )
                    break

        return failing

    except FileNotFoundError:
        logger.debug("[pyright] Not installed, skipping file-level check")
        return set()
    except subprocess.TimeoutExpired:
        logger.warning("[pyright] Timed out on file-level check")
        return set()
    except Exception:
        logger.debug("[pyright] File-level check failed, falling back to empty set")
        return set()


# ── Class pre-filter ──


def needs_class_docstring(cls: ClassInfo, use_pydocstyle: bool = True) -> bool:
    """Deterministic check: does this class need a docstring?

    Args:
        cls: Extracted class metadata.
        use_pydocstyle: If True, validate existing docstring with pydocstyle.

    Returns:
        True if the class needs a docstring generated or rewritten.

    """
    if cls.existing_docstring is None:
        return True

    if use_pydocstyle:
        return _pydocstyle_fails_class(cls)

    return False


def _pydocstyle_fails_class(cls: ClassInfo) -> bool:
    """Run pydocstyle on a class stub to check docstring quality."""
    stub = f"class {cls.name}:\n"
    if cls.existing_docstring:
        stub += f'    """{cls.existing_docstring}"""\n'
    stub += "    pass\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(stub)
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [
                "pydocstyle",
                "--convention=google",
                "--add-ignore=D100",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        tmp_path.unlink(missing_ok=True)
        return result.returncode != 0

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception:
        return False


# ── Helpers ──


def _extract_param_names(signature: str) -> list[str]:
    """Pull parameter names from a function signature string, excluding self/cls."""
    try:
        params_str = signature.split("(", 1)[1].rstrip(")")
    except IndexError:
        return []

    params = []
    for p in params_str.split(","):
        name = p.strip().split(":")[0].split("=")[0].strip()
        if name and name not in ("self", "cls"):
            params.append(name)
    return params
