"""Diff-aware function narrowing for incremental pipeline runs.

Parses unified diff output (git diff, git diff --cached, git diff HEAD~N)
and narrows the pipeline's candidate function set to only those touched by the diff.

Usage:
    diff_text = get_git_diff(repo_path, since="HEAD")
    ranges = parse_diff(diff_text)
    narrowed, module_changed = intersect_with_functions(ranges, functions, source, file_path)
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scrub_mcp.models import ClassInfo, FunctionInfo

logger = logging.getLogger(__name__)


@dataclass
class ChangedRange:
    """A contiguous range of lines changed in a single file."""

    file_path: str
    start_line: int
    end_line: int  # sys.maxsize means "rest of file" (new file sentinel)


def parse_diff(diff_text: str) -> list[ChangedRange]:
    """Parse unified diff text into per-file line ranges.

    Handles standard git diff output including:
    - Normal hunks (@@...@@)
    - New files (--- /dev/null) → full-file range (end=sys.maxsize)
    - Deleted files (+++ /dev/null) → no ranges (skip entirely)
    - Renames (git diff -M) → full-file range for new path

    Args:
        diff_text: Unified diff text from git or similar tool.

    Returns:
        List of ChangedRange covering the post-change line numbers.

    """
    ranges: list[ChangedRange] = []
    current_file: str | None = None
    is_new_file = False
    is_deleted = False
    emitted_new_file = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            current_file = None
            is_new_file = False
            is_deleted = False
            emitted_new_file = False
        elif line.startswith("rename to "):
            # git diff -M rename: treat as full-file change on new path
            new_path = line[len("rename to "):].strip()
            if new_path:
                ranges.append(ChangedRange(new_path, 1, sys.maxsize))
                current_file = None  # suppress hunk processing for this file
        elif line.startswith("--- "):
            if line.startswith("--- /dev/null"):
                is_new_file = True
        elif line.startswith("+++ "):
            if line.startswith("+++ /dev/null"):
                is_deleted = True
            else:
                # Strip leading "b/" prefix that git adds
                path = line[4:]
                if path.startswith("b/"):
                    path = path[2:]
                # Strip trailing tab + metadata git sometimes appends
                path = path.split("\t")[0].strip()
                current_file = path
                if is_new_file and current_file and not emitted_new_file:
                    ranges.append(ChangedRange(current_file, 1, sys.maxsize))
                    emitted_new_file = True
                    current_file = None  # no further hunk processing needed
        elif line.startswith("@@") and current_file and not is_deleted:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) is not None else 1
                end = max(start, start + count - 1)
                ranges.append(ChangedRange(current_file, start, end))

    return ranges


def get_git_diff(repo_path: str, since: str = "HEAD") -> str:
    """Run git diff and return the unified diff text.

    Args:
        repo_path: Path to the git repository root (or any directory within it).
        since: Git ref to diff against. Defaults to HEAD (uncommitted changes).
               Use "HEAD~1" for the last commit, "main" for branch divergence, etc.

    Returns:
        Unified diff text. Empty string if no changes.

    Raises:
        RuntimeError: If git diff exits with a non-0/1 return code.

    """
    result = subprocess.run(
        ["git", "diff", "-M", since],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # git diff returns 1 when there are differences — that's normal
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git diff failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _paths_match(diff_path: str, file_path: str) -> bool:
    """Return True if diff_path and file_path refer to the same file.

    Handles the common case where diff_path is repo-relative
    (e.g. "src/scrub_mcp/pipeline.py") and file_path is absolute.
    Falls back to suffix matching when absolute resolution fails.
    """
    if not diff_path or not file_path or file_path == "<stdin>":
        return False

    # Try absolute resolution first
    try:
        abs_file = Path(file_path).resolve()
        # Try to resolve diff_path relative to cwd or as-is
        abs_diff = Path(diff_path).resolve()
        if abs_file == abs_diff:
            return True
    except Exception:
        pass

    # Suffix match: diff_path "src/foo/bar.py" matches "/any/prefix/src/foo/bar.py"
    norm_diff = diff_path.replace("\\", "/")
    norm_file = str(file_path).replace("\\", "/")
    return norm_file.endswith("/" + norm_diff) or norm_file == norm_diff


def intersect_with_functions(
    changed_ranges: list[ChangedRange],
    functions: list[FunctionInfo],
    source: str,
    file_path: str,
) -> tuple[list[FunctionInfo], bool]:
    """Narrow functions to only those touched by the diff.

    Also detects:
    - Class attribute changes: if a range covers class body outside any method,
      all methods of that class are included.
    - Module-level changes: if any changed lines fall outside every function's
      range, module_level_changed is True (triggers module docstring re-run).

    Args:
        changed_ranges: Ranges from parse_diff(), already filtered to this file.
        functions: Full list of FunctionInfo extracted from the file.
        source: Python source code (used to extract class info).
        file_path: Absolute path to the file being processed.

    Returns:
        (narrowed_functions, module_level_changed)

    """
    if not changed_ranges or not functions:
        return [], False

    # Collect all changed line numbers for module-level detection
    all_changed: set[int] = set()
    for r in changed_ranges:
        if r.end_line == sys.maxsize:
            # New file: all lines changed
            return list(functions), True
        all_changed.update(range(r.start_line, r.end_line + 1))

    # Direct overlap: range overlaps [func.line_start, func.line_end]
    included: set[str] = set()
    for func in functions:
        for r in changed_ranges:
            if r.start_line <= func.line_end and r.end_line >= func.line_start:
                included.add(func.name)
                break

    # Class attribute change: range covers class body but outside any method
    # → include all methods of that class
    try:
        from scrub_mcp.tools.parser import extract_classes

        classes: list[ClassInfo] = extract_classes(source)
        # Build a set of all function line ranges for fast lookup
        func_ranges = {(f.line_start, f.line_end) for f in functions}

        for cls in classes:
            cls_lines = set(range(cls.line_start, cls.line_end + 1))
            # Lines in the class body but not in any method body
            method_lines: set[int] = set()
            for f in functions:
                if f.parent_class == cls.name:
                    method_lines.update(range(f.line_start, f.line_end + 1))
            cls_only_lines = cls_lines - method_lines

            # If any changed line is in the class-only area, include all methods
            if cls_only_lines & all_changed:
                for f in functions:
                    if f.parent_class == cls.name:
                        included.add(f.name)
    except Exception:
        logger.debug("[diff] Class attribute detection failed, skipping", exc_info=True)

    # Module-level change detection: any changed line outside every function
    func_covered: set[int] = set()
    for func in functions:
        func_covered.update(range(func.line_start, func.line_end + 1))
    module_level_changed = bool(all_changed - func_covered)

    narrowed = [f for f in functions if f.name in included]
    logger.info(
        "[diff] Narrowed %d → %d functions; module_level_changed=%s",
        len(functions),
        len(narrowed),
        module_level_changed,
    )
    return narrowed, module_level_changed
