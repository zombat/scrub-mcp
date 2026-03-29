"""File-system traversal that respects .gitignore via pathspec."""

from __future__ import annotations

import os
from pathlib import Path

import pathspec


def get_tracked_files(
    root_dir: str | Path,
    extra_excludes: list[str] | None = None,
) -> list[Path]:
    """Return all .py files under *root_dir* that are not gitignored.

    Reads the project's ``.gitignore`` (if present) and prunes both
    directories and files that match.  Additional glob patterns in
    *extra_excludes* (from ``config.yaml`` ``exclude_paths``) are
    appended to the same spec so everything stays in one place.

    Args:
        root_dir: Project root — the directory that contains ``.gitignore``.
        extra_excludes: Additional gitignore-style patterns to exclude,
            e.g. ``["tests/fixtures/**", "legacy_api/**"]``.

    Returns:
        Sorted list of absolute ``Path`` objects for every non-ignored
        ``.py`` file found under *root_dir*.
    """
    root = Path(root_dir).resolve()

    lines: list[str] = []
    gitignore = root / ".gitignore"
    if gitignore.is_file():
        lines.extend(gitignore.read_text(encoding="utf-8").splitlines())
    if extra_excludes:
        lines.extend(extra_excludes)

    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, lines)

    results: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        # Prune ignored directories in-place so os.walk skips them entirely.
        dirs[:] = [
            d for d in dirs
            if not spec.match_file(
                os.path.relpath(os.path.join(dirpath, d), root) + "/"
            )
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fname), root)
            if not spec.match_file(rel):
                results.append(Path(dirpath) / fname)

    return sorted(results)
