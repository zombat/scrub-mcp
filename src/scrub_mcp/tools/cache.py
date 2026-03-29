"""Composite hash artifact cache for the hygiene pipeline.

Three-layer composite hash:
  Layer 1: sha256(signature + body)  — changes when the function's own code changes
  Layer 2: mix in sha256(sorted local import file contents)  — catches dep changes
  Final:   sha256(layer2 + artifact_type + model_fingerprint)  — type- and model-specific

Storage: {cache_dir}/{hash[:2]}/{hash}.json   (two-level bucketing for inode efficiency)

Cache entries are validated against the current pre-filter tools before being used:
  docstring → pydocstyle passes on the cached artifact
  type      → pyright passes on the cached artifact
  comment   → cyclomatic complexity unchanged (within ±1)
  test      → ast.parse succeeds on the cached test code
  lint/security → not cached (always fast to re-run)

LRU eviction is file-mtime-based. Touch the file on each read to maintain recency.
Eviction runs at most once per pipeline invocation, not per write.

Thread safety: evict_if_needed is guarded by a module-level Lock. resolve_local_imports
mutates sys.path and must NOT be called concurrently — call it before ThreadPoolExecutor.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from scrub_mcp.models import FunctionInfo

logger = logging.getLogger(__name__)

_EVICT_LOCK = threading.Lock()

# Artifact types that are never cached (deterministic tools are fast enough)
_UNCACHED_TYPES = frozenset({"lint", "security"})


class CacheEntry(BaseModel):
    """A single cached artifact produced by a DSPy module."""

    composite_hash: str
    artifact_type: str
    schema_version: int
    timestamp: str = Field(description="ISO 8601 UTC")
    generated_artifact: str = Field(description="JSON-serialized artifact")
    model_fingerprint: str = Field(description="{provider}/{model}")
    metadata: dict = Field(default_factory=dict, description="e.g. cyclomatic_complexity for comments")


# ── Hash computation ──


def _layer1_hash(func: FunctionInfo) -> str:
    """sha256 of normalised signature + body (trailing whitespace stripped per line)."""
    sig = "\n".join(l.rstrip() for l in func.signature.splitlines())
    body = "\n".join(l.rstrip() for l in func.body.splitlines())
    return hashlib.sha256(f"{sig}\n{body}".encode()).hexdigest()


def _layer2_hash(layer1: str, source: str, project_root: str) -> str:
    """Mix in a hash of local import file contents."""
    local_files = resolve_local_imports(source, project_root)
    import_hash = hash_local_files(local_files)
    return hashlib.sha256(f"{layer1}:{import_hash}".encode()).hexdigest()


def compute_composite_hash(
    func: FunctionInfo,
    artifact_type: str,
    source: str,
    project_root: str,
    model_fingerprint: str,
) -> str:
    """Compute the three-layer composite cache key for a function artifact.

    Args:
        func: Function to hash.
        artifact_type: One of docstring, type, comment, test.
        source: Full source of the file (used for import resolution).
        project_root: Project root directory for local import resolution.
        model_fingerprint: "{provider}/{model}" string binding the key to the model.

    Returns:
        64-character hex SHA-256 hash.

    """
    l1 = _layer1_hash(func)
    l2 = _layer2_hash(l1, source, project_root)
    final = hashlib.sha256(f"{l2}:{artifact_type}:{model_fingerprint}".encode()).hexdigest()
    return final


def resolve_local_imports(source: str, project_root: str) -> list[str]:
    """Find local Python files imported by the given source.

    Uses importlib.util.find_spec with project_root prepended to sys.path.
    Only returns files within the project tree (stdlib and site-packages excluded).

    WARNING: Not thread-safe — mutates sys.path. Call before ThreadPoolExecutor.

    Args:
        source: Python source code to inspect.
        project_root: Root directory of the project.

    Returns:
        Sorted list of absolute paths to local imported files.

    """
    root = Path(project_root).resolve()
    found: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    old_path = sys.path[:]
    sys.path.insert(0, str(root))
    try:
        for node in ast.walk(tree):
            module_name: str | None = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    _check_and_add(alias.name, root, found)
                continue

            if module_name:
                _check_and_add(module_name, root, found)
    finally:
        sys.path[:] = old_path

    return sorted(set(found))


def _check_and_add(module_name: str, root: Path, found: list[str]) -> None:
    """Resolve a module name and add its path if it's within the project root."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            origin = Path(spec.origin).resolve()
            try:
                origin.relative_to(root)
                found.append(str(origin))
            except ValueError:
                pass  # stdlib or site-packages
    except (ImportError, ModuleNotFoundError, ValueError):
        pass
    except Exception:
        pass


def hash_local_files(file_paths: list[str]) -> str:
    """Compute a deterministic hash over the contents of local files.

    Files are sorted by path before hashing, so the result is order-independent.

    Args:
        file_paths: Absolute paths to files to hash.

    Returns:
        64-character hex SHA-256 hash. Empty-string hash if no files.

    """
    h = hashlib.sha256()
    for path in sorted(file_paths):
        try:
            h.update(Path(path).read_bytes())
        except OSError:
            pass
    return h.hexdigest()


# ── Storage ──


def _cache_path(composite_hash: str, cache_dir: str) -> Path:
    return Path(cache_dir) / composite_hash[:2] / f"{composite_hash}.json"


def write_cache(entry: CacheEntry, cache_dir: str) -> None:
    """Write a cache entry to disk.

    Args:
        entry: The artifact to cache.
        cache_dir: Root cache directory.

    """
    path = _cache_path(entry.composite_hash, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry.model_dump_json(), encoding="utf-8")
    except Exception:
        logger.debug("[cache] Failed to write %s", path, exc_info=True)


def read_cache(
    composite_hash: str,
    artifact_type: str,
    cache_dir: str,
) -> CacheEntry | None:
    """Read a cache entry from disk, returning None on any miss or error.

    Touches the file on a successful hit to maintain LRU recency.

    Args:
        composite_hash: The hash key.
        artifact_type: Expected artifact type (guards against hash collisions).
        cache_dir: Root cache directory.

    Returns:
        CacheEntry or None.

    """
    path = _cache_path(composite_hash, cache_dir)
    if not path.exists():
        return None
    try:
        entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        if entry.artifact_type != artifact_type:
            return None  # type mismatch (shouldn't happen, but be safe)
        # Touch to update mtime → LRU proxy
        path.touch()
        return entry
    except Exception:
        logger.debug("[cache] Failed to read %s", path, exc_info=True)
        return None


def evict_if_needed(cache_dir: str, max_size_mb: int) -> None:
    """Remove oldest cache entries until total size is under max_size_mb.

    Uses file mtime as the LRU proxy (read_cache touches on hit).
    Thread-safe: guarded by _EVICT_LOCK.

    Args:
        cache_dir: Root cache directory.
        max_size_mb: Maximum allowed total size in megabytes.

    """
    with _EVICT_LOCK:
        _evict_locked(cache_dir, max_size_mb)


def _evict_locked(cache_dir: str, max_size_mb: int) -> None:
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return

    entries = []
    total_bytes = 0
    for f in cache_path.rglob("*.json"):
        try:
            stat = f.stat()
            entries.append((stat.st_mtime, stat.st_size, f))
            total_bytes += stat.st_size
        except OSError:
            pass

    max_bytes = max_size_mb * 1024 * 1024
    if total_bytes <= max_bytes:
        return

    entries.sort(key=lambda x: x[0])  # oldest first
    evicted = 0
    for _mtime, size, path in entries:
        if total_bytes <= max_bytes:
            break
        try:
            path.unlink(missing_ok=True)
            total_bytes -= size
            evicted += 1
        except OSError:
            pass

    if evicted:
        logger.info("[cache] Evicted %d entries to stay under %d MB", evicted, max_size_mb)


# ── Lookup and validation ──


def lookup_cached_artifact(
    func: FunctionInfo,
    artifact_type: str,
    source: str,
    project_root: str,
    model_fingerprint: str,
    cache_enabled: bool,
    cache_dir: str,
) -> CacheEntry | None:
    """Look up a cached artifact for a function.

    Args:
        func: The function to look up.
        artifact_type: One of docstring, type, comment, test.
        source: Full source of the file (for import hashing).
        project_root: Project root directory.
        model_fingerprint: "{provider}/{model}" string.
        cache_enabled: If False, always returns None.
        cache_dir: Root cache directory.

    Returns:
        CacheEntry or None.

    """
    if not cache_enabled or artifact_type in _UNCACHED_TYPES:
        return None
    h = compute_composite_hash(func, artifact_type, source, project_root, model_fingerprint)
    return read_cache(h, artifact_type, cache_dir)


def validate_cached_artifact(
    entry: CacheEntry,
    func: FunctionInfo,
    schema_version: int,
) -> bool:
    """Validate a cache entry before using it.

    Per-artifact-type gates:
      docstring: pydocstyle passes on the function stub with the cached docstring
      type:      pyright passes on the function stub with the cached annotations
      comment:   cyclomatic_complexity within ±1 of cached value
      test:      ast.parse succeeds on the cached test code

    Args:
        entry: The cache entry to validate.
        func: The function the entry was retrieved for.
        schema_version: Current config schema version.

    Returns:
        True if the entry is valid and safe to use.

    """
    if entry.schema_version != schema_version:
        logger.debug("[cache] Schema version mismatch for %s (%d vs %d)",
                     func.name, entry.schema_version, schema_version)
        return False

    if entry.artifact_type == "docstring":
        return _validate_docstring_entry(entry, func)
    if entry.artifact_type == "type":
        return _validate_type_entry(entry, func)
    if entry.artifact_type == "comment":
        return _validate_comment_entry(entry, func)
    if entry.artifact_type == "test":
        return _validate_test_entry(entry)

    return False


def _validate_docstring_entry(entry: CacheEntry, func: FunctionInfo) -> bool:
    """Validate cached docstring via pydocstyle on a function stub."""
    try:
        from scrub_mcp.utils import _pydocstyle_fails

        # Build a FunctionInfo with the cached docstring injected
        func_with_doc = func.model_copy(
            update={"existing_docstring": entry.generated_artifact}
        )
        return not _pydocstyle_fails(func_with_doc)
    except Exception:
        logger.debug("[cache] Docstring validation failed for %s", func.name, exc_info=True)
        return True  # fail open: use cached entry rather than re-running LLM


def _validate_type_entry(entry: CacheEntry, func: FunctionInfo) -> bool:
    """Validate cached type annotations via pyright on a function stub."""
    try:
        import json
        from scrub_mcp.utils import _pyright_fails
        from scrub_mcp.models import TypeAnnotation

        ann = TypeAnnotation.model_validate_json(entry.generated_artifact)
        # Build a FunctionInfo with the cached annotations injected
        merged_annotations = dict(func.existing_annotations)
        merged_annotations.update(ann.parameters)
        merged_annotations["return"] = ann.return_type
        func_with_types = func.model_copy(
            update={"existing_annotations": merged_annotations}
        )
        return not _pyright_fails(func_with_types)
    except Exception:
        logger.debug("[cache] Type validation failed for %s", func.name, exc_info=True)
        return True  # fail open


def _validate_comment_entry(entry: CacheEntry, func: FunctionInfo) -> bool:
    """Validate cached comment by checking that complexity hasn't shifted."""
    cached_cc = entry.metadata.get("cyclomatic_complexity")
    if cached_cc is None:
        return True  # no metadata → accept (legacy entry)
    return abs(func.cyclomatic_complexity - int(cached_cc)) <= 1


def _validate_test_entry(entry: CacheEntry) -> bool:
    """Validate cached test code by parsing it with ast."""
    try:
        ast.parse(entry.generated_artifact)
        return True
    except SyntaxError:
        return False


# ── Cache stats (used by CLI) ──


def cache_stats(cache_dir: str) -> dict:
    """Collect statistics about the cache directory.

    Args:
        cache_dir: Root cache directory.

    Returns:
        Dict with total_entries, total_size_mb, by_artifact_type,
        oldest_timestamp, stale_count (always 0 here; caller compares model_fp).

    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return {
            "total_entries": 0,
            "total_size_mb": 0.0,
            "by_artifact_type": {},
            "oldest_timestamp": None,
            "entries": [],
        }

    entries_data = []
    total_bytes = 0
    by_type: dict[str, int] = {}

    for f in cache_path.rglob("*.json"):
        try:
            entry = CacheEntry.model_validate_json(f.read_text(encoding="utf-8"))
            stat = f.stat()
            entries_data.append({
                "path": str(f),
                "artifact_type": entry.artifact_type,
                "timestamp": entry.timestamp,
                "model_fingerprint": entry.model_fingerprint,
                "size_bytes": stat.st_size,
            })
            total_bytes += stat.st_size
            by_type[entry.artifact_type] = by_type.get(entry.artifact_type, 0) + 1
        except Exception:
            pass

    oldest = min((e["timestamp"] for e in entries_data), default=None)

    return {
        "total_entries": len(entries_data),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "by_artifact_type": by_type,
        "oldest_timestamp": oldest,
        "entries": entries_data,
    }


def make_cache_entry(
    func: FunctionInfo,
    artifact_type: str,
    generated_artifact: str,
    source: str,
    project_root: str,
    model_fingerprint: str,
    schema_version: int,
) -> CacheEntry:
    """Create a CacheEntry ready for write_cache().

    Args:
        func: The function the artifact was generated for.
        artifact_type: One of docstring, type, comment, test.
        generated_artifact: JSON-serialized artifact string.
        source: Full source of the file.
        project_root: Project root directory.
        model_fingerprint: "{provider}/{model}" string.
        schema_version: Current config schema version.

    Returns:
        A populated CacheEntry.

    """
    h = compute_composite_hash(func, artifact_type, source, project_root, model_fingerprint)
    metadata: dict = {}
    if artifact_type == "comment":
        metadata["cyclomatic_complexity"] = func.cyclomatic_complexity

    return CacheEntry(
        composite_hash=h,
        artifact_type=artifact_type,
        schema_version=schema_version,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        generated_artifact=generated_artifact,
        model_fingerprint=model_fingerprint,
        metadata=metadata,
    )
