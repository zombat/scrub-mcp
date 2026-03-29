"""S.C.R.U.B. CLI — cache management and project warming.

Commands:
    scrub cache stats   — hit rate, size, entries by type, stale fingerprints
    scrub cache clear   — remove all, by type, or stale entries
    scrub cache warm    — pre-populate cache by running the full pipeline cold
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer

app = typer.Typer(name="scrub", help="S.C.R.U.B. code hygiene CLI", no_args_is_help=True)
cache_app = typer.Typer(help="Cache management commands", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("stats")
def cache_stats(
    cache_dir: str = typer.Option(
        ".scrub_cache/artifacts",
        "--cache-dir",
        help="Cache directory to inspect",
    ),
) -> None:
    """Show cache statistics: size, entry counts, oldest entry, stale fingerprints."""
    from scrub_mcp.config import load_config
    from scrub_mcp.tools.cache import cache_stats as _stats

    cfg = load_config()
    effective_dir = cache_dir if cache_dir != ".scrub_cache/artifacts" else cfg.cache.cache_dir
    stats = _stats(effective_dir)
    current_fp = f"{cfg.model.provider}/{cfg.model.model}"

    stale = sum(
        1 for e in stats["entries"]
        if e.get("model_fingerprint") != current_fp
    )

    typer.echo(f"Cache directory : {effective_dir}")
    typer.echo(f"Total entries   : {stats['total_entries']}")
    typer.echo(f"Total size      : {stats['total_size_mb']} MB")
    typer.echo(f"Current model   : {current_fp}")
    typer.echo(f"Stale entries   : {stale} (mismatched model fingerprint)")
    typer.echo(f"Oldest entry    : {stats['oldest_timestamp'] or 'N/A'}")
    typer.echo("")
    typer.echo("Entries by artifact type:")
    for atype, count in sorted(stats["by_artifact_type"].items()):
        typer.echo(f"  {atype:12s} : {count}")


@cache_app.command("clear")
def cache_clear(
    cache_dir: str = typer.Option(
        ".scrub_cache/artifacts",
        "--cache-dir",
        help="Cache directory to clear",
    ),
    artifact_type: str | None = typer.Option(
        None,
        "--type",
        help="Only remove entries of this artifact type (docstring, type, comment, test)",
    ),
    stale: bool = typer.Option(
        False,
        "--stale",
        help="Only remove entries with a mismatched model fingerprint",
    ),
) -> None:
    """Remove cache entries. With no flags, clears everything (prompts for confirmation)."""
    from scrub_mcp.config import load_config
    from scrub_mcp.tools.cache import CacheEntry, cache_stats as _stats

    cfg = load_config()
    effective_dir = cache_dir if cache_dir != ".scrub_cache/artifacts" else cfg.cache.cache_dir
    cache_path = Path(effective_dir)

    if not cache_path.exists():
        typer.echo("Cache directory does not exist. Nothing to clear.")
        return

    if not artifact_type and not stale:
        # Full clear with confirmation
        typer.confirm(
            f"Delete entire cache at {effective_dir}?",
            abort=True,
        )
        shutil.rmtree(effective_dir, ignore_errors=True)
        typer.echo("Cache cleared.")
        return

    current_fp = f"{cfg.model.provider}/{cfg.model.model}"
    removed = 0

    for f in cache_path.rglob("*.json"):
        try:
            entry = CacheEntry.model_validate_json(f.read_text(encoding="utf-8"))
            remove = False
            if artifact_type and entry.artifact_type == artifact_type:
                remove = True
            if stale and entry.model_fingerprint != current_fp:
                remove = True
            if remove:
                f.unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass

    typer.echo(f"Removed {removed} entries.")


@cache_app.command("warm")
def cache_warm(
    path: str = typer.Argument(..., help="Project path or directory to warm"),
    steps: list[str] = typer.Option(
        ["docstrings", "types"],
        "--step",
        help="Steps to run (repeat flag for multiple: --step docstrings --step types)",
    ),
    workers: int = typer.Option(4, "--workers", help="Parallel workers"),
) -> None:
    """Pre-populate the cache by running the pipeline on the entire project."""
    from scrub_mcp.config import load_config
    from scrub_mcp.pipeline import run_pipeline_batch_parallel
    from scrub_mcp.tools.fs import get_tracked_files

    cfg = load_config()
    project_path = Path(path).resolve()

    if not project_path.exists():
        typer.echo(f"Path not found: {project_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Warming cache for {project_path} ...")
    typer.echo(f"Steps   : {', '.join(steps)}")
    typer.echo(f"Workers : {workers}")

    py_files = get_tracked_files(str(project_path), cfg.exclude_paths)
    file_paths = [str(f) for f in py_files if f.suffix == ".py"]

    if not file_paths:
        typer.echo("No .py files found.")
        return

    typer.echo(f"Files   : {len(file_paths)}")

    results = run_pipeline_batch_parallel(
        file_paths, cfg, set(steps), write=False, max_workers=workers
    )

    success = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    errors = sum(1 for r in results if isinstance(r, Exception))

    typer.echo(f"\nDone. {success} files processed, {errors} errors.")


def main() -> None:
    """Entry point for the scrub CLI."""
    app()


if __name__ == "__main__":
    main()
