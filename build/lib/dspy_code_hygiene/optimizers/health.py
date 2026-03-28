"""Cache health checker. Detects prompt staleness and model drift.

Run periodically (CI, pre-commit, or manual) to verify that cached
optimized prompts still perform well on the current student model.

Staleness sources:
    - Student model upgraded (Qwen 2.5 -> 3.0)
    - Training examples updated (new ground truth)
    - Ollama/inference server changed chat template
    - Signature definitions modified

Usage:
    python -m dspy_code_hygiene.optimizers.health
    python -m dspy_code_hygiene.optimizers.health --threshold 0.7
    python -m dspy_code_hygiene.optimizers.health --module docstrings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import dspy

from dspy_code_hygiene.config import PipelineConfig, load_config
from dspy_code_hygiene.optimizers.tune import (
    MODULE_REGISTRY,
    _build_lm,
    load_examples,
)

logger = logging.getLogger(__name__)

# Score thresholds
DEFAULT_THRESHOLD = 0.6
WARN_THRESHOLD = 0.7


def check_health(
    config: PipelineConfig,
    cache_dir: Path,
    examples_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    modules_filter: set[str] | None = None,
    sample_size: int = 10,
) -> dict[str, dict[str, Any]]:
    """Validate cached prompts against the current student model.

    For each cached module:
        1. Load the optimized module from cache
        2. Run it against training examples on the current student
        3. Score with structural metrics (no judge, this is a quick check)
        4. Compare against threshold and previous scores
        5. Flag stale modules

    Args:
        config: Pipeline config with current model settings.
        cache_dir: Path to .dspy_cache/ directory.
        examples_dir: Path to training examples.
        threshold: Minimum acceptable score. Below = stale.
        modules_filter: Only check these modules.
        sample_size: Max examples to evaluate per module (speed vs accuracy).

    Returns:
        Dict of module_name -> health report.
    """
    student_lm = _build_lm(config.model, label="student")
    dspy.configure(lm=student_lm)

    all_examples = load_examples(examples_dir)
    results_file = cache_dir / "optimization_results.json"
    previous_results = {}
    if results_file.exists():
        previous_results = json.loads(results_file.read_text())

    # Compute a fingerprint of the current model + config
    model_fingerprint = _model_fingerprint(config)

    health: dict[str, dict[str, Any]] = {}

    for module_name, reg in MODULE_REGISTRY.items():
        base_name = reg.get("strategy_key", module_name)
        if modules_filter and base_name not in modules_filter and module_name not in modules_filter:
            continue

        cache_path = cache_dir / f"{module_name}.json"
        if not cache_path.exists():
            health[module_name] = {
                "status": "missing",
                "message": "No cached module found. Run optimizer first.",
                "score": 0.0,
            }
            continue

        # Load examples
        example_key = base_name.split("_")[0] if "_" in base_name else base_name
        examples = all_examples.get(example_key, [])
        if not examples:
            health[module_name] = {
                "status": "no_examples",
                "message": "No training examples to validate against.",
                "score": 0.0,
            }
            continue

        # Build trainset
        input_fields = reg["input_fields"]
        output_field = reg["output_field"]
        metric = reg["metric"]

        trainset = []
        for ex in examples:
            if output_field not in ex:
                continue
            example_kwargs = {f: ex.get(f, "") for f in input_fields}
            example_kwargs[output_field] = ex[output_field]
            trainset.append(
                dspy.Example(**example_kwargs).with_inputs(*input_fields)
            )

        if not trainset:
            health[module_name] = {
                "status": "no_examples",
                "message": "No valid training examples with ground truth.",
                "score": 0.0,
            }
            continue

        # Load and evaluate cached module
        try:
            module_instance = reg["module_cls"]()
            module_instance.load(str(cache_path))

            start = time.time()
            scores = []
            errors = 0
            for ex in trainset[:sample_size]:
                try:
                    pred = module_instance(**{f: getattr(ex, f) for f in input_fields})
                    scores.append(metric(ex, pred))
                except Exception:
                    errors += 1
                    scores.append(0.0)

            elapsed = time.time() - start
            avg_score = sum(scores) / max(len(scores), 1)

            # Compare to previous
            prev = previous_results.get(module_name, {})
            prev_score = prev.get("score_on_student", prev.get("score", 0.0))
            prev_model = prev.get("student", "unknown")
            score_delta = avg_score - prev_score if prev_score else 0.0

            # Check model fingerprint against cached
            cached_fingerprint = _read_cached_fingerprint(cache_dir, module_name)
            model_changed = cached_fingerprint and cached_fingerprint != model_fingerprint

            # Determine status
            if avg_score < threshold:
                status = "stale"
                message = f"Score {avg_score:.3f} below threshold {threshold}. Recompile recommended."
            elif model_changed:
                status = "model_changed"
                message = (
                    f"Score {avg_score:.3f} is acceptable but model changed since compilation. "
                    f"Recompile may improve quality."
                )
            elif score_delta < -0.1:
                status = "degraded"
                message = (
                    f"Score dropped {score_delta:+.3f} from {prev_score:.3f} to {avg_score:.3f}. "
                    f"Investigate or recompile."
                )
            elif avg_score < WARN_THRESHOLD:
                status = "warn"
                message = f"Score {avg_score:.3f} is marginal. Consider recompiling."
            else:
                status = "healthy"
                message = f"Score {avg_score:.3f} is good."

            health[module_name] = {
                "status": status,
                "message": message,
                "score": round(avg_score, 3),
                "prev_score": round(prev_score, 3) if prev_score else None,
                "score_delta": round(score_delta, 3) if prev_score else None,
                "errors": errors,
                "samples": len(scores),
                "eval_time_s": round(elapsed, 1),
                "model_changed": model_changed,
                "current_model": config.model.model,
                "compiled_model": prev_model,
            }

        except Exception as e:
            health[module_name] = {
                "status": "error",
                "message": f"Failed to load or evaluate: {e}",
                "score": 0.0,
            }

    return health


def print_health_report(health: dict[str, dict[str, Any]]) -> None:
    """Pretty-print the health check results."""
    status_icons = {
        "healthy": "OK",
        "warn": "!?",
        "stale": "XX",
        "degraded": "DN",
        "model_changed": "MC",
        "missing": "--",
        "no_examples": "NE",
        "error": "ER",
    }

    logger.info("=" * 72)
    logger.info("Cache Health Report")
    logger.info("=" * 72)

    stale_count = 0
    for name, info in sorted(health.items()):
        icon = status_icons.get(info["status"], "??")
        score = info.get("score", 0.0)
        delta = info.get("score_delta")
        delta_str = f" ({delta:+.3f})" if delta is not None else ""

        logger.info(
            "  [%s] %-25s  score=%.3f%s  %s",
            icon, name, score, delta_str, info["message"],
        )

        if info["status"] in ("stale", "degraded", "model_changed"):
            stale_count += 1

    logger.info("-" * 72)
    if stale_count:
        logger.info(
            "  %d module(s) need recompilation. Run:",
            stale_count,
        )
        stale_names = [
            n for n, i in health.items()
            if i["status"] in ("stale", "degraded", "model_changed")
        ]
        logger.info(
            "    python -m dspy_code_hygiene.optimizers.tune --modules %s",
            ",".join(stale_names),
        )
    else:
        logger.info("  All modules healthy.")
    logger.info("=" * 72)


def save_fingerprint(cache_dir: Path, module_name: str, config: PipelineConfig) -> None:
    """Save the model fingerprint alongside a compiled module."""
    fp = _model_fingerprint(config)
    fp_path = cache_dir / f"{module_name}.fingerprint"
    fp_path.write_text(fp)


def _model_fingerprint(config: PipelineConfig) -> str:
    """Generate a fingerprint from model config for drift detection."""
    key = f"{config.model.provider}:{config.model.model}:{config.model.base_url}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _read_cached_fingerprint(cache_dir: Path, module_name: str) -> str | None:
    """Read a previously saved model fingerprint."""
    fp_path = cache_dir / f"{module_name}.fingerprint"
    if fp_path.exists():
        return fp_path.read_text().strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check health of cached DSPy optimized prompts",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".dspy_cache"),
        help="Path to the cache directory",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("examples"),
        help="Path to training examples",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum acceptable score (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Comma-separated module names to check (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Max examples to evaluate per module (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable table",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_config(args.config)
    modules_filter = set(args.modules.split(",")) if args.modules else None

    health = check_health(
        config,
        cache_dir=args.cache_dir,
        examples_dir=args.examples_dir,
        threshold=args.threshold,
        modules_filter=modules_filter,
        sample_size=args.samples,
    )

    if args.json:
        print(json.dumps(health, indent=2))
    else:
        print_health_report(health)

    # Save report
    report_path = args.cache_dir / "health_report.json"
    report_path.write_text(json.dumps(health, indent=2))

    # Exit code: non-zero if any module is stale
    stale = any(
        i["status"] in ("stale", "error")
        for i in health.values()
    )
    raise SystemExit(1 if stale else 0)


if __name__ == "__main__":
    main()
