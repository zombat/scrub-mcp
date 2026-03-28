"""Configuration management for dspy-code-hygiene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Local LLM endpoint configuration."""

    provider: str = "ollama"
    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434"
    max_tokens: int = 4096
    temperature: float = 0.1


class RuffConfig(BaseModel):
    """Ruff linter configuration."""

    fix: bool = True
    select: list[str] = Field(default_factory=lambda: ["E", "F", "W", "I", "UP", "ANN", "D"])
    ignore: list[str] = Field(default_factory=lambda: ["ANN101", "ANN102", "D100"])
    line_length: int = 100


class CommentConfig(BaseModel):
    """Controls when the comment module fires."""

    min_lines: int = 8
    min_cyclomatic_complexity: int = 3
    skip_trivial: bool = True


class OptimizerConfig(BaseModel):
    """Per-module optimizer strategy with teacher-student support.

    Teacher-student pattern:
        The teacher model (large cloud model) generates the optimized
        prompts and few-shot examples during compilation. The student
        model (local Qwen Coder) runs them at inference time.

        This separates compilation cost (one-time, cloud) from runtime
        cost (ongoing, local). MIPROv2 on Claude as teacher produces
        better prompts than MIPROv2 on Qwen, and those prompts transfer
        to the student.

    Strategy choices:
        bootstrap: BootstrapFewShot. Fast, selects best examples.
        bootstrap_rs: BootstrapFewShotWithRandomSearch. Tries combos.
        mipro: MIPROv2. Rewrites instructions + selects examples.
    """

    # Teacher model: used during optimization only (one-time cost)
    teacher: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com",
            max_tokens=4096,
            temperature=0.0,
        ),
        description="Large model for compilation. Cloud cost is one-time.",
    )
    use_teacher: bool = Field(
        default=False,
        description="If True, use teacher model for optimization. If False, optimize on the student (runtime) model.",
    )

    # Per-module strategy
    docstrings: str = "bootstrap"
    types: str = "bootstrap"
    comments: str = "bootstrap"
    complexity: str = "mipro"
    tests: str = "mipro"
    refactoring: str = "mipro"
    imports: str = "bootstrap_rs"
    dead_code: str = "bootstrap_rs"

    # Hyperparameters
    mipro_auto: str = Field(
        default="light",
        description="MIPROv2 optimization budget: light, medium, heavy",
    )
    bootstrap_max_demos: int = Field(
        default=4,
        description="Max few-shot examples for Bootstrap strategies",
    )
    bootstrap_rs_trials: int = Field(
        default=10,
        description="Random search trials for BootstrapFewShotWithRandomSearch",
    )
    num_threads: int = Field(
        default=1,
        description="Parallel threads. Bump for cloud teacher or multi-GPU.",
    )


class PipelineConfig(BaseModel):
    """Full pipeline configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    ruff: RuffConfig = Field(default_factory=RuffConfig)
    comments: CommentConfig = Field(default_factory=CommentConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    docstring_style: str = "google"
    skip_private: bool = False
    skip_dunder: bool = False
    docstring_all_classes: bool = True
    docstring_all_modules: bool = True
    type_all_args: bool = True
    type_all_returns: bool = True
    batch_size: int = Field(
        default=5,
        description="Functions per batch for DSPy calls. Higher = fewer round trips, more tokens per call.",
    )
    deterministic_prefilter: bool = Field(
        default=True,
        description="Run pyright/pydocstyle checks first to skip functions that already pass.",
    )
    optimizer_cache_dir: str = ".dspy_cache"


def load_config(config_path: Path | None = None) -> PipelineConfig:
    """Load config from YAML file, falling back to defaults."""
    if config_path is None:
        config_path = Path("config.yaml")
    if config_path and config_path.exists():
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
        return PipelineConfig(**raw)
    return PipelineConfig()
