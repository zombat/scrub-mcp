"""Pipeline orchestrator. Deterministic-first, batched DSPy for the rest.

Flow:
1. Ruff lint + autofix (deterministic, no LLM)
2. AST parse: modules, classes, functions (deterministic)
3. Pre-filter: skip anything that already has docstrings/types (deterministic)
4. Batch remaining work into N-function chunks
5. DSPy modules process each batch in a single LLM call (local Qwen Coder)
6. Source rewriter applies all changes

Latency math (batch_size=5, 30 functions needing docstrings + types):
    Without batching: 60 LLM calls
    With batching:    12 LLM calls (6 docstring batches + 6 type batches)
    + comments only fire on complex functions (complexity-gated)
"""

from __future__ import annotations

import logging
from pathlib import Path

import dspy

from dspy_code_hygiene.config import PipelineConfig, load_config
from dspy_code_hygiene.models import FunctionInfo, HygieneReport
from dspy_code_hygiene.modules.hygiene import (
    BatchDocstringGenerator,
    BatchTypeAnnotator,
    ClassDocstringGenerator,
    CommentWriter,
    DocstringGenerator,
    ModuleDocstringGenerator,
    TypeAnnotator,
)
from dspy_code_hygiene.tools.linter import run_ruff
from dspy_code_hygiene.tools.parser import extract_classes, extract_functions, extract_module_info
from dspy_code_hygiene.tools.rewriter import (
    apply_class_docstrings,
    apply_docstrings,
    apply_module_docstring,
    apply_type_annotations,
)
from dspy_code_hygiene.tools.utils import (
    batch,
    needs_class_docstring,
    needs_docstring,
    needs_type_annotations,
)

logger = logging.getLogger(__name__)


def configure_dspy(config: PipelineConfig) -> None:
    """Set up DSPy to use the configured local LLM (default: Qwen Coder via Ollama)."""
    if config.model.provider == "ollama":
        lm = dspy.LM(
            model=f"ollama_chat/{config.model.model}",
            api_base=config.model.base_url,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    elif config.model.provider == "apple":
        # Uses Ray's AppleLocalLM adapter from DSPy PR #9473
        lm = dspy.LM(
            model=f"apple_local/{config.model.model}",
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    else:
        lm = dspy.LM(
            model=config.model.model,
            api_base=config.model.base_url,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    dspy.configure(lm=lm)


def run_pipeline(
    source: str,
    file_path: str = "<stdin>",
    config: PipelineConfig | None = None,
    steps: set[str] | None = None,
) -> HygieneReport:
    """Run the full hygiene pipeline on Python source code.

    Args:
        source: Python source code string.
        file_path: Original file path for reporting.
        config: Pipeline configuration. Loads defaults if None.
        steps: Which steps to run. None = all.
            Valid: {"lint", "docstrings", "types", "comments"}

    Returns:
        HygieneReport with modified source and per-step details.
    """
    cfg = config or load_config()
    all_steps = steps or {"lint", "docstrings", "types", "comments"}
    bs = cfg.batch_size

    report = HygieneReport(file_path=file_path)
    working_source = source

    # ── Step 1: Ruff (deterministic, no LLM) ──
    if "lint" in all_steps:
        logger.info("[lint] Running Ruff on %s", file_path)
        working_source, lint_result = run_ruff(working_source, cfg.ruff)
        report.lint = lint_result
        logger.info(
            "[lint] %d violations found, %d auto-fixed, %d remaining",
            lint_result.violations_before,
            lint_result.auto_fixed,
            lint_result.violations_after,
        )

    # ── Step 2: Parse everything (deterministic) ──
    module_info = extract_module_info(working_source)
    classes = extract_classes(working_source)
    functions = extract_functions(
        working_source,
        skip_private=cfg.skip_private,
        skip_dunder=cfg.skip_dunder,
    )

    # ── Step 3: Pre-filter (deterministic, no LLM) ──
    # Each filter tier only runs when its step is requested.
    # Tier 1 (AST): presence check, free.
    # Tier 2 (pydocstyle/pyright): quality check, subprocess.
    # Both tiers gated: no wasted cycles on steps you didn't ask for.
    use_tools = cfg.deterministic_prefilter

    funcs_needing_docs: list[FunctionInfo] = []
    funcs_needing_types: list[FunctionInfo] = []
    classes_needing_docs = []

    if "docstrings" in all_steps:
        funcs_needing_docs = [
            f for f in functions
            if needs_docstring(f, use_pydocstyle=use_tools)
        ]
        classes_needing_docs = [
            c for c in classes
            if needs_class_docstring(c, use_pydocstyle=use_tools)
        ]
        logger.info(
            "[pre-filter:docstrings] %d/%d functions, %d/%d classes need work (%d skipped)",
            len(funcs_needing_docs), len(functions),
            len(classes_needing_docs), len(classes),
            len(functions) - len(funcs_needing_docs),
        )

    if "types" in all_steps:
        funcs_needing_types = [
            f for f in functions
            if needs_type_annotations(f, use_pyright=use_tools)
        ]
        logger.info(
            "[pre-filter:types] %d/%d functions need annotations (%d skipped)",
            len(funcs_needing_types), len(functions),
            len(functions) - len(funcs_needing_types),
        )

    # ── Step 4: DSPy modules (batched, local Qwen Coder) ──
    # Only spin up the LLM connection if there's actual work to do.
    has_doc_work = funcs_needing_docs or classes_needing_docs or (
        cfg.docstring_all_modules and not module_info.existing_docstring
    )
    has_type_work = bool(funcs_needing_types)
    has_comment_work = "comments" in all_steps and functions

    needs_llm = (
        ("docstrings" in all_steps and has_doc_work)
        or ("types" in all_steps and has_type_work)
        or has_comment_work
    )
    if needs_llm:
        configure_dspy(cfg)
    else:
        logger.info("[pipeline] All checks passed deterministically, no LLM calls needed")

    if "docstrings" in all_steps:
        # Module docstring (1 call max)
        if cfg.docstring_all_modules and not module_info.existing_docstring:
            logger.info("[docstrings] Generating module docstring")
            mod_gen = ModuleDocstringGenerator()
            try:
                mod_doc = mod_gen(module_info)
                report.docstrings.append(mod_doc)
                working_source = apply_module_docstring(working_source, mod_doc.docstring)
            except Exception:
                logger.exception("[docstrings] Module docstring failed")

        # Class docstrings (1 call per class, usually few)
        if cfg.docstring_all_classes and classes_needing_docs:
            logger.info("[docstrings] Generating %d class docstrings", len(classes_needing_docs))
            cls_gen = ClassDocstringGenerator()
            class_docs = []
            for cls in classes_needing_docs:
                try:
                    doc = cls_gen(cls)
                    report.docstrings.append(doc)
                    class_docs.append((cls, doc))
                except Exception:
                    logger.exception("[docstrings] Class docstring failed for %s", cls.name)
            working_source = apply_class_docstrings(working_source, class_docs)

        # Function/method docstrings (BATCHED)
        if funcs_needing_docs:
            batches = batch(funcs_needing_docs, bs)
            logger.info(
                "[docstrings] %d functions in %d batches (batch_size=%d)",
                len(funcs_needing_docs), len(batches), bs,
            )

            if bs > 1:
                batch_gen = BatchDocstringGenerator()
                for i, func_batch in enumerate(batches):
                    try:
                        docs = batch_gen(func_batch)
                        report.docstrings.extend(docs)
                        logger.info("[docstrings] Batch %d/%d: %d docstrings", i + 1, len(batches), len(docs))
                    except Exception:
                        logger.exception("[docstrings] Batch %d failed, falling back to sequential", i + 1)
                        # Fallback: process this batch one-by-one
                        single_gen = DocstringGenerator()
                        for func in func_batch:
                            try:
                                doc = single_gen(func)
                                report.docstrings.append(doc)
                            except Exception:
                                logger.exception("[docstrings] Sequential fallback failed for %s", func.name)
            else:
                # batch_size=1: sequential mode
                single_gen = DocstringGenerator()
                for func in funcs_needing_docs:
                    try:
                        doc = single_gen(func)
                        report.docstrings.append(doc)
                    except Exception:
                        logger.exception("[docstrings] Failed for %s", func.name)

            working_source = apply_docstrings(working_source, report.docstrings)

    if "types" in all_steps and funcs_needing_types:
        batches = batch(funcs_needing_types, bs)
        logger.info(
            "[types] %d functions in %d batches (batch_size=%d)",
            len(funcs_needing_types), len(batches), bs,
        )

        if bs > 1:
            batch_ann = BatchTypeAnnotator()
            for i, func_batch in enumerate(batches):
                try:
                    anns = batch_ann(func_batch)
                    report.type_annotations.extend(anns)
                    logger.info("[types] Batch %d/%d: %d annotations", i + 1, len(batches), len(anns))
                except Exception:
                    logger.exception("[types] Batch %d failed, falling back to sequential", i + 1)
                    single_ann = TypeAnnotator()
                    for func in func_batch:
                        try:
                            ann = single_ann(func)
                            report.type_annotations.append(ann)
                        except Exception:
                            logger.exception("[types] Sequential fallback failed for %s", func.name)
        else:
            single_ann = TypeAnnotator()
            for func in funcs_needing_types:
                try:
                    ann = single_ann(func)
                    report.type_annotations.append(ann)
                except Exception:
                    logger.exception("[types] Failed for %s", func.name)

        working_source = apply_type_annotations(working_source, report.type_annotations)

    if "comments" in all_steps and functions:
        # Comments are NOT batched: they need full function body context
        # and only fire on complex functions anyway
        commenter = CommentWriter(config=cfg.comments)
        eligible = [f for f in functions if commenter.should_comment(f)]
        logger.info(
            "[comments] %d/%d functions exceed complexity threshold",
            len(eligible), len(functions),
        )
        for func in eligible:
            try:
                comments = commenter(func)
                report.comments.extend(comments)
            except Exception:
                logger.exception("[comments] Failed for %s", func.name)

    report.modified_source = working_source
    return report


def run_pipeline_on_file(
    file_path: Path,
    config: PipelineConfig | None = None,
    steps: set[str] | None = None,
    write: bool = False,
) -> HygieneReport:
    """Convenience: run pipeline on a file, optionally writing changes back.

    Args:
        file_path: Path to the Python file.
        config: Pipeline config. Loads defaults if None.
        steps: Which steps to run. None = all.
        write: If True, overwrite the file with modified source.

    Returns:
        HygieneReport with modified source and per-step details.
    """
    source = file_path.read_text(encoding="utf-8")
    report = run_pipeline(source, str(file_path), config, steps)

    if write and report.modified_source:
        file_path.write_text(report.modified_source, encoding="utf-8")
        logger.info("Wrote modified source to %s", file_path)

    return report
