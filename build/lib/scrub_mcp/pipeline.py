"""Pipeline orchestrator. Deterministic-first, batched DSPy for the rest.

Flow:
1. Ruff lint + autofix (deterministic, no LLM)
2. AST parse: modules, classes, functions (deterministic, single parse shared)
3. Pre-filter: skip anything that already has docstrings/types (deterministic)
   - File-level pyright/pydocstyle: 1 subprocess per tool per file (not per function)
4. Batch remaining work into N-function chunks (adaptive sizing)
5. DSPy modules process each batch in a single LLM call (local Qwen Coder)
   - Results applied per-batch; failed batches retry per-function
6. Source rewriter applies all changes (bottom-up, line-number stable)

Latency math (batch_size=5, 30 functions needing docstrings + types):
    Without batching: 60 LLM calls
    With batching:    12 LLM calls (6 docstring batches + 6 type batches)
    + comments only fire on complex functions (complexity-gated)
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dspy

from scrub_mcp.config import PipelineConfig, load_config
from scrub_mcp.models import (
    FunctionInfo,
    GeneratedDocstring,
    HygieneReport,
    SemanticComment,
    TypeAnnotation,
)
from scrub_mcp.modules.hygiene import (
    BatchDocstringGenerator,
    BatchTypeAnnotator,
    ClassDocstringGenerator,
    CommentWriter,
    DocstringGenerator,
    ModuleDocstringGenerator,
    TypeAnnotator,
)
from scrub_mcp.tools.linter import run_ruff
from scrub_mcp.tools.parser import extract_classes, extract_functions, extract_module_info, parse_source
from scrub_mcp.tools.rewriter import (
    apply_class_docstrings,
    apply_docstrings,
    apply_module_docstring,
    apply_type_annotations,
)
from scrub_mcp.utils import (
    _pydocstyle_file_check,
    _pyright_file_check,
    batch,
    needs_class_docstring,
    needs_docstring,
    needs_type_annotations,
)

logger = logging.getLogger(__name__)

_TEST_PATTERN = re.compile(r"(^|[/\\])(test_.*|.*_test|conftest)\.py$")


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

    # Keep-alive ping: prevents Ollama's 5-min idle timeout from expiring during
    # a long deterministic pre-filter phase before the first real LLM call.
    if config.model.provider == "ollama":
        try:
            lm("ping", max_tokens=1)
        except Exception:
            pass  # non-fatal; model will cold-load on first real call if needed


def _compute_adaptive_batch_size(funcs: list[FunctionInfo], config: PipelineConfig) -> int:
    """Return an adaptive batch size based on average function body length.

    Short functions get larger batches (more per call); long functions get
    smaller batches to stay within the model's context window.
    """
    if not funcs or not config.adaptive_batch:
        return config.batch_size
    avg_lines = sum(f.body_line_count for f in funcs) / len(funcs)
    # ~5 tokens per line heuristic
    funcs_per_target = max(1, int(config.target_batch_tokens / max(1, avg_lines * 5)))
    return min(funcs_per_target, config.max_batch_size)


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

    # ── Step 2: Parse everything (deterministic, single AST parse shared) ──
    tree = parse_source(working_source)
    module_info = extract_module_info(working_source, tree=tree)
    classes = extract_classes(working_source, tree=tree)
    functions = extract_functions(
        working_source,
        skip_private=cfg.skip_private,
        skip_dunder=cfg.skip_dunder,
        tree=tree,
    )

    # ── Step 3: Pre-filter (deterministic, no LLM) ──
    # File-level checks: run pyright/pydocstyle once on the whole file (O(1) subprocesses),
    # then map diagnostics to function line ranges.
    use_tools = cfg.deterministic_prefilter
    real_path = file_path if file_path != "<stdin>" else None

    pydocstyle_failing: set[str] = set()
    pyright_failing: set[str] = set()

    if use_tools and "docstrings" in all_steps:
        pydocstyle_failing = _pydocstyle_file_check(working_source, functions, real_path)

    if use_tools and "types" in all_steps:
        pyright_failing = _pyright_file_check(working_source, functions, real_path)

    funcs_needing_docs: list[FunctionInfo] = []
    funcs_needing_types: list[FunctionInfo] = []
    classes_needing_docs = []

    if "docstrings" in all_steps:
        funcs_needing_docs = [
            f for f in functions
            if needs_docstring(f, use_pydocstyle=False) or f.name in pydocstyle_failing
        ]
        classes_needing_docs = [
            c for c in classes if needs_class_docstring(c, use_pydocstyle=use_tools)
        ]
        logger.info(
            "[pre-filter:docstrings] %d/%d functions, %d/%d classes need work (%d skipped)",
            len(funcs_needing_docs),
            len(functions),
            len(classes_needing_docs),
            len(classes),
            len(functions) - len(funcs_needing_docs),
        )

    if "types" in all_steps:
        # Skip test files for type annotation (they're conventionally untyped)
        if cfg.skip_test_types and real_path and _TEST_PATTERN.search(real_path):
            funcs_needing_types = []
            logger.info("[pre-filter:types] Skipping test file %s", file_path)
        else:
            funcs_needing_types = [
                f for f in functions
                if needs_type_annotations(f, use_pyright=False) or f.name in pyright_failing
            ]
        logger.info(
            "[pre-filter:types] %d/%d functions need annotations (%d skipped)",
            len(funcs_needing_types),
            len(functions),
            len(functions) - len(funcs_needing_types),
        )

    # ── Step 4: DSPy modules (batched, local Qwen Coder) ──
    # Only spin up the LLM connection if there's actual work to do.
    has_doc_work = (
        funcs_needing_docs
        or classes_needing_docs
        or (cfg.docstring_all_modules and not module_info.existing_docstring)
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

        # Function/method docstrings (BATCHED, adaptive size, streaming per-batch apply)
        if funcs_needing_docs:
            bs = _compute_adaptive_batch_size(funcs_needing_docs, cfg)
            batches = batch(funcs_needing_docs, bs)
            logger.info(
                "[docstrings] %d functions in %d batches (batch_size=%d)",
                len(funcs_needing_docs),
                len(batches),
                bs,
            )

            if bs > 1:
                batch_gen = BatchDocstringGenerator()
                for i, func_batch in enumerate(batches):
                    try:
                        payload = json.dumps(
                            [
                                {
                                    "name": f.name,
                                    "signature": f.signature,
                                    # Signature-only mode: skip body for short functions
                                    "body": (
                                        ""
                                        if f.body_line_count < cfg.signature_only_threshold
                                        else f.body[:2000]
                                    ),
                                    "parent_class": f.parent_class or "",
                                }
                                for f in func_batch
                            ]
                        )
                        raw = batch_gen(functions_json=payload)
                        parsed = json.loads(raw) if raw else {}
                        docs = [
                            GeneratedDocstring(
                                function_name=f.name,
                                docstring=parsed[f.name].strip().strip('"""').strip("'''").strip(),
                                style="google",
                                target_type="function",
                            )
                            for f in func_batch
                            if parsed.get(f.name)
                        ]
                        report.docstrings.extend(docs)
                        # Apply immediately (streaming) — bottom-up order preserved by rewriter
                        working_source = apply_docstrings(working_source, docs)
                        logger.info(
                            "[docstrings] Batch %d/%d: %d docstrings applied",
                            i + 1,
                            len(batches),
                            len(docs),
                        )
                    except Exception:
                        logger.exception(
                            "[docstrings] Batch %d failed, falling back to sequential", i + 1
                        )
                        single_gen = DocstringGenerator()
                        for func in func_batch:
                            try:
                                text = single_gen(
                                    function_signature=func.signature,
                                    function_body=(
                                        ""
                                        if func.body_line_count < cfg.signature_only_threshold
                                        else func.body
                                    ),
                                    decorators=", ".join(func.decorators),
                                    parent_class=func.parent_class or "",
                                )
                                doc = GeneratedDocstring(
                                    function_name=func.name,
                                    docstring=text,
                                    style="google",
                                    target_type="function",
                                )
                                report.docstrings.append(doc)
                                working_source = apply_docstrings(working_source, [doc])
                            except Exception:
                                logger.warning(
                                    "[docstrings] Skipping %s after both batch and solo failed",
                                    func.name,
                                )
            else:
                # batch_size=1: sequential mode
                single_gen = DocstringGenerator()
                for func in funcs_needing_docs:
                    try:
                        text = single_gen(
                            function_signature=func.signature,
                            function_body=(
                                ""
                                if func.body_line_count < cfg.signature_only_threshold
                                else func.body
                            ),
                            decorators=", ".join(func.decorators),
                            parent_class=func.parent_class or "",
                        )
                        doc = GeneratedDocstring(
                            function_name=func.name,
                            docstring=text,
                            style="google",
                            target_type="function",
                        )
                        report.docstrings.append(doc)
                        working_source = apply_docstrings(working_source, [doc])
                    except Exception:
                        logger.exception("[docstrings] Failed for %s", func.name)

    if "types" in all_steps and funcs_needing_types:
        bs = _compute_adaptive_batch_size(funcs_needing_types, cfg)
        batches = batch(funcs_needing_types, bs)
        logger.info(
            "[types] %d functions in %d batches (batch_size=%d)",
            len(funcs_needing_types),
            len(batches),
            bs,
        )

        if bs > 1:
            batch_ann = BatchTypeAnnotator()
            for i, func_batch in enumerate(batches):
                try:
                    payload = json.dumps(
                        [
                            {"name": f.name, "signature": f.signature, "body": f.body[:2000]}
                            for f in func_batch
                        ]
                    )
                    raw = batch_ann(functions_json=payload)
                    parsed_all = json.loads(raw) if raw else {}
                    anns = []
                    for f in func_batch:
                        fa = dict(parsed_all.get(f.name, {}))
                        if not fa:
                            continue
                        fa.pop("self", None)
                        fa.pop("cls", None)
                        return_type = fa.pop("return", "None")
                        anns.append(
                            TypeAnnotation(
                                function_name=f.name,
                                parameters=fa,
                                return_type=return_type,
                                confidence=0.8,
                            )
                        )
                    report.type_annotations.extend(anns)
                    # Apply immediately (streaming)
                    working_source = apply_type_annotations(working_source, anns)
                    logger.info(
                        "[types] Batch %d/%d: %d annotations applied", i + 1, len(batches), len(anns)
                    )
                except Exception:
                    logger.exception("[types] Batch %d failed, falling back to sequential", i + 1)
                    single_ann = TypeAnnotator()
                    for func in func_batch:
                        try:
                            raw = single_ann(
                                function_signature=func.signature,
                                function_body=func.body,
                                existing_annotations=json.dumps(func.existing_annotations),
                            )
                            fa = json.loads(raw) if raw else {}
                            fa.pop("self", None)
                            fa.pop("cls", None)
                            return_type = fa.pop("return", "None")
                            ann = TypeAnnotation(
                                function_name=func.name,
                                parameters=fa,
                                return_type=return_type,
                                confidence=0.8,
                            )
                            report.type_annotations.append(ann)
                            working_source = apply_type_annotations(working_source, [ann])
                        except Exception:
                            logger.warning(
                                "[types] Skipping %s after both batch and solo failed", func.name
                            )
        else:
            single_ann = TypeAnnotator()
            for func in funcs_needing_types:
                try:
                    raw = single_ann(
                        function_signature=func.signature,
                        function_body=func.body,
                        existing_annotations=json.dumps(func.existing_annotations),
                    )
                    fa = json.loads(raw) if raw else {}
                    fa.pop("self", None)
                    fa.pop("cls", None)
                    return_type = fa.pop("return", "None")
                    ann = TypeAnnotation(
                        function_name=func.name,
                        parameters=fa,
                        return_type=return_type,
                        confidence=0.8,
                    )
                    report.type_annotations.append(ann)
                    working_source = apply_type_annotations(working_source, [ann])
                except Exception:
                    logger.exception("[types] Failed for %s", func.name)

    if "comments" in all_steps and functions:
        # Comments are NOT batched: they need full function body context
        # and only fire on complex functions anyway
        commenter = CommentWriter(config=cfg.comments)
        eligible = [
            f
            for f in functions
            if commenter.should_comment(f.body_line_count, f.cyclomatic_complexity)
        ]
        logger.info(
            "[comments] %d/%d functions exceed complexity threshold",
            len(eligible),
            len(functions),
        )
        for func in eligible:
            try:
                raw = commenter(
                    code_block=func.body,
                    context=func.signature,
                    complexity=func.cyclomatic_complexity,
                )
                items = json.loads(raw) if raw else []
                report.comments.extend(
                    [
                        SemanticComment(
                            line_number=func.line_start + item.get("line_offset", 0),
                            comment=item.get("comment", ""),
                            category=item.get("category", "explanation"),
                        )
                        for item in items
                    ]
                )
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


def run_pipeline_batch_parallel(
    paths: list[str],
    config: PipelineConfig | None = None,
    steps: set[str] | None = None,
    write: bool = False,
    max_workers: int | None = None,
) -> list[HygieneReport | Exception | None]:
    """Run the pipeline on multiple files concurrently.

    Args:
        paths: File paths to process. Non-.py or missing paths yield None.
        config: Pipeline config. Loads defaults if None.
        steps: Which steps to run. None = all.
        write: If True, overwrite each file with its modified source.
        max_workers: Thread pool size. Defaults to config.batch_max_workers or 4.

    Returns:
        List parallel to paths. Each element is a HygieneReport, an Exception
        (if that file failed), or None (if the path was skipped).

    """
    cfg = config or load_config()
    workers = max_workers if max_workers is not None else cfg.batch_max_workers

    results: list[HygieneReport | Exception | None] = [None] * len(paths)

    valid: dict[int, Path] = {
        i: Path(p)
        for i, p in enumerate(paths)
        if p.endswith(".py") and Path(p).exists()
    }

    if not valid:
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_pipeline_on_file, fp, cfg, steps, write): i
            for i, fp in valid.items()
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = exc
                logger.exception("[batch_parallel] Failed on %s", paths[idx])

    return results
