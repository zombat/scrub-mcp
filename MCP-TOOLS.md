# S.C.R.U.B. MCP Tool Reference

Complete reference for all 22 tools exposed by the S.C.R.U.B. MCP server.

---

## Exploration

| Tool | Category | Description |
|------|----------|-------------|
| `explore_architecture` | Exploration | Returns AST skeletons (signatures + one-line docstrings, bodies replaced with `...`) for a file or directory. Call this first before reading any file. |
| `read_files` | Exploration | Batch-read multiple files in one call. Use after `explore_architecture` to pull targeted line ranges. |
| `find_symbols` | Exploration | Extract function and class signatures from Python files via AST — name, signature line, line range, decorators, parent class, docstring/annotation presence. No full read needed. |
| `grep_multi` | Exploration | Run multiple regex patterns across the codebase in a single call with context lines and per-pattern match caps. |

---

## Code Hygiene

| Tool | Category | Description |
|------|----------|-------------|
| `hygiene_full` | Hygiene | Full pipeline: lint → docstrings → types → comments in one pass. **First action on any Python file you are about to touch.** |
| `hygiene_batch` | Hygiene | `hygiene_full` across a list of `.py` files in one call. Prefer over repeated `hygiene_full` calls. Supports `write=true` to apply to disk directly. |
| `hygiene_incremental` | Hygiene | Diff-aware, cache-accelerated hygiene. Skips unchanged functions. Pass `diff` (unified diff text) or `since` (git ref). Use on repeated runs of the same file. |
| `lint_file` | Hygiene | Lint-only pass via Ruff. Returns fixed source + violation counts + remaining issues. |
| `generate_docstrings` | Hygiene | Generate Google-style docstrings via local LLM. Do not write docstrings manually. |
| `annotate_types` | Hygiene | Infer and apply type annotations via local LLM. Skips already-annotated parameters. |
| `add_comments` | Hygiene | Add inline `#` comments to non-obvious logic. Only fires on functions above the complexity threshold. |

---

## Coding Tools

| Tool | Category | Description |
|------|----------|-------------|
| `analyze_complexity` | Coding | Compute exact cyclomatic and cognitive complexity with per-function hotspot ranking. Deterministic, no LLM. Call before any refactor. |
| `suggest_simplifications` | Coding | Provide concrete simplification strategies (early returns, guard clauses, extract function, lookup tables) for functions with CC ≥ threshold. |
| `suggest_refactoring` | Coding | Extract-function candidate analysis with boundary detection and rename suggestions with rationale. Call before extracting or renaming. |
| `optimize_imports` | Coding | Remove unused imports (Ruff), sort (isort), and infer genuinely missing imports from usage patterns (LLM). Call after any import change. |
| `generate_tests` | Coding | Generate parametrized pytest tests with happy path, edge cases, and error paths. Do not write tests manually. |
| `run_tests` | Coding | Run pytest with `PYTHONPATH=src` injected. Returns `exit_code`, `passed`, and full output. Use instead of `python -c` or raw bash. |
| `find_dead_code` | Coding | AST-based detection of unreachable blocks, unused variables, redundant else branches, and commented-out code. Call before deleting anything. |

---

## Security

| Tool | Category | Description |
|------|----------|-------------|
| `security_scan` | Security | Bandit scan with CWE mapping. Detects hardcoded secrets, injection, insecure deserialization, weak crypto, path traversal. Returns severity + confidence + CWE IDs + line numbers. Call before every commit. |
| `security_remediate` | Security | Fix findings from `security_scan`: corrected rewrite, `nosec` annotation with justification, or risk acceptance record. Do not patch security issues manually. |

---

## Supply Chain

| Tool | Category | Description |
|------|----------|-------------|
| `generate_sbom` | Supply Chain | Generate a CycloneDX 1.5 or SPDX 2.3 Software Bill of Materials from pip freeze, pyproject.toml, requirements.txt, and lock files. Call after dependency changes. |
| `scan_vulnerabilities` | Supply Chain | Cross-reference all dependencies against OSV.dev (PyPI advisories, GitHub Security Advisories, NVD). Returns CVE/GHSA IDs, CVSS scores, affected versions, and upgrade paths. Generates SBOM automatically if needed. |

---

## Pre-Processing Gate

The following tools **must** be called before the Cloud Orchestrator takes action on a file.
No edits, writes, or plans may proceed without passing through this gate first.

| Trigger | Required tool |
|---------|--------------|
| Any file or directory not yet read | `explore_architecture` |
| Before touching a single Python file | `hygiene_full` |
| Before touching multiple Python files | `hygiene_batch` |
| Repeated runs on a file already processed | `hygiene_incremental` |
| Before any commit | `security_scan` → `security_remediate` (if findings) → `optimize_imports` |

---

## CLI Commands

S.C.R.U.B. also ships a `scrub` CLI for use in CI pipelines, pre-commit hooks, and local workflows. These commands do **not** require an MCP client.

| Command | LLM? | What it does |
|---------|------|-------------|
| `scrub check` | No | Detect violations (missing docstrings/types, complexity, security, vulnerabilities). Exit 1 if `--fail-on` criteria met. Supports `--format text\|json\|sarif`. |
| `scrub fix` | Yes | Run the full hygiene pipeline and write fixes to disk. Optional `--commit` and `--branch`. |
| `scrub diff` | Yes | Show what `scrub fix` would change without touching files. Output as unified diff or JSON. |
| `scrub audit` | No | Combined Bandit security scan + OSV.dev vulnerability scan + optional SBOM export. |
| `scrub cache stats` | No | Show cache hit rate, size, entries by type, stale fingerprints. |
| `scrub cache clear` | No | Remove cache entries (all, by `--type`, or `--stale`). |
| `scrub cache warm` | Yes | Pre-populate cache by running the pipeline on all project files. |

All commands respect `config.yaml` and accept `--config`, `--quiet`, and `--since` (diff-aware narrowing).

### SARIF Rule IDs

| Rule ID | Source |
|---------|--------|
| `SCRUB-DOC-001` | Missing function docstring |
| `SCRUB-DOC-002` | Missing class docstring |
| `SCRUB-DOC-003` | Missing module docstring |
| `SCRUB-TYPE-001` | Missing type annotation |
| `SCRUB-CMPLX-001` | High cyclomatic complexity |
| `SCRUB-CMPLX-002` | High cognitive complexity |
| `SCRUB-SEC-{ID}` | Bandit finding (e.g. `SCRUB-SEC-B101`) |
| `SCRUB-VULN-{ID}` | OSV vulnerability (e.g. `SCRUB-VULN-CVE-2024-1234`) |
