# S.C.R.U.B. Quickstart

## 1. Install

```bash
pip install scrub-mcp
pip install scrub-mcp[security]    # adds Bandit (security_scan, security_remediate)
pip install scrub-mcp[prefilter]   # adds pyright + pydocstyle (stricter pre-filtering)
pip install scrub-mcp[all]         # everything
```

## 2. Start a local model

S.C.R.U.B. requires a local LLM via [Ollama](https://ollama.com). Recommended for most machines:

```bash
ollama pull qwen2.5-coder:7b
```

See [README.md](README.md#llm-considerations) for GPU/RAM guidance on larger models.

## 3. Configure

Create `config.yaml` in your project root:

```yaml
model:
  provider: ollama
  model: qwen2.5-coder:7b
  base_url: http://localhost:11434
  max_tokens: 4096
  temperature: 0.1

batch_size: 5
deterministic_prefilter: true
```

## 4. Connect to your MCP client

**Claude Code** — add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "scrub": {
      "command": "python",
      "args": ["-m", "scrub_mcp.mcp.server"]
    }
  }
}
```

**VS Code / any MCP client** — same config. S.C.R.U.B. speaks MCP over stdio.

Verify the server starts:

```bash
python -m scrub_mcp.mcp.server
```

## 5. Drop agent instructions

Tell the cloud LLM to route through S.C.R.U.B. instead of writing boilerplate itself:

```bash
python -m scrub_mcp.mcp.server --agent-instructions .
```

This writes `AGENTS.md` into the current directory. It establishes the agent's role as **Cloud Orchestrator** — plan and delegate, don't generate boilerplate — and includes:

- A division-of-labor table (what you do vs. what S.C.R.U.B. does)
- Mandatory workflow checklists (hygiene before every file, security before every commit)
- The bouncer rule: a decision tree that intercepts manual docstring/type/test writing

Commit both `.mcp.json` and `AGENTS.md` to your repo so every contributor and every CI agent gets the same instructions.

## 6. Optimize the local model (recommended)

S.C.R.U.B. ships with bundled training examples. Run the optimizer once so the local model learns your project's conventions:

**Offline — free, uses your local model only:**

```bash
python -m scrub_mcp.optimizers.tune
```

**Online — one-time Claude API cost, better prompt quality:**

```bash
python -m scrub_mcp.optimizers.tune --teacher --teacher-model claude-sonnet-4-20250514
```

Compiled prompts are saved to `.dspy_cache/`. The pipeline runs without this step — just with generic prompts.

## 7. Generate more training examples (optional)

The bundled examples cover 2 domains. To expand your training set:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Generate 10 examples, then tune offline
python -m scrub_mcp.optimizers.tune --build-examples ./examples --build-count 10

# Generate examples and tune with teacher in one shot
python -m scrub_mcp.optimizers.tune \
    --build-examples ./examples --build-count 10 \
    --teacher --teacher-model claude-sonnet-4-20250514
```

Each topic produces three files: `{topic}_messy.py` (undocumented input), `{topic}_clean.py` (annotated ground truth), and `{topic}_test.py` (pytest tests). Commit them to `src/scrub_mcp/examples/` to bundle with the package for all users.

## 8. Use the tools

With S.C.R.U.B. connected and `AGENTS.md` in place, the cloud LLM will call tools automatically. You can also invoke them directly:

```
Run hygiene_full on src/mymodule.py
```

Full tool list:

| Tool | When |
|------|------|
| `hygiene_full` | First action on any file (lint + docs + types + comments) |
| `lint_file` | Lint-only pass |
| `generate_docstrings` | Docstrings only |
| `annotate_types` | Type annotations only |
| `add_comments` | Inline comments for complex functions |
| `analyze_complexity` | Before any refactor |
| `suggest_simplifications` | Simplify complex functions |
| `suggest_refactoring` | Extract function / rename decisions |
| `optimize_imports` | After any import change |
| `generate_tests` | pytest generation |
| `find_dead_code` | Before deleting suspected dead code |
| `security_scan` | Before every commit |
| `security_remediate` | Fix findings from security_scan |
| `generate_sbom` | After dependency changes |
| `scan_vulnerabilities` | CVE scan before release |

## 9. Check optimization health (CI)

```bash
# Warn if any compiled prompt scores below 0.7
python -m scrub_mcp.optimizers.health --threshold 0.7
```

Non-zero exit if stale. Run this in CI when you upgrade your local model.
