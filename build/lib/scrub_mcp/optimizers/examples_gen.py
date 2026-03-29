"""Generate messy/clean/test training example triplets using Claude API.

Each topic produces three files:
  {topic}_messy.py  — realistic Python with no docstrings or type annotations
  {topic}_clean.py  — same code, fully documented and typed (ground truth)
  {topic}_test.py   — pytest tests for the most testable functions

Accessible after pip install via:
  python -m scrub_mcp.optimizers.tune --build-examples ./examples
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

log = logging.getLogger(__name__)

TOPICS = [
    "retry_backoff",
    "csv_transformer",
    "event_bus",
    "metric_aggregator",
    "task_queue",
    "rate_limiter",
    "date_utils",
    "file_watcher",
    "json_validator",
    "diff_calculator",
    "connection_pool",
    "env_loader",
    "progress_tracker",
    "config_loader",
    "html_scraper",
]

_DOCSTRING_RULES = """\
- Google style: Args / Returns / Raises sections
- Oxford commas in all lists
- No em dashes (use commas or parentheses instead)
- First line is a verb phrase (not the function name)
- One blank line between sections
- Include Raises only when the function actually raises"""

_TYPE_RULES = """\
- Lowercase generics: list[str] not List[str]
- Union syntax: str | None not Optional[str]
- Specific collections: dict[str, int] not dict
- Path | str for filesystem args
- Always include return type (even -> None)"""

_TEST_RULES = """\
- Names: test_<function>_<scenario>
- assert statements only (no unittest assertEqual)
- pytest.raises for expected exceptions
- @pytest.mark.parametrize for multiple inputs
- No mocking unless the function calls external I/O"""

# Inline clean-style example so the module has no dependency on samples/
_CLEAN_HINT = """\
class LogParser:
    def __init__(self, log_dir: str | Path, patterns: dict[str, str] | None = None) -> None:
        \"\"\"Initialize the LogParser.

        Args:
            log_dir: Path to the directory containing log files.
            patterns: Mapping of pattern names to regex strings.
                Defaults to an empty dict if not provided.
        \"\"\"
        ...
"""


def _extract_fenced(text: str, lang: str = "") -> str:
    """Extracts the content within a fenced code block from the given text.

    Args:
        text (str): The input text containing the fenced code block.
        lang (str): The language of the code block. If empty, any code block will be extracted.

    Returns:
        str: The content within the fenced code block, stripped of leading/trailing whitespace.
    """
    pattern = rf"```{lang}\s*\n(.*?)```" if lang else r"```(?:\w+)?\s*\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _call(client, model: str, prompt: str, max_tokens: int) -> str:
    """Calls a language model using the provided client and parameters.

    Args:
        client: The client object used to make the API call.
        model (str): The name of the language model to use.
        prompt (str): The input prompt for the model.
        max_tokens (int): The maximum number of tokens to generate in the response.

    Returns:
        str: The generated response from the language model.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def generate_messy(client, model: str, topic: str) -> str:
    """Generate an undocumented Python module for *topic*."""
    prompt = f"""\
Write a realistic Python module implementing utilities for: **{topic}**.

Hard rules:
- NO docstrings anywhere (no triple-quoted strings on functions, classes, or module).
- NO type annotations on any parameter or return value.
- At least 3 functions OR 1 class with 3+ methods.
- At least one function must have cyclomatic complexity >= 4 (real nested loops/branches).
- Meaningful variable names and realistic logic (not toy/demo code).
- stdlib imports only.

Output ONLY the Python source inside a ```python block. No explanation.
"""
    raw = _call(client, model, prompt, 2048)
    return _extract_fenced(raw, "python") or _extract_fenced(raw)


def generate_clean(client, model: str, topic: str, messy: str) -> str:
    """Generate the fully documented and typed version of *messy*."""
    prompt = f"""\
Add documentation and type annotations to this Python file (topic: {topic}).

Produce a clean version that adds:
1. Google-style docstrings to every function and method.
2. Type annotations to every parameter and return value.
3. A module-level docstring (one sentence).

Docstring style rules:
{_DOCSTRING_RULES}

Type annotation style rules:
{_TYPE_RULES}

Example of the expected style:
```python
{_CLEAN_HINT}
```

Source to annotate:
```python
{messy}
```

Output ONLY the annotated Python source inside a ```python block.
Do NOT change any logic, variable names, or structure.
"""
    raw = _call(client, model, prompt, 3072)
    return _extract_fenced(raw, "python") or _extract_fenced(raw)


def generate_tests(client, model: str, topic: str, clean: str) -> str:
    """Generate a pytest test file for the functions in *clean*."""
    prompt = f"""\
Write a pytest test file for the Python module below (topic: {topic}).

Test the 2-3 most testable functions (prefer pure functions with no I/O).

Test style rules:
{_TEST_RULES}

Source:
```python
{clean}
```

Import from the module filename `{topic}_messy` (e.g. `from {topic}_messy import func`).

Output ONLY the test source inside a ```python block. No explanation.
"""
    raw = _call(client, model, prompt, 2048)
    return _extract_fenced(raw, "python") or _extract_fenced(raw)


def _validate(messy: str, clean: str) -> list[str]:
    """Validates the content of a messy and clean file, identifying potential issues.

    Args:
        messy (str): The content of the messy file.
        clean (str): The content of the clean file.

    Returns:
        list[str]: A list of warnings, each describing an issue found in the files.
    """
    warns = []
    if '"""' in messy or "'''" in messy:
        warns.append("messy file contains triple-quoted strings (docstrings leaked in)")
    if not clean.strip():
        warns.append("clean file is empty")
    if '"""' not in clean and "'''" not in clean:
        warns.append("clean file has no docstrings")
    return warns


def generate_triplet(
    client,
    model: str,
    topic: str,
    output_dir: Path,
    overwrite: bool = False,
) -> bool:
    """Generate and save a messy/clean/test triplet for *topic*.

    Args:
        client: anthropic.Anthropic client instance.
        model: Claude model to use for generation.
        topic: Short topic name (e.g. "retry_backoff").
        output_dir: Directory to write the three files into.
        overwrite: If False, skip topics that already have all three files.

    Returns:
        True if the triplet was saved successfully, False on error.

    """
    messy_path = output_dir / f"{topic}_messy.py"
    clean_path = output_dir / f"{topic}_clean.py"
    test_path = output_dir / f"{topic}_test.py"

    if messy_path.exists() and clean_path.exists() and test_path.exists() and not overwrite:
        log.info("[skip] %s already exists", topic)
        return True

    log.info("[gen ] %s — messy source...", topic)
    try:
        messy = generate_messy(client, model, topic)
    except Exception as exc:
        log.error("[fail] %s — messy generation failed: %s", topic, exc)
        return False

    log.info("[gen ] %s — clean (documented) version...", topic)
    try:
        clean = generate_clean(client, model, topic, messy)
    except Exception as exc:
        log.error("[fail] %s — clean generation failed: %s", topic, exc)
        return False

    log.info("[gen ] %s — tests...", topic)
    try:
        tests = generate_tests(client, model, topic, clean)
    except Exception as exc:
        log.warning("[warn] %s — test generation failed: %s (saving without tests)", topic, exc)
        tests = ""

    for w in _validate(messy, clean):
        log.warning("[warn] %s — %s", topic, w)

    messy_path.write_text(messy)
    clean_path.write_text(clean)
    if tests:
        test_path.write_text(tests)
        log.info("[ok  ] %s — saved messy + clean + test", topic)
    else:
        log.info("[ok  ] %s — saved messy + clean (no tests)", topic)
    return True


def generate_examples(
    output_dir: Path,
    model: str = "claude-sonnet-4-20250514",
    count: int = 10,
    topics: list[str] | None = None,
    overwrite: bool = False,
) -> int:
    """Generate training example triplets into *output_dir*.

    Args:
        output_dir: Directory to write generated files into.
        model: Claude model to use. Defaults to claude-sonnet-4-20250514.
        count: Number of triplets to generate. Defaults to 10.
        topics: Specific topic names to generate. If None, selects randomly
            from the built-in list.
        overwrite: Regenerate files that already exist. Defaults to False.

    Returns:
        Number of triplets successfully generated.

    Raises:
        ImportError: If the anthropic package is not installed.
        anthropic.AuthenticationError: If ANTHROPIC_API_KEY is not set or invalid.

    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package is required for --build-examples. "
            "Install it with: pip install anthropic"
        ) from exc

    selected = topics if topics else random.sample(TOPICS, min(count, len(TOPICS)))
    output_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    ok = 0
    for topic in selected[:count]:
        if generate_triplet(client, model, topic, output_dir, overwrite):
            ok += 1

    log.info("Generated %d/%d triplets in %s", ok, min(len(selected), count), output_dir)
    return ok
