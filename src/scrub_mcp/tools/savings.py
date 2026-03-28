"""Token and API call savings estimator.

Estimates the cloud LLM cost that S.C.R.U.B. avoided by handling a task locally.

Formula:
    file_tokens  = len(source) // 4               (character heuristic)
    tokens_saved = file_tokens * 2.2              (1.2x input context + 1x full-file output)
    calls_saved  = fixes // 25                    (~25 formatting tweaks per reliable LLM call)
    est_cost     = tokens_saved * price / 1e6     (configurable cloud API pricing)

When fixes == 0 tokens_saved and calls_saved are zero — no work was done, no savings to claim.
file_tokens is always reported so callers can see the file size even on clean passes.
"""

from __future__ import annotations


def estimate_savings(
    source: str,
    fixes: int,
    price_per_mtoken: float = 15.0,
    currency_unit: str = "USD",
) -> dict:
    """Estimate cloud tokens and cost avoided by running this task locally.

    Args:
        source: The Python source string that was processed.
        fixes: Number of items auto-fixed or generated (lint issues, docstrings, etc.).
        price_per_mtoken: Cloud API output token price per million tokens, in currency_unit.
        currency_unit: Label for the monetary unit (e.g. "USD", "EUR", "credits").

    Returns:
        Dict with keys: file_tokens, tokens_saved, calls_saved, est_cost, currency.
    """
    file_tokens = len(source) // 4
    if fixes <= 0:
        return {
            "file_tokens": file_tokens,
            "tokens_saved": 0,
            "calls_saved": 0,
            "est_cost": 0.0,
            "currency": currency_unit,
        }
    tokens_saved = int(file_tokens * 2.2)
    calls_saved = fixes // 25
    est_cost = round(tokens_saved * price_per_mtoken / 1_000_000, 4)
    return {
        "file_tokens": file_tokens,
        "tokens_saved": tokens_saved,
        "calls_saved": calls_saved,
        "est_cost": est_cost,
        "currency": currency_unit,
    }
