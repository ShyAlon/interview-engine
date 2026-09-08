"""Per-call token/cost logging. A launch requirement: the $5-per-run claim is
a belief until these records exist (Cost Addendum section 5).

Records StageTokenUsage per the vmodel-contracts schema. The sink is
pluggable: delivery-api subscribes over its API; the CLI demo appends JSONL.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from vmodel_contracts import StageTokenUsage

# Versioned price table (USD per MTok). Fetched and corrected 2026-09-08:
# Sonnet 5's Sept 1 increase did NOT take effect; $2/$10 stands.
# Update here AND bump PRICE_TABLE_VERSION - price drift is real (two changes
# this month) and silent drift makes cost reports lie retroactively.
PRICE_TABLE_VERSION = "2026-09-08"
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "deepseek-chat": (0.27, 1.10),        # verify against deepseek.com pricing before first real run
    "deepseek-reasoner": (0.55, 2.19),    # verify before use
    "claude-sonnet-5": (2.00, 10.00),     # platform.claude.com, 2026-09-08
    "claude-haiku-4.5": (1.00, 5.00),     # platform.claude.com, 2026-09-08
    "gpt-5": (1.25, 10.00),               # OpenAI model page, 2026-09-08
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}


def price_for(model: str, provider: str = "") -> tuple[float, float]:
    if provider == "ollama":
        return (0.0, 0.0)  # local compute: $0 marginal, logged for the record
    if model not in PRICES_USD_PER_MTOK:
        raise KeyError(f"no price for {model!r} in table v{PRICE_TABLE_VERSION} - add it; unpriced calls are unbudgeted calls")
    return PRICES_USD_PER_MTOK[model]


Sink = Callable[[StageTokenUsage], None]


def jsonl_sink(path: str | Path) -> Sink:
    def _write(u: StageTokenUsage) -> None:
        with open(path, "a") as f:
            f.write(u.model_dump_json() + "\n")
    return _write


class UsageLog:
    """Collects usage for one run; delivery-api turns it into a RunCostReport."""

    def __init__(self, session_id: str, sink: Sink | None = None):
        self.session_id = session_id
        self.records: list[StageTokenUsage] = []
        self.sink = sink

    def record(self, stage: str, provider: str, model: str, usage: dict) -> StageTokenUsage:
        pin, pout = price_for(model, provider)
        rec = StageTokenUsage(
            session_id=self.session_id,
            stage=stage,
            provider=provider,
            model=model,
            prompt_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            completion_tokens=int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            cached_tokens=int((usage.get("input_token_details") or {}).get("cache_read", usage.get("cached_tokens", 0)) or 0),
            price_per_mtok_prompt_usd=pin,
            price_per_mtok_completion_usd=pout,
        )
        self.records.append(rec)
        if self.sink:
            self.sink(rec)
        return rec

    def total_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)
