"""Per-call token/cost logging. A launch requirement: the $5-per-run claim is
a belief until these records exist (Cost Addendum section 5).

Records StageTokenUsage per the vmodel-contracts schema. The sink is
pluggable: delivery-api subscribes over its API; the CLI demo appends JSONL.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from vmodel_contracts import StageTokenUsage

# Versioned price table (USD per MTok). Fetched and corrected 2026-09-08:
# Sonnet 5's Sept 1 increase did NOT take effect; $2/$10 stands.
# Update here AND bump PRICE_TABLE_VERSION - price drift is real (two changes
# this month) and silent drift makes cost reports lie retroactively.
PRICE_TABLE_VERSION = "2026-09-10"
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "deepseek-flash": (0.30, 1.20),       # V4.1 Flash peak; off-peak $0.15/$0.60
    "deepseek-v4-pro": (1.32, 3.96),      # peak; off-peak $0.66/$1.98; retires to Flash 2026-09-14
    "claude-sonnet-5": (2.00, 10.00),     # platform.claude.com, 2026-09-08
    "claude-haiku-4.5": (1.00, 5.00),     # platform.claude.com, 2026-09-08
    "gpt-5": (1.25, 10.00),               # OpenAI model page, 2026-09-08
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}


def _load_price_table() -> dict[str, tuple[float, float]]:
    """Built-in defaults, overridable without a code change: point
    VMODEL_PRICE_TABLE at a JSON file {"model": [input_per_mtok, output_per_mtok], ...}.
    The built-in table's prices are unverified snapshots (see note above) - the
    operator's file WINS on conflict, so correcting a price is a config edit,
    not a release."""
    table = dict(PRICES_USD_PER_MTOK)
    path = os.environ.get("VMODEL_PRICE_TABLE")
    if path:
        with open(path) as f:
            override = json.load(f)
        for model, pair in override.items():
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ValueError(f"VMODEL_PRICE_TABLE entry {model!r} must be [input, output] per MTok")
            table[model] = (float(pair[0]), float(pair[1]))
    return table


def price_for(model: str, provider: str = "") -> tuple[float, float]:
    # Local by construction, not by table lookup: any ollama-routed call is $0.
    # The hard-fail below is reserved for UNPRICED REMOTE calls - a missing
    # price for a paid model means an unbudgeted call, and that must crash loudly.
    if provider == "ollama" or model.startswith("ollama:") or model.startswith("ollama/"):
        return (0.0, 0.0)
    table = _load_price_table()
    if model not in table:
        raise KeyError(f"no price for {model!r} in table v{PRICE_TABLE_VERSION} - add it via VMODEL_PRICE_TABLE; unpriced calls are unbudgeted calls")
    return table[model]


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
        self.durations: list[float | None] = []
        self.sink = sink

    def record(self, stage: str, provider: str, model: str, usage: dict,
               duration_s: float | None = None) -> StageTokenUsage:
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
        self.durations.append(duration_s)
        if self.sink:
            self.sink(rec)
        return rec

    def total_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def summary(self) -> dict:
        """Per-stage rollup: calls, tokens, LLM seconds, cost. The end-of-run
        artifact and CLI table both render from this."""
        stages: dict[str, dict] = {}
        for rec, dur in zip(self.records, self.durations):
            s = stages.setdefault(rec.stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                              "seconds": 0.0, "cost_usd": 0.0, "models": set()})
            s["calls"] += 1
            s["prompt_tokens"] += rec.prompt_tokens
            s["completion_tokens"] += rec.completion_tokens
            s["seconds"] += dur or 0.0
            s["cost_usd"] += rec.cost_usd
            s["models"].add(f"{rec.provider}:{rec.model}")
        for s in stages.values():
            s["models"] = sorted(s["models"])
        return {"session_id": self.session_id,
                "stages": stages,
                "total_seconds": sum(s["seconds"] for s in stages.values()),
                "total_cost_usd": self.total_usd()}

    def write_summary(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2))

