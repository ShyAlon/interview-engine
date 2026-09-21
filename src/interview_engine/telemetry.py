"""Per-call token/cost logging. A launch requirement: the $5-per-run claim is
a belief until these records exist (Cost Addendum section 5).

Records StageTokenUsage per the vmodel-contracts schema. The sink is
pluggable: delivery-api subscribes over its API; the CLI demo appends JSONL.

The price table itself lives in vmodel-contracts (shared-module doctrine:
one table for every repo, 2026-09-12). The names are re-exported here so
existing imports keep working.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from vmodel_contracts import StageTokenUsage
from vmodel_contracts import (PRICE_TABLE_VERSION, PRICES_USD_PER_MTOK,
                              load_price_table, price_for)

_load_price_table = load_price_table  # backward-compatible private alias

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

    def record_estimated(self, stage: str, provider: str, model: str, usage: dict,
                         usd_per_mtok: float, duration_s: float | None = None) -> StageTokenUsage:
        """Record usage priced from an explicitly identified estimated rate."""
        rec = StageTokenUsage(
            session_id=self.session_id, stage=stage, provider=provider, model=model,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0), cached_tokens=0,
            price_per_mtok_prompt_usd=usd_per_mtok,
            price_per_mtok_completion_usd=usd_per_mtok,
        )
        self.records.append(rec); self.durations.append(duration_s)
        if self.sink: self.sink(rec)
        return rec

    def record_unpriced(self, stage: str, provider: str, model: str, usage: dict,
                        duration_s: float | None = None) -> StageTokenUsage:
        """Record measured remote usage when no authoritative price exists.

        Null price is deliberate and distinguishable from a genuinely free call.
        """
        rec = StageTokenUsage(
            session_id=self.session_id, stage=stage, provider=provider, model=model,
            prompt_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            completion_tokens=int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            cached_tokens=int(usage.get("cached_tokens", 0) or 0),
            price_per_mtok_prompt_usd=None, price_per_mtok_completion_usd=None,
        )
        self.records.append(rec); self.durations.append(duration_s)
        if self.sink: self.sink(rec)
        return rec

    def total_usd(self) -> float:
        return sum((r.cost_usd or 0.0) for r in self.records)

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
            if rec.cost_usd is None:
                s["unpriced_calls"] = s.get("unpriced_calls", 0) + 1
            else:
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

