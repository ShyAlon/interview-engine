"""Structured progress events - one stream, many consumers.

The CLI printer below is ONE consumer. delivery-api / the web UI will consume
the same event stream later (the founder watches a live step list, not a
spinner). The shape is deliberately identical across repos (spec-generator
carries the same module); when events cross a process boundary to the UI, the
shape promotes into vmodel-contracts with a minor version bump (ADR-0002).
Until then: intra-module, documented in ARCHITECTURE.md 'Progress events'.

Event shape:
    stage:     machine name, stable per pipeline step (e.g. 'distill_intent')
    label:     human-readable present-tense phrase ('Distilling intent model')
    status:    'started' | 'completed' | 'failed'
    ts:        epoch seconds when the event fired
    elapsed_s: seconds the stage took (completed/failed only)
    detail:    optional context ('turn 4', 'retrying after schema feedback')
"""
from __future__ import annotations

import sys
import time
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from . import model_config as _mc
from vmodel_contracts.runtime import format_console_line


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    label: str
    status: str          # started | completed | failed
    ts: float
    elapsed_s: float | None = None
    detail: str | None = None
    completed: int | None = None
    total: int | None = None


class ProgressBus:
    """Synchronous fan-out. A broken consumer must never crash a run, so
    subscriber errors are counted and safely reported (progress is observability,
    not control flow)."""

    def __init__(self) -> None:
        self._subscribers: list = []
        self.subscriber_failures: dict[int, int] = {}
        self._last_warning: dict[int, float] = {}

    def subscribe(self, fn) -> None:
        self._subscribers.append(fn)

    def emit(self, event: ProgressEvent) -> None:
        for index, fn in enumerate(self._subscribers):
            try:
                fn(event)
            except Exception as error:  # observability never kills the run
                self.subscriber_failures[index] = self.subscriber_failures.get(index, 0) + 1
                now = time.monotonic()
                if index not in self._last_warning or now - self._last_warning[index] >= 60:
                    self._last_warning[index] = now
                    # No event payload, exception text or callable repr in operator logs.
                    stage = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(event.stage))[:80]
                    try:
                        _logger.warning('progress.subscriber_failed subscriber=%d stage=%s category=%s count=%d',
                                        index, stage, type(error).__name__, self.subscriber_failures[index])
                    except Exception:
                        pass  # even a broken diagnostic handler must stay nonfatal


@contextmanager
def track(bus: ProgressBus | None, stage: str, label: str, detail: str | None = None):
    """Wrap a pipeline step: emits started, then completed/failed with elapsed."""
    t0 = time.time()
    if bus is not None:
        bus.emit(ProgressEvent(stage, label, "started", t0, None, detail))
    try:
        yield
    except Exception:
        if bus is not None:
            bus.emit(ProgressEvent(stage, label, "failed", time.time(), time.time() - t0, detail))
        raise
    if bus is not None:
        bus.emit(ProgressEvent(stage, label, "completed", time.time(), time.time() - t0, detail))


def stderr_printer():
    """The CLI consumer. Plain stderr lines; no TUI dependencies."""
    def _print(ev: ProgressEvent) -> None:
        with _mc._CONSOLE_LOCK:  # never interleave with the heartbeat thread
            _print_inner(ev)

    def _print_inner(ev: ProgressEvent) -> None:
        detail = f" ({ev.detail})" if ev.detail else ""
        if ev.status == "started":
            print(format_console_line(f"  -> {ev.label}{detail}...", ev.ts, stream=sys.stderr), file=sys.stderr)
        elif ev.status == "completed":
            print(format_console_line(f"  ok {ev.label}{detail} [{ev.elapsed_s:.1f}s]", ev.ts, "success", sys.stderr), file=sys.stderr)
        else:
            print(format_console_line(f"  !! {ev.label}{detail} FAILED after {ev.elapsed_s:.1f}s", ev.ts, "error", sys.stderr), file=sys.stderr)
    return _print

