"""Question verification scores (JEV-D12, E5).

Deterministic checks on the question the interviewer model proposed, scored
every turn. Scoring is always on; whether a failed check triggers a retry is
the config flag verification.retry_enabled (default off). Every check records
its score and verdict so calibration can run from the debug outputs.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from . import question_config
from .coverage import implementation_question, question_substance
from .coverage_taxonomy import AREA_KEYS
from .session_retrieval import _score

# "Who is responsible for creating/building/maintaining ..." asked of the founder
# who already named the developer role: a founder-owned-role question.
_FOUNDER_ROLE_RE = re.compile(
    r"\bwho (?:is|will be|would be|are) (?:responsible|in charge|going) (?:for|to)\b[^?]{0,40}"
    r"\b(?:creat\w*|build\w*|develop\w*|maintain\w*|writ\w*|produc\w*)\b", re.I)
# A question about another area is rejected only on a clear signal.
_OTHER_AREA_MIN = 2


def _content_words(text: str) -> frozenset[str]:
    return question_substance(text)


def verify_question(question: str, target: str | None, spans: list[dict], parked_topics: list[str] | None = None,
                    cfg: dict | None = None) -> dict:
    """Return the per-check scores and an overall verdict.

    spans: the ranked retrieval spans for the target area ({id, text, score}).
    """
    cfg = cfg or question_config.current()
    need = cfg["verification"]["grounding_overlap_min"]
    parked_topics = parked_topics or []
    qwords = _content_words(question)

    best_id, best_overlap = None, 0
    for s in spans:
        overlap = len(qwords & _content_words(s["text"]))
        if overlap > best_overlap:
            best_id, best_overlap = s.get("uid", s["id"]), overlap
    parked_overlap = max((len(qwords & _content_words(t)) for t in parked_topics), default=0)
    if not spans:
        grounding = {"score": 0, "span": None, "result": "no_span", "passed": True}  # nothing to ground on
    else:
        ok = best_overlap >= need or parked_overlap >= need
        grounding = {"score": best_overlap, "span": best_id, "parked_overlap": parked_overlap,
                     "result": "grounded" if ok else "no_overlap", "passed": ok}

    impl = implementation_question(question)
    role = bool(_FOUNDER_ROLE_RE.search(question or ""))

    cues = dict(cfg["area_cues"])
    # The integrations area also owns the none-stated vocabulary: a closing question
    # that asks "is it fully self-contained?" must hit its own area.
    cues["integrations"] = list(cues.get("integrations", [])) + list(cfg["integrations"]["self_contained_cues"]) \
        + list(cfg["integrations"]["external_system_verbs"])
    scores = {a: _score(question, cues.get(a, [])) for a in AREA_KEYS}
    target_score = scores.get(target or "", 0)
    other_area, other_score = max(((a, v) for a, v in scores.items() if a != target), key=lambda t: t[1],
                                  default=(None, 0))
    on_target = not (target and target_score == 0 and other_score >= _OTHER_AREA_MIN)

    failed = [name for name, bad in (("grounding", not grounding["passed"]), ("implementation_level", impl),
                                     ("founder_role", role), ("off_target", not on_target)) if bad]
    return {
        "target_area": target,
        "grounding": grounding,
        "implementation_hit": impl,
        "founder_role_hit": role,
        "target_cue_score": target_score,
        "best_other_area": other_area,
        "best_other_score": other_score,
        "failed_checks": failed,
        "passed": not failed,
    }


class VerificationLog:
    """Appends one JSON object per turn, next to usage.jsonl."""

    def __init__(self, path: str | Path):
        from .diagnostic_sink import SinkStatus
        self._sink_status=SinkStatus('question-verification')
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        from .stop_fence import guarded_publish
        try:return guarded_publish(lambda:self._write(record))
        except OSError as error:self._sink_status.failed(error)

    def _write(self, record: dict) -> None:
        line = json.dumps({"ts": time.time(), **record}, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line + "\n")
