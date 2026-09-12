"""The coverage model (ADR-0016): the interview aims at a fixed checklist of
elicitation areas instead of drifting by turn count. The interviewer (option a)
self-reports what each founder answer covered in its structured reply; the
concrete-answers-only rule in the prompt keeps self-grading honest, and the
distillation-side mechanical guards (ADR-0015) catch optimism downstream.

Founder control, detected deterministically in the CLI/graph: "skip" marks the
current target area skipped (never re-asked), "enough" ends the interview.
Saturation ending: complete when every area is covered or skipped.
"""
from __future__ import annotations

import re

# (key, human label, what "concrete" means for closing it) - order is priority.
AREAS = [
    ("goal", "goal & core value", "the problem and who hurts, in a sentence"),
    ("users", "users & stakeholder classes", "primary users plus at least one other class (admins, support, integrators, regulators)"),
    ("flows", "core flows", "the 1-3 things a user actually does, end to end"),
    ("data", "data model & sources", "what data, where it lives, who owns it"),
    ("integrations", "integrations", "named systems/APIs, not categories"),
    ("nfrs", "quantified NFRs", "NUMBERS: users, volume, latency/freshness fail-point, uptime - 'strict security' is thin"),
    ("failures", "failure modes", "what breaks, what degrades, what must never happen"),
    ("constraints", "constraints", "budget, deadlines, team - and asked OPENLY: 'any technology constraints - language/ecosystem, cloud affinity?' ('agnostic' closes it too)"),
    ("risks", "risks", "what assumption kills the product if wrong"),
    ("done", "done criteria", "how the founder will know it works"),
    ("glossary", "glossary", "the founder's domain terms pinned to their meanings"),
]
AREA_KEYS = [k for k, _, _ in AREAS]
_LABELS = {k: label for k, label, _ in AREAS}


def turn_output_schema() -> dict:
    """Provider-neutral constrained-output shape for one interview turn.

    The area arrays are enum-constrained to code-owned keys. Content remains
    model-authored; JSON structure is enforced by the provider rather than by
    prompt compliance and a paid retry.
    """
    area = {"type": "string", "enum": AREA_KEYS}
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "covered": {"type": "array", "items": area},
            "thin": {"type": "array", "items": area},
            "skipped": {"type": "array", "items": area},
            "complete": {"type": "boolean"},
        },
        "required": ["question", "covered", "thin", "skipped", "complete"],
        "additionalProperties": False,
    }

UNCOVERED, THIN, COVERED, SKIPPED = "uncovered", "thin", "covered", "skipped"
_MARK = {UNCOVERED: "-", THIN: "~", COVERED: "+", SKIPPED: "x"}


def initial_coverage() -> dict[str, str]:
    return {k: UNCOVERED for k in AREA_KEYS}


def next_area(coverage: dict[str, str]) -> str | None:
    """The weakest area in priority order: uncovered first, then thin."""
    for want in (UNCOVERED, THIN):
        for k in AREA_KEYS:
            if coverage.get(k) == want:
                return k
    return None


def status_block(coverage: dict[str, str]) -> str:
    """Compact per-turn coverage status for the interviewer system prompt."""
    marks = " ".join(f"{k}={_MARK[coverage.get(k, UNCOVERED)]}" for k in AREA_KEYS)
    nxt = next_area(coverage)
    legend = "+ covered, ~ thin (mushy - needs concretes), - uncovered, x skipped by founder"
    block = f"COVERAGE [{marks}]  ({legend})"
    if nxt:
        label, concretes = next(((l, c) for k, l, c in AREAS if k == nxt))
        block += f"\nNEXT TARGET: {label} - closed only by: {concretes}."
    else:
        block += "\nAll areas covered or skipped - close the interview (complete=true)."
    return block


def progress_line(coverage: dict[str, str]) -> str:
    done = [k for k in AREA_KEYS if coverage.get(k) in (COVERED, SKIPPED)]
    nxt = next_area(coverage)
    line = f"coverage {len(done)}/{len(AREA_KEYS)} ({', '.join(done) or 'none yet'})"
    return line + (f" - next: {_LABELS[nxt]}" if nxt else " - saturated")


def apply_reply(coverage: dict[str, str], parsed: dict) -> dict[str, str]:
    """Fold the interviewer's self-reported statuses into the map. The model
    can only improve or skip an area, never un-cover one."""
    out = dict(coverage)
    for k in parsed.get("covered") or []:
        if k in out:
            out[k] = COVERED
    for k in parsed.get("thin") or []:
        if k in out and out[k] != COVERED:
            out[k] = THIN
    for k in parsed.get("skipped") or []:
        if k in out and out[k] != COVERED:
            out[k] = SKIPPED
    return out


_SKIP_RE = re.compile(r"^\s*(skip|pass)( this| it)?\s*[.!]?\s*$", re.IGNORECASE)
_ENOUGH_RE = re.compile(
    r"^\s*(enough|that'?s enough|wrap ?up|let'?s wrap|finish|done here|"
    r"move to (the )?(spec|distill)|end( the)? interview|end|stop|quit|exit|"
    r"(?:i (?:think|believe) )?you have enough(?: (?:to )?(?:move on|continue|proceed))?|"
    r"(?:i (?:think|believe) )?(?:we|you) can (?:move on|continue|proceed))\s*[.!]?\s*$",
    re.IGNORECASE)


_DEFERRAL_RE = re.compile(
    r"^\s*(?:i (?:do not|don't) know|not sure|no (?:idea|preference)|"
    r"(?:that is|that's|it is|it's|this is) (?:up to you|your call)|"
    r"you decide|your call|use (?:the )?(?:standard|best practice|state of the art)|"
    r"state[- ]of[- ]the[- ]art(?:[.! ]+you decide)?|whatever (?:is )?(?:standard|best)|"
    r"you do it|your choice|up to you|not up to me|whatever you (?:think|decide|choose)|"
    r"no[.!,]?\s+(?:you\s+(?:do\s+it|decide|choose|pick)|it'?s up to you)|"
    r"(?:i'?ll\s+|i will )?leave (?:it|that|this|the \w+) (?:up )?to you|"
    r"(?:i am|i'm) (?:agnostic|not technical enough to say))\s*[.!?]*\s*$",
    re.IGNORECASE)

# Topic-level dismissal: "move on from the UI", "next topic", "stop asking
# about X". Distinct from 'enough' (whole interview) and from a deferral
# (delegate ONE detail): the founder closes the current AREA.
_DISMISS_RE = re.compile(
    r"\bmove on from\b|"
    r"^(?:can|could) (?:we|you) (?:please )?move on\b|"
    r"^(?:next|another) (?:topic|question|area)\b|"
    r"^ask (?:me )?(?:about )?something else\b|"
    r"^(?:please )?(?:stop|don'?t) asking (?:me )?(?:about|for)\b|"
    r"^(?:can|could) (?:we|you) (?:please )?(?:drop|skip) (?:this|that|it)\b",
    re.IGNORECASE)

_TOPIC_PATTERNS = (
    ("security/encryption detail", re.compile(r"\b(encrypt|encryption|cipher|tls|ssl|cryptograph|security protocol)\w*\b", re.I)),
    ("identity/authentication detail", re.compile(r"\b(auth|authentication|authorization|oauth|mfa|login protocol)\w*\b", re.I)),
    ("scale/capacity target", re.compile(r"\b(scale|capacity|concurrent|users per|requests per|throughput)\w*\b", re.I)),
    ("latency target", re.compile(r"\b(latency|response time|how fast|milliseconds?)\b", re.I)),
    ("availability target", re.compile(r"\b(uptime|availability|downtime|sla)\w*\b", re.I)),
    ("technology-stack detail", re.compile(r"\b(programming language|framework|tech stack)\b", re.I)),
    ("database detail", re.compile(r"\b(database|sql|nosql|data store)\b", re.I)),
    ("persistence/storage detail", re.compile(r"\b(persist\w*|storage|data store)\b", re.I)),
    ("cloud/infrastructure detail", re.compile(r"\b(cloud|aws|azure|gcp|infrastructure|hosting)\b", re.I)),
)


def is_deferral(text: str) -> bool:
    """A founder explicitly delegates or cannot answer the current design detail."""
    return bool(_DEFERRAL_RE.match(text or ""))


def founder_steer(text: str) -> str | None:
    """An explicit founder steer (2026-09-12 transcript: 'No. You do it.',
    'Can you move on from the UI?'): 'delegate' hands the current detail to
    implementation; 'dismiss' closes the current area entirely. Both are
    terminal - no rephrase-and-ask-again. Whole-message matching only for
    deferrals; dismissal needs one clear phrase."""
    t = (text or "").strip()
    if is_deferral(t):
        return "delegate"
    if _DISMISS_RE.search(t):
        return "dismiss"
    return None


# Implementation-choice questions the interviewer must never ask (the same
# 2026-09-12 transcript: 'Which UI framework do you envision...?'). Technology
# selection belongs to the spec stage under the system baseline; the interview
# owns product intent and behavior. Deterministic guard, prompt-independent.
_IMPL_QUESTION_RE = re.compile(
    r"\b(?:which|what|pick|choose|envision)\b[^?]{0,90}"
    r"\b(?:framework|librar\w*|toolkit|programming language|scripting language|"
    r"tech(?:nology)? stack|database (?:engine|system|technology|solution)|orm|cms|"
    r"(?:data )?storage (?:solution|engine|system|technology|mechanism)|"
    r"persistence (?:solution|layer|mechanism|technology)|"
    r"hosting (?:provider|platform)|auth(?:entication)? (?:provider|solution)|"
    r"ui (?:framework|toolkit|library))\b",
    re.IGNORECASE)
# Constraint/affinity questions are product-level and legitimate (the
# constraints area asks them openly; 'agnostic' closes it too).
_IMPL_ALLOW_RE = re.compile(
    r"constraint|affinity|mandat|must (?:use|be)|existing|already|locked", re.IGNORECASE)


def implementation_question(question: str) -> bool:
    """True when the question asks the founder to CHOOSE an implementation."""
    q = question or ""
    return bool(_IMPL_QUESTION_RE.search(q)) and not _IMPL_ALLOW_RE.search(q)


def question_topic(question: str) -> str | None:
    """Return a code-owned sub-topic label for repeat-probe prevention."""
    for label, pattern in _TOPIC_PATTERNS:
        if pattern.search(question or ""):
            return label
    return None


_FALLBACK_QUESTIONS = {
    "goal": "What outcome would make this product worth building?",
    "users": "Who besides the primary user will operate, support, or be affected by it?",
    "flows": "What is the most important user flow from start to finish?",
    "data": "What information goes in, where does it come from, and who owns it?",
    "integrations": "Which external systems must exchange data with it?",
    "nfrs": "Apart from parked design details, what user volume, response-time, or uptime target matters most?",
    "failures": "What failure must the system handle without losing the user's work?",
    "constraints": "What fixed deadline, budget, team, platform, or compliance constraint applies?",
    "risks": "What assumption would most threaten the product if it proved false?",
    "done": "What observable result will tell you the product is ready?",
    "glossary": "Which domain terms need a precise shared meaning?",
}


def safe_question(question: str, parked_topics: list[str], target: str | None) -> str:
    """Mechanically prevent a model from re-probing a parked sub-topic."""
    topic = question_topic(question)
    if topic and topic in parked_topics:
        return _FALLBACK_QUESTIONS.get(target or "", "What important detail have we not covered yet?")
    return question


def guard_question(question: str, parked_topics: list[str], target: str | None) -> str:
    """safe_question plus the implementation-choice guard: a technology-choice
    question is replaced by the product-level fallback for the target area
    (the behavior need stays with the interview; the tech choice moves to the
    spec stage)."""
    if implementation_question(question):
        return _FALLBACK_QUESTIONS.get(target or "", "What important detail have we not covered yet?")
    return safe_question(question, parked_topics, target)


def founder_control(text: str) -> str | None:
    """'skip' -> skip the current target area; 'enough' -> end the interview.
    Deliberately strict: a sentence merely CONTAINING 'skip' is not a command."""
    t = text.strip()
    if _SKIP_RE.match(t):
        return "skip"
    if _ENOUGH_RE.match(t):
        return "enough"
    return None


def extract_turn_json(text: str) -> dict | None:
    """Tolerant structured-turn parse: whole-body JSON, fenced JSON, or the
    first balanced {...} object embedded in prose. Returns None if no object
    with a string 'question' is found."""
    import json
    candidate = text.strip().strip("`")
    if candidate.startswith("json"):
        candidate = candidate[4:].strip()
    for attempt in (candidate, text):
        try:
            obj = json.loads(attempt)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("question"), str):
            return obj
    start = text.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict) and isinstance(obj.get("question"), str):
                return obj
        except ValueError:
            pass
    return None


_NUM_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(%|percent|ms|s(?:ec)?|seconds|minutes|hours|users|req(?:uests)?/?s|"
    r"events?/?s|rps|qps|tps|gb|mb|tb|k\b|m\b|million|thousand|nines|9s|uptime|latency)",
    re.IGNORECASE)
_NO_NUMBERS_RE = re.compile(
    r"\b(no|don't have|do not have|without|skip) (the )?numbers?\b"
    r"|\bno (metrics|quantification|targets?|slas)\b"
    r"|\bnot applicable\b|\bdemo (only|mode|purposes)\b|\bjust a demo\b",
    re.IGNORECASE)


def nfr_quantified(founder_text: str) -> bool:
    """The 'quantified NFRs' area may close only on numbers in the founder's own
    words, or an explicit founder acknowledgment that there are none (demo).
    'Strict security' and a model's confidence are not quantification."""
    return bool(_NUM_RE.search(founder_text) or _NO_NUMBERS_RE.search(founder_text))

