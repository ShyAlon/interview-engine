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

from .coverage_taxonomy import AREAS, AREA_KEYS, LABELS as _LABELS  # single source of truth



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

UNCOVERED, THIN, COVERED, SKIPPED, DELEGATED = "uncovered", "thin", "covered", "skipped", "delegated"
# Founder postponed the area's open question ('I'll check and get back to
# you'). Still open: never counted as covered or delegated, but the interview
# moves on instead of re-asking now. A later concrete answer can still cover it.
DEFERRED = "deferred"
_MARK = {UNCOVERED: "-", THIN: "~", COVERED: "+", SKIPPED: "x", DELEGATED: "d", DEFERRED: "?"}


def initial_coverage() -> dict[str, str]:
    return {k: UNCOVERED for k in AREA_KEYS}


def next_area(coverage: dict[str, str]) -> str | None:
    """The weakest area in priority order: uncovered first, then thin."""
    for want in (UNCOVERED, THIN):
        for k in AREA_KEYS:
            if coverage.get(k) == want:
                return k
    return None


def status_block(coverage: dict[str, str], target: str | None = None) -> str:
    """Compact per-turn coverage status for the interviewer system prompt.
    ``target`` overrides the priority pick (the partial-answer drill keeps the
    thin area it is following up on)."""
    marks = " ".join(f"{k}={_MARK[coverage.get(k, UNCOVERED)]}" for k in AREA_KEYS)
    nxt = target or next_area(coverage)
    legend = "+ covered, d covered by explicit delegation, ~ thin, - uncovered, x skipped, ? deferred by founder (open, answer pending)"
    block = f"COVERAGE [{marks}]  ({legend})"
    if nxt:
        label, concretes = next(((l, c) for k, l, c in AREAS if k == nxt))
        block += f"\nNEXT TARGET: {label} - closed only by: {concretes}."
    else:
        block += "\nAll areas covered or skipped - close the interview (complete=true)."
    deferred = [k for k in AREA_KEYS if coverage.get(k) == DEFERRED]
    if deferred:
        block += ("\nDEFERRED BY FOUNDER (still open; they will answer later - do not re-ask now, do not treat as decided): "
                  + ", ".join(_LABELS[k] for k in deferred))
    return block


def progress_line(coverage: dict[str, str]) -> str:
    done = [k for k in AREA_KEYS if coverage.get(k) in (COVERED, SKIPPED, DELEGATED)]
    nxt = next_area(coverage)
    line = f"coverage {len(done)}/{len(AREA_KEYS)} ({', '.join(done) or 'none yet'})"
    deferred = [k for k in AREA_KEYS if coverage.get(k) == DEFERRED]
    if deferred:
        line += f" - deferred by founder: {', '.join(deferred)}"
    return line + (f" - next: {_LABELS[nxt]}" if nxt else " - saturated")


def apply_reply(coverage: dict[str, str], parsed: dict) -> dict[str, str]:
    """Fold the interviewer's self-reported statuses into the map. The model
    can only improve or skip an area, never un-cover one."""
    out = dict(coverage)
    for k in parsed.get("covered") or []:
        if k in out and out[k] not in (SKIPPED, DELEGATED):
            out[k] = COVERED
    for k in parsed.get("thin") or []:
        if k in out and out[k] not in (COVERED, SKIPPED, DELEGATED):
            out[k] = THIN
    for k in parsed.get("skipped") or []:
        if k in out and out[k] not in (COVERED, DELEGATED):
            out[k] = SKIPPED
    return out


_SKIP_RE = re.compile(r"^\s*(skip|pass)( this| it)?\s*[.!]?\s*$", re.IGNORECASE)
_ENOUGH_RE = re.compile(
    r"^\s*(enough|that'?s enough|wrap ?up|let'?s wrap|finish|done here|"
    r"move to (the )?(spec|distill)|end( the)? interview|end|stop|quit|exit|"
    r"(?:i (?:think|believe) )?you have enough(?: (?:to )?(?:move on|continue|proceed))?|"
    r"(?:i (?:think|believe) )?(?:we|you) can (?:continue|proceed))\s*[.!]?\s*$",
    re.IGNORECASE)


_DEFERRAL_RE = re.compile(
    r"^\s*(?:i (?:do not|don't) know(?![^.!?]*(?:but i can|i can get|will find|will check))\b.*|not sure(?![^.!?]*(?:but i can|i can get|will find|will check))\b.*|no (?:idea|preference)\b.*|"
    r"(?:that is|that's|this is|it is|it's) (?:for|up to) (?:the )?(?:subject matter )?experts?\b.*|"
    r"(?:the )?(?:subject matter )?experts? (?:will|should|can|need to) (?:answer|decide|define|specify|determine)\b.*|"
    r"(?:it|that|this) (?:does not|doesn't) need to\b.*|"
    r"(?:that is|that's|it is|it's|this is) (?:up to you|your call)|"
    r"you decide|your call|use (?:the )?(?:standard|best practice|state of the art)|"
    r"state[- ]of[- ]the[- ]art(?:[.! ]+you decide)?|whatever (?:is )?(?:standard|best)|"
    r"you do it|your choice|up to you|not up to me|whatever you (?:think|decide|choose)|"
    r"you suggest|you pick|i'?m good with any\b.*|anything(?:'?s| is) fine(?: by me)?|"
    r"whatever works|i have no preference|"
    r"no[.!,]?\s+(?:you\s+(?:do\s+it|decide|choose|pick)|it'?s up to you)|"
    r"(?:i'?ll\s+|i will )?leave (?:it|that|this|the \w+) (?:up )?to you|"
    r"(?:i am|i'm) (?:agnostic|not technical enough to say)|"
    r"(?:the |our )?[\w -]+ team (?:will|can|should) (?:decide|select|choose|approve)(?: the final (?:choice|selection))?|"
    r"(?:[a-z][\w -]+) (?:will|can|should) (?:decide|select|choose|approve)(?: the final (?:choice|selection))?)\s*[.!?]*\s*$",
    re.IGNORECASE)

# Move-on / decline phrases: "move on from the UI", "next topic", "stop asking
# about X". Distinct from 'enough' (whole interview) and from a deferral
# (delegate ONE detail). v1.3: these are skip (dismiss was folded into skip).
_DISMISS_RE = re.compile(
    r"^this is (?:an? )?implementation\b|"
    r"^(?:please )?move on\b|"
    r"\bmove on from\b|"
    r"^(?:can|could) (?:we|you) (?:please )?move on\b|"
    r"^(?:next|another) (?:topic|question|area)\b|"
    r"^ask (?:me )?(?:about )?something else\b|"
    r"^(?:please )?(?:stop|don'?t) asking (?:me )?(?:about|for)\b|"
    r"^(?:can|could) (?:we|you) (?:please )?(?:drop|skip) (?:this|that|it)\b|"
    r"^(?:please )?(?:leave|drop|skip) (?:the |this |that )?[a-z][\w -]*(?:data|topic|question|detail|area)\b|"
    r"^(?:please )?(?:leave|drop|skip) (?:the |this |that )?(?:historical|training) data\b",
    re.IGNORECASE)

_TOPIC_PATTERNS = (
    ("historical-data source selection", re.compile(r"\b(?:(?:historical|training) (?:data (?:source|sources|set|sets|provider|providers)|datasets?)|data sources? (?:for )?(?:training|history|historical))\b", re.I)),
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


# Explicit postponement (2026-09-23 taxonomy): the founder will answer later.
# Not a delegation (nobody was handed the choice) and not settled.
_DEFER_RE = re.compile(
    r"\b(?:i'?ll|i will|we'?ll|we will|let me|i can) (?:\w+ )?(?:get|come) back to you\b|"
    r"\b(?:i|we) (?:need|have|'?d need|would need) to (?:ask|check with|talk to|consult|confirm (?:it |this |that )?with)\b|"
    r"\blet me (?:ask|check with|talk to|confirm (?:it |this |that )?with|find out)\b|"
    r"\b(?:i'?ll|i will|we'?ll) (?:ask(?! you\b)|check with|talk to|confirm (?:it |this |that )?with|find out)\b",
    re.IGNORECASE)


def is_postponement(text: str) -> bool:
    """Short replies anywhere; long replies only when the postponement leads,
    so a substantive answer with one 'I'll check X' aside stays an answer."""
    t = (text or "").strip()
    if len(t.split()) > 25:
        t = re.split(r"(?<=[.!?])\s+", t, maxsplit=1)[0]
    return bool(_DEFER_RE.search(t))


def founder_steer(text: str) -> str | None:
    """An explicit founder steer (2026-09-12 transcript: 'No. You do it.',
    'Can you move on from the UI?'): 'delegate' hands the current detail to
    implementation; 'skip' moves on from the current question or topic
    (v1.3 - this used to be 'dismiss'). Both are terminal - no
    rephrase-and-ask-again. Whole-message matching only for deferrals; a skip
    needs one clear phrase."""
    t = (text or "").strip()
    if is_postponement(t):
        return "defer"
    if is_deferral(t):
        return "delegate"
    # a steer may LEAD a longer answer ('You suggest. I'm good with any cloud.')
    first = re.split(r"(?<=[.!?])\s+", t, maxsplit=1)[0] if t else ""
    if first != t and (is_deferral(first) or _DISMISS_RE.search(first)):
        return "delegate" if is_deferral(first) else "skip"
    if _DISMISS_RE.search(t):
        return "skip"
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
    r"cloud (?:service|services|infrastructure|platform|provider)|"
    r"infrastructure (?:component|components|choice|choices)|"
    r"backend (?:service|services|infrastructure|stack)|free (?:resource|resources)|"
    r"ui (?:framework|toolkit|library)|(?:api|apis|data source|data sources|provider|providers) (?:exist|available)|verification (?:method|approach)|quality (?:verification|metric|metrics)|how (?:will|should) (?:model )?quality be (?:verified|measured|tested))\b",
    re.IGNORECASE)
# Constraint/affinity questions are product-level and legitimate (the
# constraints area asks them openly; 'agnostic' closes it too).
_ENGINEERING_METHOD_RE = re.compile(
    r"\bhow (?:will|should) (?:the )?(?:model )?quality be (?:verified|measured|tested)\b|"
    r"\b(?:which|what) (?:quality )?(?:metric|metrics|verification method|verification approach)\b|"
    r"\b(?:specific |exact |maximum |minimum |target )*(?:latency|response time|throughput|requests? per second|rps|availability|uptime)\b[^?]*(?:seconds?|milliseconds?|percent|%|number|target|maximum|minimum|budget|sla)\b|"
    r"\b(?:how many|what number of) (?:concurrent )?(?:users|requests|transactions)\b",
    re.IGNORECASE)


def engineering_default(question: str) -> tuple[str, str] | None:
    """Classify engineering-owned probes and return a visible provisional default.

    The founder may override these later; the interview asks only for business
    impact, while architecture/specification owns the numeric translation.
    """
    q = question or ""
    if not implementation_question(q):
        return None
    if re.search(r"latency|response time|milliseconds?|seconds?", q, re.I):
        return ("interactive response-time target",
                "routine user interactions respond within 2 seconds; slower operations show immediate progress")
    if re.search(r"throughput|requests? per second|rps|concurrent|how many", q, re.I):
        return ("capacity target", "size capacity from observed peak demand with tested headroom")
    if re.search(r"availability|uptime|sla", q, re.I):
        return ("availability target", "use the standard service tier appropriate to business impact and recovery needs")
    if re.search(r"verif|metric|quality", q, re.I):
        return ("verification method", "derive measurable acceptance thresholds from the approved workflows and risks")
    return (question_topic(q) or "engineering method", "use the relevant industry-standard default")

_IMPL_ALLOW_RE = re.compile(
    r"constraint|affinity|mandat|must (?:use|be)|existing|already|locked", re.IGNORECASE)


def implementation_question(question: str) -> bool:
    """True when the question asks the founder to CHOOSE an implementation."""
    q = question or ""
    return bool(_ENGINEERING_METHOD_RE.search(q) or _IMPL_QUESTION_RE.search(q)) and not _IMPL_ALLOW_RE.search(q)


def question_topic(question: str) -> str | None:
    """Return a code-owned sub-topic label for repeat-probe prevention."""
    for label, pattern in _TOPIC_PATTERNS:
        if pattern.search(question or ""):
            return label
    return None


_FALLBACK_QUESTIONS = {
    "goal": "What change should people see in their work or lives if this product succeeds?",
    "users": "Who will use, support, or be affected by this product, and in what situation?",
    "flows": "Please walk through one important real-world use from the moment it starts to a successful result.",
    "data": "What information does that use need, where does it come from, and who is responsible for it?",
    "integrations": "Which existing tools or services must this product exchange information with?",
    "nfrs": "In that important use, what kind of slowness, outage, or busy-period failure would make the product unacceptable?",
    "failures": "If something goes wrong during that use, what must be preserved or recoverable for the user?",
    "constraints": "What real deadline, budget, staffing, platform, legal, or policy limit must the solution respect?",
    "risks": "Which assumption about the users, problem, or operating environment are you least certain about?",
    "done": "What would you need to observe before calling the first useful version ready?",
    "glossary": "Are there any business or domain terms here that different people might interpret differently?",
}

_LAST_RESORT_QUESTION = "What important product fact have we not captured yet?"



def safe_question(question: str, parked_topics: list[str], target: str | None) -> str:
    """Mechanically prevent a model from re-probing a parked sub-topic."""
    topic = question_topic(question)
    if topic and topic in parked_topics:
        return _FALLBACK_QUESTIONS.get(target or "", _LAST_RESORT_QUESTION)
    return question


def guard_question(question: str, parked_topics: list[str], target: str | None) -> str:
    """safe_question plus the implementation-choice guard: a technology-choice
    question is replaced by the product-level fallback for the target area
    (the behavior need stays with the interview; the tech choice moves to the
    spec stage)."""
    if engineering_default(question):
        return _FALLBACK_QUESTIONS.get(target or "", _LAST_RESORT_QUESTION)
    return safe_question(question, parked_topics, target)


def question_substance(question: str) -> frozenset[str]:
    """Stable content words for deterministic same-substance detection."""
    stop = {"what", "which", "the", "a", "an", "do", "does", "did", "is", "are", "will", "would",
            "could", "should", "you", "your", "we", "our", "it", "this", "that", "specific", "please",
            "tell", "me", "about", "for", "to", "of", "in", "on", "and", "or", "between", "from"}
    words = re.findall(r"[a-z0-9]+", (question or "").lower())
    stems = {"sources":"source", "datasets":"dataset", "providers":"provider", "predictive":"prediction",
             "models":"model", "training":"train", "trained":"train", "used":"use", "using":"use",
             "makes":"make", "made":"make", "worthwhile":"worth"}
    return frozenset(stems.get(w, w) for w in words if w not in stop)


def same_question_substance(left: str, right: str) -> bool:
    a, b = question_substance(left), question_substance(right)
    if not a or not b:
        return (left or "").strip().lower() == (right or "").strip().lower()
    return len(a & b) / len(a | b) >= .45


def unique_question(question: str, prior_questions: list[dict], target: str | None) -> str:
    """Never expose a repeated probe. Reframe once to the actual current gap."""
    prior = [str(x.get("question", "")) for x in prior_questions]
    if not any(same_question_substance(question, old) for old in prior):
        return question
    fallback = _FALLBACK_QUESTIONS.get(target or "", _LAST_RESORT_QUESTION)
    if not any(same_question_substance(fallback, old) for old in prior):
        return fallback
    # A deterministic, target-specific final reframe. The graph parks the area
    # after two attempts, so this is only a compatibility safety net.
    return f"What else should we understand about {_LABELS.get(target or '', 'this part of the product')} before moving on?"

def founder_control(text: str) -> str | None:
    """'skip' -> skip the current target area; 'enough' -> end the interview.
    Deliberately strict: a sentence merely CONTAINING 'skip' is not a command."""
    t = text.strip()
    if _SKIP_RE.match(t):
        return "skip"
    if _ENOUGH_RE.match(t):
        return "enough"
    return None



def _valid_turn_object(obj) -> bool:
    """Require the complete interview-turn envelope before coverage can change."""
    if not isinstance(obj, dict) or set(obj) != {"question", "covered", "thin", "skipped", "complete"}:
        return False
    if not isinstance(obj["question"], str) or not isinstance(obj["complete"], bool):
        return False
    # The JSON example is a schema illustration, never a valid founder-facing
    # question. Some models copy it verbatim; reject that as a contract failure.
    if obj["question"].strip().casefold() == "one short concrete question":
        return False
    for key in ("covered", "thin", "skipped"):
        if not isinstance(obj[key], list) or any(x not in AREA_KEYS for x in obj[key]):
            return False
    return True

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
        if _valid_turn_object(obj):
            return obj
    start = text.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if _valid_turn_object(obj):
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

