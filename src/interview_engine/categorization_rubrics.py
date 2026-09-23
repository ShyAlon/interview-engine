"""Versioned rubric variants for the intent and coverage-relevance slots.

Each variant is a complete, code-owned rubric: the questions a provider sees
plus an optional ``finalize`` step that turns decomposed yes/no signals back
into the slot's canonical label. The variant id is recorded as the request's
``rubric_version`` so every shadow/replay record says exactly which rubric
produced it.

Production default stays ``*-v1`` (the rubric Shy's 2026-09-23 labels were
collected against) until a replay bake-off picks a winner. Select another
variant with ``INTERVIEW_ENGINE_RUBRIC_INTENT`` / ``INTERVIEW_ENGINE_RUBRIC_RELEVANCE``.

Held-out note: examples below are deliberately paraphrased. They must not copy
inputs from the labeled 133-case set or the rare-class supplement, or the
bake-off would score the rubric on its own examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .categorization import ChoiceQuestion, ClassificationAnswer, NoulQuestion, Question
from .intent_taxonomy import INTENT_DEFS, INTENT_EXAMPLES, INTENT_ORDER, INTENT_SHORT, TAXONOMY_VERSION

V1_RUBRIC_VERSION = "categorization-request-v1"
# intent-v1 plus the defer label (2026-09-23). The pre-defer text is kept
# verbatim as intent-v1-133 so replays stay comparable with the 133 labels.
V1_DEFER_RUBRIC_VERSION = "categorization-request-v1.1"
# v1.2: Shy's "what the founder wants" criteria, from intent_taxonomy.
V1_2_RUBRIC_VERSION = TAXONOMY_VERSION
INTENT_KEY = "intent"
RELEVANCE_KEY = "relevance"
INTENT_CANONICAL = INTENT_ORDER  # from intent_taxonomy, the single source of truth
RELEVANCE_CANONICAL = ("addressed", "partial", "dodged")

Finalize = Callable[[Mapping[str, ClassificationAnswer]], dict[str, ClassificationAnswer]]


@dataclass(frozen=True)
class RubricVariant:
    id: str
    slot_key: str               # "intent" or "relevance"
    description: str
    questions: Mapping[str, Question]
    finalize: Finalize | None = None
    production_ok: bool = True  # decomposed variants need finalize wiring in the chain first
    version: str | None = None  # explicit rubric_version; default derives from the id

    @property
    def rubric_version(self) -> str:
        if self.version: return self.version
        return V1_RUBRIC_VERSION if self.id.endswith("-v1") else f"categorization-request-v2/{self.id}"

    def apply(self, answers: Mapping[str, ClassificationAnswer]) -> dict[str, ClassificationAnswer]:
        return dict(self.finalize(answers)) if self.finalize else dict(answers)


# ------------------------------------------------------------------ intent
# Frozen: the exact 5-label rubric the 2026-09-23 labels were collected against.
_INTENT_V1_133 = {
    "answer": "ordinary product answer", "delegate": "founder delegates the current detail",
    "dismiss": "founder closes the current topic/area", "skip": "founder explicitly skips the current area",
    "enough": "founder explicitly ends the whole interview",
}
# Frozen: v1.1 = the labeled v1 text plus defer (commit 710d6c4).
_INTENT_V1_1 = {
    "answer": "ordinary product answer", "delegate": "founder delegates the current detail",
    "defer": "founder postpones the current question to answer later (not decided, not delegated)",
    "dismiss": "founder closes the current topic/area", "skip": "founder explicitly skips the current area",
    "enough": "founder explicitly ends the whole interview",
}
_INTENT_V1 = dict(INTENT_SHORT)  # v1.2, canonical
_INTENT_DEFS = dict(INTENT_DEFS)
_INTENT_EXAMPLES = dict(INTENT_EXAMPLES)

_OFFTOPIC_RULE = (
    " Rule for off-topic or pasted text: if the reply does not engage with the question (generic or canned text, "
    "text about a different topic, or a request unrelated to the question) and contains no explicit control phrase, "
    "label it answer. Relevance is judged separately; do not turn a non-responsive reply into delegate, defer, dismiss or skip.")

_INTENT_V2A_INSTR = ("Classify what the founder wants with this one reply. Choose delegate, defer, dismiss, skip or enough "
                     "ONLY when the reply expresses that wish explicitly; otherwise choose answer.")
_INTENT_V2B_INSTR = (_INTENT_V2A_INSTR + " delegate = the interviewer decides; defer = the founder answers later; dismiss = the founder won't tackle this topic; skip = move on to the next question; "
                     "enough = end the whole interview. A rule saying the product/app decides something is answer.")


def _intent_choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Question]:
    return {INTENT_KEY: ChoiceQuestion(instructions, dict(criteria))}


_DECOMP_INTENT = {
    "intent_ends_interview": "The founder explicitly asks to end the whole interview now.",
    "intent_closes_topic": "The founder wants to move on from the current question to the next one (the interview continues).",
    "intent_declines": "The founder does not want to tackle this topic at all (for example 'rather not get into that').",
    "intent_hands_choice": ("The founder explicitly hands this decision to the interviewer or builder to choose for them "
                            "(for example 'your call', 'pick whatever is standard'). A rule saying the product or app decides is NOT this."),
    "intent_postpones": ("The founder explicitly postpones this question to answer it later, for example after checking "
                         "with someone on their side. Nobody is handed the choice."),
    "intent_gives_content": "The reply states product facts, rules, preferences or examples for the question.",
}
DECOMPOSED_THRESHOLD = .5


def _value(answers, key) -> float:
    a = answers.get(key)
    try: return float(a.value) if a is not None else 0.0
    except (TypeError, ValueError): return 0.0


def finalize_decomposed_intent(answers: Mapping[str, ClassificationAnswer]) -> dict[str, ClassificationAnswer]:
    """Strict precedence over explicit-move signals. Content alone is answer."""
    s = {k: _value(answers, k) for k in _DECOMP_INTENT}
    t = DECOMPOSED_THRESHOLD
    content = s["intent_gives_content"]
    if s["intent_ends_interview"] >= t: label, conf = "enough", s["intent_ends_interview"]
    # v1.2 meanings: declining the topic = dismiss; moving on = skip.
    elif s["intent_declines"] >= t and s["intent_declines"] > content: label, conf = "dismiss", s["intent_declines"]
    elif s["intent_closes_topic"] >= t and s["intent_closes_topic"] > content: label, conf = "skip", s["intent_closes_topic"]
    elif s["intent_postpones"] >= t and s["intent_postpones"] > content: label, conf = "defer", s["intent_postpones"]
    elif s["intent_hands_choice"] >= t: label, conf = "delegate", s["intent_hands_choice"]
    else:
        label = "answer"
        conf = 1 - max(s["intent_ends_interview"], s["intent_declines"], s["intent_closes_topic"], s["intent_postpones"], s["intent_hands_choice"])
    out = dict(answers)
    out[INTENT_KEY] = ClassificationAnswer(label, round(conf, 4), {k: v for k, v in s.items()})
    return out


INTENT_VARIANTS: dict[str, RubricVariant] = {v.id: v for v in (
    RubricVariant("intent-v1", INTENT_KEY, "Production default: v1.2, Shy's 'what the founder wants' criteria from intent_taxonomy.",
                  _intent_choice("Classify the founder's control intent", _INTENT_V1), version=V1_2_RUBRIC_VERSION),
    RubricVariant("intent-v1.1", INTENT_KEY, "Frozen: the labeled v1 text plus defer (710d6c4).",
                  _intent_choice("Classify the founder's control intent", _INTENT_V1_1), version=V1_DEFER_RUBRIC_VERSION),
    RubricVariant("intent-v1-133", INTENT_KEY, "Frozen control: the exact 5-label rubric the 133-case labels were collected against (no defer).",
                  _intent_choice("Classify the founder's control intent", _INTENT_V1_133), version=V1_RUBRIC_VERSION),
    RubricVariant("intent-v2a-defs", INTENT_KEY, "Minimal sharpening: one-sentence boundaries, answer is the default.",
                  _intent_choice(_INTENT_V2A_INSTR, _INTENT_DEFS)),
    RubricVariant("intent-v2b-examples", INTENT_KEY, "Taxonomy + positive/negative examples per label.",
                  _intent_choice(_INTENT_V2B_INSTR, _INTENT_EXAMPLES)),
    RubricVariant("intent-v2c-offtopic", INTENT_KEY, "Taxonomy + examples + the doc's off-topic rule (off-topic = answer; relevance judges it).",
                  _intent_choice(_INTENT_V2B_INSTR + _OFFTOPIC_RULE,
                                 {**_INTENT_EXAMPLES, "answer": _INTENT_EXAMPLES["answer"] + " Also answer: off-topic or pasted text with no control phrase."})),
    RubricVariant("intent-v2d-decomposed", INTENT_KEY, "Strict boundary: one yes/no signal per control move, code-owned precedence.",
                  {k: NoulQuestion(v) for k, v in _DECOMP_INTENT.items()}, finalize_decomposed_intent, production_ok=False),
)}


# --------------------------------------------------------------- relevance
_REL_V1 = {"addressed": "directly and substantively answers it", "partial": "answers only part or remains vague",
           "dodged": "does not answer, changes subject, or refuses without delegating"}

_REL_ANCHORED = {
    "addressed": ("Answers what the question actually asked, with at least one concrete, question-specific fact "
                  "(a who, what, number, rule or example). Every main part of the question gets an answer."),
    "partial": ("Engages with the question's subject but leaves a main part unanswered, or stays generic/hedged "
                "without a concrete question-specific fact."),
    "dodged": ("Does not engage with what was asked: a different topic, canned or generic text that would fit any question, "
               "a request unrelated to the question, or a refusal. Topical words alone do not make it partial."),
}
_REL_STRICT = {
    "addressed": _REL_ANCHORED["addressed"] + " Example: Q 'Who approves refunds?' A 'Store managers, up to $200.'",
    "partial": _REL_ANCHORED["partial"] + (" Example: Q 'Who approves refunds and how fast?' A 'Managers do.' "
                                           "(speed missing). Q 'What happens when sync fails?' A 'It should be reliable.' (generic)."),
    "dodged": _REL_ANCHORED["dodged"] + (" Example: Q 'What happens when sync fails?' A 'Our goal is to help families stay organised.' "
                                         "(true, but not about sync failure)."),
}
_REL_V2A_INSTR = "Did the answer address the question that was asked? Judge against the question, not against the product in general."
_REL_V2B_INSTR = (_REL_V2A_INSTR + " Be strict: when unsure between addressed and partial, choose partial; when the answer "
                  "does not mention anything the question asked about, choose dodged even if it contains other product facts.")

_DECOMP_REL = {
    "rel_on_question": "The answer talks about the specific thing this question asked, not just the product in general.",
    "rel_concrete": "The answer gives at least one concrete fact specific to this question (who, what, number, rule or example).",
    "rel_all_parts": "Every main part of the question is answered.",
    "rel_generic": "The answer is generic or canned text that would fit under almost any question.",
}


def finalize_decomposed_relevance(answers: Mapping[str, ClassificationAnswer]) -> dict[str, ClassificationAnswer]:
    s = {k: _value(answers, k) for k in _DECOMP_REL}
    t = DECOMPOSED_THRESHOLD
    if s["rel_on_question"] < t or s["rel_generic"] >= t:
        label, conf = "dodged", max(1 - s["rel_on_question"], s["rel_generic"])
    elif s["rel_concrete"] >= t and s["rel_all_parts"] >= t:
        label, conf = "addressed", min(s["rel_concrete"], s["rel_all_parts"])
    else:
        label, conf = "partial", s["rel_on_question"]
    out = dict(answers)
    out[RELEVANCE_KEY] = ClassificationAnswer(label, round(conf, 4), dict(s))
    return out


RELEVANCE_VARIANTS: dict[str, RubricVariant] = {v.id: v for v in (
    RubricVariant("relevance-v1", RELEVANCE_KEY, "Control: the rubric the 133-case labels were collected against.",
                  {RELEVANCE_KEY: ChoiceQuestion("Did the answer address the question?", _REL_V1)}),
    RubricVariant("relevance-v2a-anchored", RELEVANCE_KEY, "Anchored definitions (concrete question-specific fact required for addressed).",
                  {RELEVANCE_KEY: ChoiceQuestion(_REL_V2A_INSTR, _REL_ANCHORED)}),
    RubricVariant("relevance-v2b-strict", RELEVANCE_KEY, "Anchored + examples + tie-break toward the stricter label + off-topic = dodged.",
                  {RELEVANCE_KEY: ChoiceQuestion(_REL_V2B_INSTR, _REL_STRICT)}),
    RubricVariant("relevance-v2c-decomposed", RELEVANCE_KEY, "Four yes/no signals, code-owned mapping to addressed/partial/dodged.",
                  {k: NoulQuestion(v) for k, v in _DECOMP_REL.items()}, finalize_decomposed_relevance, production_ok=False),
)}


def get_variant(variant_id: str) -> RubricVariant:
    table = {**INTENT_VARIANTS, **RELEVANCE_VARIANTS}
    try: return table[variant_id]
    except KeyError: raise ValueError(f"unknown rubric variant {variant_id!r}; known: {', '.join(table)}") from None


def configured_variant(slot_key: str) -> RubricVariant:
    """Production selection. Unset means v1; decomposed variants are replay-only for now."""
    from .model_config import _env
    env = "INTERVIEW_ENGINE_RUBRIC_" + slot_key.upper()
    raw = (_env(env) or f"{slot_key}-v1").strip()
    variant = get_variant(raw)
    if variant.slot_key != slot_key:
        raise ValueError(f"{env}={raw!r} is a {variant.slot_key} variant")
    if not variant.production_ok:
        raise ValueError(f"{env}={raw!r} is replay-only (needs finalize wiring in the provider chain)")
    return variant
