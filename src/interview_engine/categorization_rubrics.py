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

V1_RUBRIC_VERSION = "categorization-request-v1"
INTENT_KEY = "intent"
RELEVANCE_KEY = "relevance"
INTENT_CANONICAL = ("answer", "delegate", "dismiss", "skip", "enough")
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

    @property
    def rubric_version(self) -> str:
        return V1_RUBRIC_VERSION if self.id.endswith("-v1") else f"categorization-request-v2/{self.id}"

    def apply(self, answers: Mapping[str, ClassificationAnswer]) -> dict[str, ClassificationAnswer]:
        return dict(self.finalize(answers)) if self.finalize else dict(answers)


# ------------------------------------------------------------------ intent
_INTENT_V1 = {
    "answer": "ordinary product answer", "delegate": "founder delegates the current detail",
    "dismiss": "founder closes the current topic/area", "skip": "founder explicitly skips the current area",
    "enough": "founder explicitly ends the whole interview",
}

_INTENT_DEFS = {
    "answer": ("The founder gives product content for the question: facts, rules, preferences, examples, "
               "or an honest 'not sure yet' with some content. This is the default when no explicit control move is present."),
    "delegate": ("The founder explicitly hands THIS decision to the interviewer/builder to choose on their behalf, "
                 "instead of choosing themselves."),
    "dismiss": ("The founder explicitly asks to stop discussing the current topic and move on, while the interview continues."),
    "skip": ("The founder explicitly declines to answer this question or area at all, without handing the choice to anyone."),
    "enough": ("The founder explicitly asks to end the whole interview now."),
}

_INTENT_EXAMPLES = {
    "answer": _INTENT_DEFS["answer"] + (
        " Examples: 'Admins approve refunds over $50.' / 'Probably weekly, but I haven't decided.' "
        "/ 'The system should assign it automatically, round-robin.' (a product rule about the software deciding, not a delegation). "
        "Not answer: any reply whose main move is one of the control phrases below."),
    "delegate": _INTENT_DEFS["delegate"] + (
        " Examples: 'Your call.' / 'Pick whatever you think works.' / 'Use the usual default here.' / "
        "'Do it the way most booking tools do it.' (hands the detail to the builder's judgment). "
        "Not delegate: the founder states a rule where the PRODUCT decides; the founder says someone else on their side will answer later."),
    "dismiss": _INTENT_DEFS["dismiss"] + (
        " Examples: 'Let's move on.' / 'That's covered, next topic.' / 'I've said all I have on that.' "
        "Not dismiss: a short answer that still gives content."),
    "skip": _INTENT_DEFS["skip"] + (
        " Examples: 'Skip this one.' / 'I'd rather not get into that.' / 'Not relevant for us, skip it.' "
        "Not skip: 'you choose' (that is delegate); 'next topic' after answering (that is dismiss)."),
    "enough": _INTENT_DEFS["enough"] + (
        " Examples: 'That's enough, write the spec.' / 'I'm done with the interview.' "
        "Not enough: closing only the current topic (that is dismiss)."),
}

_OFFTOPIC_RULE = (
    " Rule for off-topic or pasted text: if the reply does not engage with the question (generic or canned text, "
    "text about a different topic, or a request unrelated to the question) and contains no explicit control phrase, "
    "label it answer. Relevance is judged separately; do not turn a non-responsive reply into delegate, dismiss or skip.")

_INTENT_V2A_INSTR = ("Classify the founder's control intent for this one reply. Choose delegate, dismiss, skip or enough "
                     "ONLY when the reply contains an explicit control move; otherwise choose answer.")
_INTENT_V2B_INSTR = (_INTENT_V2A_INSTR + " delegate = the builder chooses; skip = nobody answers; dismiss = stop this topic; "
                     "enough = stop the whole interview. A rule saying the product/app decides something is answer.")


def _intent_choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Question]:
    return {INTENT_KEY: ChoiceQuestion(instructions, dict(criteria))}


_DECOMP_INTENT = {
    "intent_ends_interview": "The founder explicitly asks to end the whole interview now.",
    "intent_closes_topic": "The founder explicitly asks to stop discussing the current topic and move on (the interview continues).",
    "intent_declines": "The founder explicitly declines to answer this question at all (for example 'skip this', 'rather not say').",
    "intent_hands_choice": ("The founder explicitly hands this decision to the interviewer or builder to choose for them "
                            "(for example 'your call', 'pick whatever is standard'). A rule saying the product or app decides is NOT this."),
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
    elif s["intent_declines"] >= t and s["intent_declines"] > content: label, conf = "skip", s["intent_declines"]
    elif s["intent_closes_topic"] >= t and s["intent_closes_topic"] > content: label, conf = "dismiss", s["intent_closes_topic"]
    elif s["intent_hands_choice"] >= t: label, conf = "delegate", s["intent_hands_choice"]
    else:
        label = "answer"
        conf = 1 - max(s["intent_ends_interview"], s["intent_declines"], s["intent_closes_topic"], s["intent_hands_choice"])
    out = dict(answers)
    out[INTENT_KEY] = ClassificationAnswer(label, round(conf, 4), {k: v for k, v in s.items()})
    return out


INTENT_VARIANTS: dict[str, RubricVariant] = {v.id: v for v in (
    RubricVariant("intent-v1", INTENT_KEY, "Control: the rubric the 133-case labels were collected against.",
                  _intent_choice("Classify the founder's control intent", _INTENT_V1)),
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
