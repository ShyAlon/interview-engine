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
# v1.2: Shy's "what the founder wants" criteria (6 labels incl. dismiss), frozen.
V1_2_RUBRIC_VERSION = "categorization-request-v1.2"
# v1.3: dismiss folded into skip (2026-09-24), canonical, from intent_taxonomy.
V1_3_RUBRIC_VERSION = TAXONOMY_VERSION
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
    # Relevance variants only. None keeps the legacy covers_* template (v1..v2).
    covers: Mapping[str, str] | None = None          # area -> statement WITH the question clause
    covers_plain: Mapping[str, str] | None = None    # area -> statement without it (option B, non-target areas)
    covers_clause_all: bool = True                   # A: clause on every area; B: target area only
    preamble: str | None = None                      # prepended to the request state

    @property
    def rubric_version(self) -> str:
        if self.version: return self.version
        return V1_RUBRIC_VERSION if self.id.endswith("-v1") else f"categorization-request-v2/{self.id}"

    def covers_questions(self, target: str | None) -> dict[str, "Question"] | None:
        """The 11 covers_* questions this variant asks, or None for the legacy template."""
        if self.covers is None: return None
        out = {}
        for area, text in self.covers.items():
            if not self.covers_clause_all and area != (target or "") and self.covers_plain:
                text = self.covers_plain[area]
            out[f"covers_{area}"] = NoulQuestion(text)
        return out

    def state(self, state: str) -> str:
        return f"{self.preamble}\n\n{state}" if self.preamble else state

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
# Frozen: v1.2 = Shy's 2026-09-23 rewrite, the last text with dismiss (2e03444).
_INTENT_V1_2 = {
    "answer": "Founder gives product information, even if it doesn't directly address the question asked",
    "delegate": "Founder wants the interviewer to decide", "defer": "Founder wants to answer later",
    "dismiss": "Founder doesn't want to tackle this topic", "skip": "Founder wants to move on",
    "enough": "Founder wants to end the whole interview",
}
_INTENT_V1 = dict(INTENT_SHORT)  # v1.3, canonical
_INTENT_DEFS = dict(INTENT_DEFS)
# v1.4 (r2 markup): v1.3 criteria unchanged, enough/skip gain examples.
# Paraphrased - not copied from the benchmark pool.
_INTENT_V1_4 = {**_INTENT_V1,
    "enough": _INTENT_V1["enough"] + (". Examples: 'That's enough, write the spec.' / 'I'm done with the interview.' / "
                                      "'Stop the questions, just put it together.' Not enough: 'That's everything I have on this one.' "
                                      "(done with this question only - skip) / 'we should finish soon' (not ending yet)."),
    "skip": _INTENT_V1["skip"] + (". Examples: 'Let's move on.' / 'Next question.' / 'I've said all I have on that.' / "
                                  "'I'd rather not get into that.' Not skip: 'no more questions at all' / 'write it up now' "
                                  "(ends the interview - enough)."),
}
_INTENT_EXAMPLES = dict(INTENT_EXAMPLES)

_OFFTOPIC_RULE = (
    " Rule for off-topic or pasted text: if the reply does not engage with the question (generic or canned text, "
    "text about a different topic, or a request unrelated to the question) and contains no explicit control phrase, "
    "label it answer. Relevance is judged separately; do not turn a non-responsive reply into delegate, defer or skip.")

_INTENT_V2A_INSTR = ("Classify what the founder wants with this one reply. Choose delegate, defer, skip or enough "
                     "ONLY when the reply expresses that wish explicitly; otherwise choose answer.")
_INTENT_V2B_INSTR = (_INTENT_V2A_INSTR + " delegate = the interviewer decides; defer = the founder answers later; skip = move on, or won't tackle this topic; "
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
    # v1.3: declining the topic and moving on are both skip.
    elif max(s["intent_declines"], s["intent_closes_topic"]) >= t and max(s["intent_declines"], s["intent_closes_topic"]) > content:
        label, conf = "skip", max(s["intent_declines"], s["intent_closes_topic"])
    elif s["intent_postpones"] >= t and s["intent_postpones"] > content: label, conf = "defer", s["intent_postpones"]
    elif s["intent_hands_choice"] >= t: label, conf = "delegate", s["intent_hands_choice"]
    else:
        label = "answer"
        conf = 1 - max(s["intent_ends_interview"], s["intent_declines"], s["intent_closes_topic"], s["intent_postpones"], s["intent_hands_choice"])
    out = dict(answers)
    out[INTENT_KEY] = ClassificationAnswer(label, round(conf, 4), {k: v for k, v in s.items()})
    return out


INTENT_VARIANTS: dict[str, RubricVariant] = {v.id: v for v in (
    RubricVariant("intent-v1", INTENT_KEY, "Production default: v1.3, Shy's 'what the founder wants' criteria with dismiss folded into skip.",
                  _intent_choice("Classify the founder's control intent", _INTENT_V1), version=V1_3_RUBRIC_VERSION),
    RubricVariant("intent-v1.4", INTENT_KEY, "Replay-only until Shy rules: v1.3 plus enough/skip examples (r2 markup, 2026-09-24) to separate 'no more questions at all' (enough) from 'done with this one' (skip).",
                  _intent_choice("Classify the founder's control intent", _INTENT_V1_4), version="categorization-request-v1.4",
                  production_ok=False),
    RubricVariant("intent-v1.2", INTENT_KEY, "Frozen: v1.2, the 6-label text with dismiss (2e03444).",
                  _intent_choice("Classify the founder's control intent", _INTENT_V1_2), version=V1_2_RUBRIC_VERSION),
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


# relevance-v3-se: the relevance makeover draft Shy +1'd on 2026-09-24, with the
# question clause on the targeted area only (option B). relevance-v3-se-a keeps
# the clause on every area (option A) so the replay can compare both. Systems-
# engineering preamble, relevance criteria without "substantively"/"without
# delegating", and one plain statement per coverage area replacing the old
# "supplies concrete product facts for coverage area: <token>" template.
REL_V3_PREAMBLE = ("In the context of a systems engineering task: a founder is being interviewed about a software system "
                   "they want built. The interviewer asked one question and the founder replied. Judge each statement "
                   "below against that question and that reply.")
_REL_V3_INSTR = "Does the answer address the question asked?"
_REL_V3 = {"addressed": "The answer addresses every main part of the question asked.",
           "partial": "The answer addresses the question asked, but leaves a main part open or stays vague.",
           "dodged": "The answer does not address the question asked (another topic, generic or pasted text, or a refusal)."}
_CLAUSE = " in response to the question asked"
COVERS_V3 = {
    "goal": "The answer addresses the goal of the system, and the problem it solves, in response to the question asked.",
    "users": "The answer addresses the users and other stakeholders of the system in response to the question asked.",
    "flows": "The answer addresses how the system is used - an operational scenario from trigger to outcome - in response to the question asked.",
    "data": "The answer addresses the data the system handles - what it is, where it comes from, who owns it - in response to the question asked.",
    "integrations": "The answer addresses the external systems or services the system connects to in response to the question asked.",
    "nfrs": ("The answer addresses the quality needs of the system - such as speed, reliability, scale, security or privacy "
             "as the founder experiences them - in response to the question asked."),
    "failures": "The answer addresses the failure modes of the system - what can break or degrade, and what must never happen - in response to the question asked.",
    "constraints": "The answer addresses the constraints on the system - budget, deadline, team or technology - in response to the question asked.",
    "risks": "The answer addresses the risks to the system - assumptions that would sink the product if wrong - in response to the question asked.",
    "done": "The answer addresses how the founder will know the system works - its done criteria - in response to the question asked.",
    "glossary": "The answer addresses the domain terms used for the system and what they mean in response to the question asked.",
}
COVERS_V3_PLAIN = {k: v.replace(_CLAUSE, "").replace(" ,", ",") for k, v in COVERS_V3.items()}
assert all(_CLAUSE in v for v in COVERS_V3.values())

# v3.1-se (Shy's r2 markup, 2026-09-24): only the relevance choice changes -
# preamble and covers_* stay as v3-se (option B). Fixes the bench-1 leak:
# concrete or vague answers to a DIFFERENT question were called partial.
# Partial now needs at least one part of THIS question answered.
_REL_V31_INSTR = ("Does the answer respond to the question asked? Judge it only against this question. If the reply "
                  "answers no part of this question, choose dodged - even if it is detailed and useful, about the same "
                  "product, or close to the topic.")
_REL_V31 = {"addressed": "The answer answers every main part of the question asked.",
            "partial": ("The answer answers at least one part of the question asked, and leaves another main part open "
                        "or answers it only vaguely."),
            "dodged": ("The answer answers no part of the question asked: it answers a different question (even with "
                       "concrete product detail, or on a nearby topic), is generic or pasted text that could follow any "
                       "question, or refuses.")}

# Taxonomy v2 covers_* statements (nfrs split; bench-3) - PREPARED, NOT
# REGISTERED. Needs coverage_request to take its area list from the variant
# before it can be selected; lands with coverage-v2 after bench-2.
_COVERS_V3_QUALITY_V2 = {
    "performance": "The answer addresses how fast the system must be - for whom and in which moments - in response to the question asked.",
    "reliability": "The answer addresses when and for how long the system may be unavailable, and what must never be lost, in response to the question asked.",
    "security": "The answer addresses who may see or change what in the system, which data is sensitive, and any rules to follow, in response to the question asked.",
    "scalability": "The answer addresses the volume the system must handle for now - users, transactions or records - in response to the question asked.",
}
COVERS_V3_TAX2 = {}
for _k, _v in COVERS_V3.items():
    COVERS_V3_TAX2.update(_COVERS_V3_QUALITY_V2 if _k == "nfrs" else {_k: _v})
assert all(_CLAUSE in v for v in COVERS_V3_TAX2.values())

RELEVANCE_VARIANTS: dict[str, RubricVariant] = {v.id: v for v in (
    RubricVariant("relevance-v1", RELEVANCE_KEY, "Control: the rubric the 133-case labels were collected against.",
                  {RELEVANCE_KEY: ChoiceQuestion("Did the answer address the question?", _REL_V1)}),
    RubricVariant("relevance-v2a-anchored", RELEVANCE_KEY, "Anchored definitions (concrete question-specific fact required for addressed).",
                  {RELEVANCE_KEY: ChoiceQuestion(_REL_V2A_INSTR, _REL_ANCHORED)}),
    RubricVariant("relevance-v2b-strict", RELEVANCE_KEY, "Anchored + examples + tie-break toward the stricter label + off-topic = dodged.",
                  {RELEVANCE_KEY: ChoiceQuestion(_REL_V2B_INSTR, _REL_STRICT)}),
    RubricVariant("relevance-v2c-decomposed", RELEVANCE_KEY, "Four yes/no signals, code-owned mapping to addressed/partial/dodged.",
                  {k: NoulQuestion(v) for k, v in _DECOMP_REL.items()}, finalize_decomposed_relevance, production_ok=False),
    RubricVariant("relevance-v3-se", RELEVANCE_KEY, "Relevance makeover (Shy +1, 2026-09-24): SE preamble + plain per-area covers_*; question clause on the targeted area only (option B).",
                  {RELEVANCE_KEY: ChoiceQuestion(_REL_V3_INSTR, _REL_V3)}, version="relevance-v3-se",
                  covers=COVERS_V3, covers_plain=COVERS_V3_PLAIN, covers_clause_all=False, preamble=REL_V3_PREAMBLE),
    RubricVariant("relevance-v3.1-se", RELEVANCE_KEY, "Replay-only until Shy rules: v3-se with the partial fix (r2 markup, 2026-09-24) - partial needs part of THIS question answered; a different question's answer is dodged.",
                  {RELEVANCE_KEY: ChoiceQuestion(_REL_V31_INSTR, _REL_V31)}, version="relevance-v3.1-se",
                  covers=COVERS_V3, covers_plain=COVERS_V3_PLAIN, covers_clause_all=False, preamble=REL_V3_PREAMBLE,
                  production_ok=False),
    RubricVariant("relevance-v3-se-a", RELEVANCE_KEY, "Replay-comparison only: v3-se with the question clause on every area (option A, as first drafted).",
                  {RELEVANCE_KEY: ChoiceQuestion(_REL_V3_INSTR, _REL_V3)}, version="relevance-v3-se-a",
                  covers=COVERS_V3, covers_plain=COVERS_V3_PLAIN, covers_clause_all=True, preamble=REL_V3_PREAMBLE),
)}


# Version-named aliases for the moving production ids, so "intent-v1.3" means
# the current canonical intent rubric (intent-v1) wherever a variant is named.
VARIANT_ALIASES: dict[str, str] = {"intent-v1.3": "intent-v1"}


def resolve_variant_id(variant_id: str) -> str:
    return VARIANT_ALIASES.get(variant_id.strip(), variant_id.strip())


def get_variant(variant_id: str) -> RubricVariant:
    table = {**INTENT_VARIANTS, **RELEVANCE_VARIANTS}
    variant_id = resolve_variant_id(variant_id)
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
