"""Single source of truth for the founder-intent taxonomy.

Every place that enumerates intent labels reads them from here: the
categorization request (``categorization.INTENT_LABELS``), the rubric
variants (``categorization_rubrics``), the replay scorer
(``scripts/score-rubric-variants.py``), and labeling tools, which take
their options from ``python -m interview_engine.intent_taxonomy`` (JSON).
Add or rename a label here and nowhere else.

``defer`` (2026-09-23, Shy): the founder explicitly postpones the question
("I have to check with someone and come back to you"). It is not delegate
(nobody was handed the choice) and not settled, so the interview must not
treat the question as decided.
"""
from __future__ import annotations

import json

# Canonical order: the order labels appear in prompts, tables and tools.
INTENT_ORDER: tuple[str, ...] = ("answer", "delegate", "defer", "dismiss", "skip", "enough")

# One-line criteria: the canonical label text shown to classifiers and
# labelers. v1.2 (2026-09-23, Shy's rewrite): each label is phrased as what
# the founder WANTS, so the classes do not overlap. Note the v1.2 meanings of
# dismiss and skip: dismiss = does not want to tackle the topic at all;
# skip = wants to move on (v1/v1.1 had "move on" under dismiss).
INTENT_SHORT: dict[str, str] = {
    "answer": "Founder gives product information, even if it doesn't directly address the question asked",
    "delegate": "Founder wants the interviewer to decide",
    "defer": "Founder wants to answer later",
    "dismiss": "Founder doesn't want to tackle this topic",
    "skip": "Founder wants to move on",
    "enough": "Founder wants to end the whole interview",
}

# One-sentence boundaries, consistent with INTENT_SHORT.
INTENT_DEFS: dict[str, str] = {
    "answer": ("The founder gives product information: facts, rules, preferences or examples, even if it does not "
               "directly address the question asked. Off-topic or pasted product text is still answer; whether it "
               "addresses the question is judged separately (relevance). This is the default when the founder expresses no control wish."),
    "delegate": "The founder wants the interviewer/builder to make this decision for them.",
    "defer": ("The founder wants to answer this later, for example after checking with someone on their side. "
              "Nothing is decided and nobody was handed the choice."),
    "dismiss": ("The founder does not want to tackle this topic at all: they decline it, without handing the choice to anyone "
                "and without promising to answer later."),
    "skip": "The founder wants to move on from the current question to the next one; the interview continues.",
    "enough": "The founder wants to end the whole interview now.",
}

# Definitions + positive/negative examples. Examples are paraphrased and must
# not copy inputs from the labeled 133-case set or the rare-class supplement.
INTENT_EXAMPLES: dict[str, str] = {
    "answer": INTENT_DEFS["answer"] + (
        " Examples: 'Admins approve refunds over $50.' / 'Probably weekly, but I haven't decided.' "
        "/ 'The system should assign it automatically, round-robin.' (a product rule about the software deciding, not a delegation) "
        "/ a general product description that ignores the question (still answer). "
        "Not answer: a reply whose main move is one of the wishes below and that gives no product information."),
    "delegate": INTENT_DEFS["delegate"] + (
        " Examples: 'Your call.' / 'Pick whatever you think works.' / 'Use the usual default here.' / "
        "'Do it the way most booking tools do it.' (hands the detail to the builder's judgment). "
        "Not delegate: a rule where the PRODUCT decides (that is answer); the founder or someone on their side will answer later (that is defer)."),
    "defer": INTENT_DEFS["defer"] + (
        " Examples: 'Let me check with my cofounder and come back to you on that.' / "
        "'I need to confirm this with our accountant first.' / 'I'll find out and get back to you.' "
        "Not defer: 'you choose' (that is delegate); 'I'd rather not get into that' (that is dismiss); a tentative answer with content (that is answer)."),
    "dismiss": INTENT_DEFS["dismiss"] + (
        " Examples: 'I'd rather not get into that.' / 'That topic isn't relevant for us.' / 'I don't want to discuss pricing here.' "
        "Not dismiss: 'you choose' (that is delegate); 'next question' (that is skip); 'I'll come back to you on that' (that is defer)."),
    "skip": INTENT_DEFS["skip"] + (
        " Examples: 'Let's move on.' / 'Next question.' / 'I've said all I have on that.' "
        "Not skip: declining the topic itself (that is dismiss); ending the whole interview (that is enough)."),
    "enough": INTENT_DEFS["enough"] + (
        " Examples: 'That's enough, write the spec.' / 'I'm done with the interview.' "
        "Not enough: moving on from only the current question (that is skip)."),
}

# Labels that close the current area / interview.
TERMINAL_INTENTS: frozenset[str] = frozenset({"dismiss", "skip", "enough"})
# Control moves a classifier must assert with high confidence before code acts
# on them; a low-confidence one falls back to answer.
GATED_INTENTS: frozenset[str] = TERMINAL_INTENTS | {"defer"}
# Labels that count as correct for each other in scoring (locked 2026-09-23):
# delegate is an autonomy grant, not a non-answer. defer is compatible with
# neither.
COMPATIBLE_GROUPS: tuple[frozenset[str], ...] = (frozenset({"answer", "delegate"}),)

assert tuple(INTENT_SHORT) == tuple(INTENT_DEFS) == tuple(INTENT_EXAMPLES) == INTENT_ORDER


def compatible(pred: str, labels) -> bool:
    """True when ``pred`` is in ``labels`` or shares a compatibility group with one."""
    labels = [str(l) for l in labels]
    if pred in labels:
        return True
    return any(pred in g and any(l in g for l in labels) for g in COMPATIBLE_GROUPS)


# Rubric version of the canonical criteria above. Bump with any text change.
TAXONOMY_VERSION = "categorization-request-v1.2"
# Labelers may pick several intent labels for one reply (locked 2026-09-23).
MULTI_SELECT = True


def taxonomy_json() -> str:
    """Machine-readable export for tools that cannot import Python."""
    return json.dumps({"version": TAXONOMY_VERSION, "slot": "intent", "key": "intent", "multi_select": MULTI_SELECT,
                       "order": list(INTENT_ORDER), "short": INTENT_SHORT, "definitions": INTENT_DEFS,
                       "compatible_groups": [sorted(g) for g in COMPATIBLE_GROUPS],
                       "terminal": sorted(TERMINAL_INTENTS)}, indent=2)


if __name__ == "__main__":  # python -m interview_engine.intent_taxonomy
    print(taxonomy_json())
