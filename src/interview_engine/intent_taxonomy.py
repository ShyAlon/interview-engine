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

# One-line criteria (the v1-style rubric text).
INTENT_SHORT: dict[str, str] = {
    "answer": "ordinary product answer",
    "delegate": "founder delegates the current detail",
    "defer": "founder postpones the current question to answer later (not decided, not delegated)",
    "dismiss": "founder closes the current topic/area",
    "skip": "founder explicitly skips the current area",
    "enough": "founder explicitly ends the whole interview",
}

# One-sentence boundaries.
INTENT_DEFS: dict[str, str] = {
    "answer": ("The founder gives product content for the question: facts, rules, preferences, examples, "
               "or an honest 'not sure yet' with some content. This is the default when no explicit control move is present."),
    "delegate": ("The founder explicitly hands THIS decision to the interviewer/builder to choose on their behalf, "
                 "instead of choosing themselves."),
    "defer": ("The founder explicitly postpones THIS question: they will answer it later, after checking with someone "
              "or finding out. Nobody was handed the choice and nothing was decided."),
    "dismiss": ("The founder explicitly asks to stop discussing the current topic and move on, while the interview continues."),
    "skip": ("The founder explicitly declines to answer this question or area at all, without handing the choice to anyone."),
    "enough": ("The founder explicitly asks to end the whole interview now."),
}

# Definitions + positive/negative examples. Examples are paraphrased and must
# not copy inputs from the labeled 133-case set or the rare-class supplement.
INTENT_EXAMPLES: dict[str, str] = {
    "answer": INTENT_DEFS["answer"] + (
        " Examples: 'Admins approve refunds over $50.' / 'Probably weekly, but I haven't decided.' "
        "/ 'The system should assign it automatically, round-robin.' (a product rule about the software deciding, not a delegation). "
        "Not answer: any reply whose main move is one of the control phrases below."),
    "delegate": INTENT_DEFS["delegate"] + (
        " Examples: 'Your call.' / 'Pick whatever you think works.' / 'Use the usual default here.' / "
        "'Do it the way most booking tools do it.' (hands the detail to the builder's judgment). "
        "Not delegate: the founder states a rule where the PRODUCT decides; the founder says someone else on their side will answer later (that is defer)."),
    "defer": INTENT_DEFS["defer"] + (
        " Examples: 'Let me check with my cofounder and come back to you on that.' / "
        "'I need to confirm this with our accountant first.' / 'Not sure yet - I'll find out and get back to you.' "
        "Not defer: 'you choose' (that is delegate); 'skip this' (that is skip); a tentative answer with content (that is answer)."),
    "dismiss": INTENT_DEFS["dismiss"] + (
        " Examples: 'Let's move on.' / 'That's covered, next topic.' / 'I've said all I have on that.' "
        "Not dismiss: a short answer that still gives content."),
    "skip": INTENT_DEFS["skip"] + (
        " Examples: 'Skip this one.' / 'I'd rather not get into that.' / 'Not relevant for us, skip it.' "
        "Not skip: 'you choose' (that is delegate); 'next topic' after answering (that is dismiss); "
        "'I'll find out and come back to you' (that is defer)."),
    "enough": INTENT_DEFS["enough"] + (
        " Examples: 'That's enough, write the spec.' / 'I'm done with the interview.' "
        "Not enough: closing only the current topic (that is dismiss)."),
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


def taxonomy_json() -> str:
    """Machine-readable export for tools that cannot import Python."""
    return json.dumps({"order": list(INTENT_ORDER), "short": INTENT_SHORT, "definitions": INTENT_DEFS,
                       "compatible_groups": [sorted(g) for g in COMPATIBLE_GROUPS],
                       "terminal": sorted(TERMINAL_INTENTS)}, indent=2)


if __name__ == "__main__":  # python -m interview_engine.intent_taxonomy
    print(taxonomy_json())
