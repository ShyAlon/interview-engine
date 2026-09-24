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

v1.3 (2026-09-24, Shy): ``dismiss`` is removed. It sat too close to skip and
added noise either way, so ``skip`` now covers both "wants to move on" and
"doesn't want to tackle this topic". Old records and frozen rubric variants
may still say dismiss; ``normalize`` maps it to skip.
"""
from __future__ import annotations

import json

# Canonical order: the order labels appear in prompts, tables and tools.
INTENT_ORDER: tuple[str, ...] = ("answer", "delegate", "defer", "skip", "enough")
# Retired labels -> their v1.3 label. Applied to classifier output, stored
# labels and scoring so older records stay usable.
LEGACY_ALIASES: dict[str, str] = {"dismiss": "skip"}

# One-line criteria: the canonical label text shown to classifiers and
# labelers. v1.2 (2026-09-23, Shy's rewrite): each label is phrased as what
# the founder WANTS, so the classes do not overlap. v1.3 folds dismiss into
# skip (the v1.2 text is frozen in categorization_rubrics as intent-v1.2).
INTENT_SHORT: dict[str, str] = {
    "answer": "Founder gives product information, even if it doesn't directly address the question asked",
    "delegate": "Founder wants the interviewer to decide",
    "defer": "Founder wants to answer later",
    "skip": "Founder wants to move on, including not wanting to tackle this topic",
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
    "skip": ("The founder wants to move on from the current question, or does not want to tackle this topic at all; "
             "nobody is handed the choice, nothing is promised for later, and the interview continues."),
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
        "Not defer: 'you choose' (that is delegate); 'I'd rather not get into that' (that is skip); a tentative answer with content (that is answer)."),
    "skip": INTENT_DEFS["skip"] + (
        " Examples: 'Let's move on.' / 'Next question.' / 'I've said all I have on that.' / 'I'd rather not get into that.' "
        "/ 'That topic isn't relevant for us.' "
        "Not skip: 'you choose' (that is delegate); 'I'll come back to you on that' (that is defer); ending the whole interview (that is enough)."),
    "enough": INTENT_DEFS["enough"] + (
        " Examples: 'That's enough, write the spec.' / 'I'm done with the interview.' "
        "Not enough: moving on from only the current question (that is skip)."),
}

# Labels that close the current area / interview.
TERMINAL_INTENTS: frozenset[str] = frozenset({"skip", "enough"})
# Control moves a classifier must assert with high confidence before code acts
# on them; a low-confidence one falls back to answer.
GATED_INTENTS: frozenset[str] = TERMINAL_INTENTS | {"defer"}
# Labels that count as correct for each other in scoring (locked 2026-09-23):
# delegate is an autonomy grant, not a non-answer. defer is compatible with
# neither.
COMPATIBLE_GROUPS: tuple[frozenset[str], ...] = (frozenset({"answer", "delegate"}),)

assert tuple(INTENT_SHORT) == tuple(INTENT_DEFS) == tuple(INTENT_EXAMPLES) == INTENT_ORDER


def normalize(label):
    """Map a retired label (dismiss) to its v1.3 label; other values pass through."""
    return LEGACY_ALIASES.get(label, label) if isinstance(label, str) else label


def compatible(pred: str, labels, legacy: bool = False) -> bool:
    """True when ``pred`` is in ``labels`` or shares a compatibility group with one.
    Retired labels are normalized first unless ``legacy`` (score under v1.2)."""
    if not legacy:
        pred = normalize(pred); labels = [normalize(str(l)) for l in labels]
    labels = [str(l) for l in labels]
    if pred in labels:
        return True
    return any(pred in g and any(l in g for l in labels) for g in COMPATIBLE_GROUPS)


# Rubric version of the canonical criteria above. Bump with any text change.
TAXONOMY_VERSION = "categorization-request-v1.3"
# Labelers may pick several intent labels for one reply (locked 2026-09-23).
MULTI_SELECT = True


def taxonomy_json() -> str:
    """Machine-readable export for tools that cannot import Python."""
    return json.dumps({"version": TAXONOMY_VERSION, "slot": "intent", "key": "intent", "multi_select": MULTI_SELECT,
                       "order": list(INTENT_ORDER), "short": INTENT_SHORT, "definitions": INTENT_DEFS,
                       "compatible_groups": [sorted(g) for g in COMPATIBLE_GROUPS],
                       "terminal": sorted(TERMINAL_INTENTS), "legacy_aliases": LEGACY_ALIASES}, indent=2)


if __name__ == "__main__":  # python -m interview_engine.intent_taxonomy
    print(taxonomy_json())
