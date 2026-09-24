"""Single source of truth for the coverage-area taxonomy (the elicitation
checklist, ADR-0016).

Every place that enumerates coverage areas reads them from here: the coverage
model and interviewer status block (``coverage``), the structured-turn JSON
schema and prompt (``coverage.turn_output_schema``, ``prompts``), the
classifier's per-area covers_* statements (``categorization.COVERAGE_AREAS``),
rendering, and the replay scorer (``scripts/score-rubric-variants.py``).
Add, rename or reorder an area here and nowhere else.

Taxonomy v1 = the 11 areas below (vmodel-contracts COVERAGE_SCHEMA 1.0.0).
A taxonomy v2 (nfrs split, Shy 2026-09-24) lands as a new version next to
this one, never by editing v1 in place.
"""
from __future__ import annotations

TAXONOMY_VERSION = "coverage-v1"

# (key, human label, what "concrete" means for closing it) - order is priority.
AREAS = [
    ("goal", "goal & core value", "the problem and who hurts, in a sentence"),
    ("users", "stakeholder context", "primary and supporting stakeholders, their roles, and where/when they operate"),
    ("flows", "operational scenarios", "at least one concrete trigger-to-outcome scenario with actors, ordered steps, and exceptions"),
    ("data", "data model & sources", "what data, where it lives, who owns it"),
    ("integrations", "integrations", "named systems/APIs, not categories"),
    ("nfrs", "quality needs", "founder-visible business impact; engineering maps it to measurable targets"),
    ("failures", "failure modes", "what breaks, what degrades, what must never happen"),
    ("constraints", "constraints", "budget, deadlines, team - and asked OPENLY: 'any technology constraints - language/ecosystem, cloud affinity?' ('agnostic' closes it too)"),
    ("risks", "risks", "what assumption kills the product if wrong"),
    ("done", "done criteria", "how the founder will know it works"),
    ("glossary", "glossary", "the founder's domain terms pinned to their meanings"),
]
AREA_KEYS = [k for k, _, _ in AREAS]
LABELS = {k: label for k, label, _ in AREAS}
_LABELS = LABELS  # legacy private name


# ---------------------------------------------------------------------------
# Taxonomy v2 - PREPARED, NOT WIRED. Ships as bench-3 after bench-2 lands
# (Shy 2026-09-24, nfrs-scoping + pool-v3 markups). v1 above stays the default;
# nothing reads the v2 names yet.
#   - nfrs splits into performance, reliability, security & privacy,
#     scalability, in that order, where nfrs sat. Usability shelved (overlaps
#     users). 14 areas.
#   - reliability = how often / when unavailable; failures = impact and
#     mitigation.
#   - numbers needed for performance, reliability, scalability; a thin answer
#     gets ONE clarification question (DRILL_PROBES_V2) - the drill case.
#   - drill cap 8 per interview.
# ---------------------------------------------------------------------------
TAXONOMY_VERSION_V2 = "coverage-v2"

_QUALITY_V2 = [
    ("performance", "performance",
     "how fast it must feel for whom, in which moments (e.g. booking on a phone, a staff tablet) - in seconds, or an explicit 'no numbers'"),
    ("reliability", "reliability",
     "when and for how long it may be unavailable, and what must never be lost (what happens when it breaks belongs to failures)"),
    ("security", "security & privacy",
     "who may see or change what, which data is sensitive, any rules they must follow; standards accepted if offered, not required"),
    ("scalability", "scalability",
     "expected volume for now - order of magnitude of users, clients, bookings - and what that volume must not break"),
]
AREAS_V2 = []
for _row in AREAS:
    AREAS_V2.extend(_QUALITY_V2 if _row[0] == "nfrs" else [_row])
AREA_KEYS_V2 = [k for k, _, _ in AREAS_V2]
LABELS_V2 = {k: label for k, label, _ in AREAS_V2}
NUMERIC_AREAS_V2 = ("performance", "reliability", "scalability")
MAX_DRILLS_V2 = 8

# One clarification question per numeric area, with Shy's example answer
# shape (pool-v3 markup): asked when the founder's answer is thin.
DRILL_PROBES_V2 = {
    "performance": "How long is OK, in seconds - half a second? Two seconds?",
    "reliability": "How much downtime a year could you live with - a few hours? A day?",
    "scalability": "Roughly how many users - tens, hundreds, thousands?",
}

# Proposed fallback questions (engine asks these when the model gives none).
FALLBACK_QUESTIONS_V2 = {
    "performance": "How fast and sleek does it need to feel, and in which moments does waiting hurt most?",
    "reliability": "When is it OK for the product to be unavailable, and for how long?",
    "security": "Who should be able to see or change what, and is any of the data sensitive?",
    "scalability": "How many users and transactions do you expect it to handle in a week, for now?",
}
