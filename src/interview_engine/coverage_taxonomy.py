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
