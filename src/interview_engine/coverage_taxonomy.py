"""Single source of truth for the coverage-area taxonomy (the elicitation
checklist, ADR-0016).

Every place that enumerates coverage areas reads them from here: the coverage
model and interviewer status block (``coverage``), the structured-turn JSON
schema and prompt (``coverage.turn_output_schema``, ``prompts``), the
classifier's per-area covers_* statements (``categorization.COVERAGE_AREAS``),
rendering, and the replay scorer (``scripts/score-rubric-variants.py``).
Add, rename or reorder an area here and nowhere else.

Taxonomy v1 = the 11 areas below (vmodel-contracts COVERAGE_SCHEMA 1.0.0).
v1.1 (D12) rewords the integrations closing text only; keys and order are unchanged,
so saved sessions are unaffected. A taxonomy v2 (nfrs split, Shy 2026-09-24) lands as a new version next to
this one, never by editing v1 in place.
"""
from __future__ import annotations

TAXONOMY_VERSION = "coverage-v1.1"
V1_TAXONOMY_VERSION = "coverage-v1"

# v1 as shipped. Never edited; kept so the v1 text stays reproducible.
# (key, human label, what "concrete" means for closing it) - order is priority.
AREAS_V1 = [
    ("goal", "goal & core value", "the problem and who hurts, in a sentence"),
    ("users", "stakeholder context", "primary and supporting stakeholders, their roles, and where/when they operate"),
    ("flows", "operational scenarios", "at least one concrete trigger-to-outcome scenario with actors, ordered steps, and exceptions"),
    ("data", "data model & sources", "what data, where it lives, who owns it"),
    ("integrations", "integrations", "named systems/APIs, not categories"),
    ("nfrs", "quality needs", "the founder-visible experience and business impact of quality failures; metrics come later"),
    ("failures", "failure modes", "what breaks, what degrades, what must never happen"),
    ("constraints", "constraints", "budget, deadlines, team - and asked OPENLY: 'any technology constraints - language/ecosystem, cloud affinity?' ('agnostic' closes it too)"),
    ("risks", "risks", "what assumption kills the product if wrong"),
    ("done", "done criteria", "what outcome would make the first useful version ready; measurable tests come later"),
    ("glossary", "glossary", "the founder's domain terms pinned to their meanings"),
]
# v1.1 (JEV-D12): same 11 keys in the same order; only the integrations closing
# text changes, so a question about integrations can end in "none" instead of
# demanding named systems the founder never mentioned.
_V1_1_TEXT = {"integrations": "any existing system or service it must exchange data with, or 'none'"}
AREAS = [(k, label, _V1_1_TEXT.get(k, text)) for k, label, text in AREAS_V1]
AREA_KEYS = [k for k, _, _ in AREAS]
LABELS = {k: label for k, label, _ in AREAS}
_LABELS = LABELS  # legacy private name
