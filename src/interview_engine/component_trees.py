"""Component trees for coverage-v2 (Shy 2026-09-24): replay-only, NOT WIRED.

One comprehensive, fixed superset tree per area of the 14-area v2 split
(coverage_taxonomy.AREAS_V2). Jev answers one yes/no statement per component
of the ASKED area in the same single coverage call; code derives relevance:
  none of the open required components filled = dodged, some = partial,
  all = addressed.
All-optional areas (risks, constraints): any component filled = addressed,
none = dodged, never partial.

Skipping (not applicable) is founder-only and hierarchical: a founder can rule
out a whole area, a persona's branch in an area, or one component. Skipping a
node closes everything under it. The interviewer never skips. A component-level
"doesn't apply" is an answer and fills that component; area and persona-branch
skips are one extra yes/no each in the same call (``na__...`` keys).

Every other area gets a cheap ``mentioned__<area>`` yes/no. It never fills a
component (cross-area answers stay dodged).

Production default is unchanged: nothing outside the replay reads this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .coverage_taxonomy import AREA_KEYS_V2

TREES_VERSION = "coverage-v2-trees-1"
THRESHOLD = .5


@dataclass(frozen=True)
class Component:
    key: str             # short key within the area, e.g. "number"
    label: str           # human label (review list, docs)
    required: bool       # R: needed for the area to be full; O: recorded, never asked
    statement: str       # yes/no statement for Jev; "{p}" = persona when per_persona
    followup: str = ""   # the question the interviewer asks when a required component is missing
    per_persona: bool = False


def _c(key, label, required, statement, followup="", per_persona=False):
    return Component(key, label, required, statement, followup, per_persona)


R, O = True, False
TREES: dict[str, tuple[Component, ...]] = {
    "goal": (
        _c("problem", "the problem", R, "The reply says what problem the product solves.", "What problem does this solve, in one sentence?"),
        _c("who", "who feels it", R, "The reply says who feels that problem.", "Who feels this problem most?"),
        _c("cost", "what it costs today", O, "The reply says what the problem costs today (time, money or customers)."),
    ),
    "users": (
        _c("personas", "named roles", R, "The reply names the kinds of people who will use the product.", "Who exactly will use it - which kinds of people?"),
        _c("context", "where/when they use it", R, "The reply says where or when the {p} uses the product.", "Where is the {p} when they use it?", True),
        _c("device", "on what device", O, "The reply says what device the {p} uses it on.", "", True),
        _c("count", "how many of them", O, "The reply says how many of the {p} there are.", "", True),
    ),
    "flows": (
        _c("trigger", "trigger", R, "The reply says what starts the scenario.", "What starts it - what does the client do first?"),
        _c("actors", "actors", R, "The reply says who takes part in the scenario.", "Who is involved along the way?"),
        _c("steps", "ordered steps", R, "The reply gives steps of the scenario in order.", "What happens next, step by step?"),
        _c("outcome", "outcome", R, "The reply says how the scenario ends.", "How does it end - what does done look like?"),
        _c("exceptions", "exceptions", O, "The reply says what happens when something in the scenario goes differently (a cancellation, a no-show, an error)."),
    ),
    "data": (
        _c("what", "what data", R, "The reply says what information the product must keep.", "What information exactly does it need to keep?"),
        _c("source", "where it comes from today", O, "The reply says where that information lives or comes from today."),
        _c("owner", "who owns it", O, "The reply says who owns the information."),
        _c("retention", "how long it is kept", O, "The reply says how long the information is kept."),
        _c("amount", "rough amount", O, "The reply says roughly how much information there is."),
    ),
    "integrations": (
        _c("systems", "named systems, or an explicit 'none'", R, "The reply names the specific outside systems the product connects to (by name, not a category), or says there are none.", "Which company exactly?"),
        _c("purpose", "what each is for", R, "The reply says what an outside system or service is needed for.", "What do you need it for?"),
        _c("pricing", "cost of the outside systems", O, "The reply says what an outside system or service costs, or who pays for it."),
        _c("direction", "which way data flows", O, "The reply says which way data flows between the product and an outside system."),
    ),
    "performance": (
        _c("moment", "the moments that matter", R, "The reply names a moment where speed matters for the {p}.", "For the {p}, which moment must never make them wait?", True),
        _c("number", "a number per moment, or an explicit 'no number'", R, "The reply gives a time limit in seconds or minutes for the {p}, or says plainly there is no number.", "For the {p}, how long is OK - in seconds?", True),
    ),
    "reliability": (
        _c("allowed", "allowed downtime", R, "The reply says when it is acceptable for the product to be unavailable.", "When is it OK for it to be down?"),
        _c("forbidden", "forbidden downtime", R, "The reply says when the product must never be unavailable.", "When must it never be down?"),
        _c("per_year", "uptime % or downtime per year", R, "The reply gives an uptime percentage or a total amount of downtime per year.", "How much downtime a year could you live with - a few hours? A day?"),
        _c("never_lost", "what must never be lost", O, "The reply says what information must never be lost."),
        _c("recovery", "how fast it must be back", O, "The reply says how fast the product must be back after an outage."),
    ),
    "security": (
        _c("see", "can see", R, "The reply says what the {p} may see.", "What can the {p} see?", True),
        _c("change", "can change", R, "The reply says what the {p} may change.", "What can the {p} change?", True),
        _c("signin", "how people sign in", O, "The reply says how people sign in."),
        _c("sensitive", "sensitive data", O, "The reply says which information is sensitive."),
        _c("trace", "who did what is traceable", O, "The reply says that actions must be traceable to who did them."),
        _c("standards", "rules or standards", O, "The reply names a rule, law or standard the product must follow."),
    ),
    "scalability": (
        _c("users_n", "users, order of magnitude", R, "The reply gives a number or order of magnitude for users or clients.", "Roughly how many users - tens, hundreds, thousands?"),
        _c("volume", "bookings/transactions per week", R, "The reply gives a number of bookings or transactions per day, week or month.", "Roughly how many bookings in a busy week?"),
        _c("peak", "peak", O, "The reply gives a peak volume."),
    ),
    "failures": (
        _c("worst", "worst failure", R, "The reply says what the worst failure would be.", "What is the worst thing that could go wrong?"),
        _c("never", "what must never happen", R, "The reply says what must never happen.", "What must never happen, no matter what?"),
        _c("when_down", "what staff do when it is down", R, "The reply says what people do when the product is down.", "If it is down on a busy day, how do you keep working?"),
    ),
    "constraints": (  # all optional (Shy 2026-09-24: founders mostly won't know)
        _c("budget", "budget to build", O, "The reply gives a budget for building the product."),
        _c("deadline", "deadline", O, "The reply gives a deadline for the product."),
        _c("maintainer", "who maintains it", O, "The reply says who will look after the product once it is live."),
        _c("technology", "technology, or 'agnostic'", O, "The reply states a technology preference or constraint, or says there is none."),
        _c("running_costs", "paying for outside services", O, "The reply says whether it is OK to pay monthly for outside services, or how much."),
    ),
    "risks": (  # all optional (Shy 2026-09-24)
        _c("assumption", "the assumption that kills it if wrong", O, "The reply names an assumption that would make the product fail if it were wrong."),
        _c("notice", "how you would notice", O, "The reply says how they would notice that assumption failing."),
    ),
    "done": (
        _c("measure", "measure", R, "The reply says what will be measured to know the product works.", "What would you measure?"),
        _c("target", "target", R, "The reply gives a target value for that measure.", "What number would count as success?"),
        _c("by_when", "by when", R, "The reply says by when the target should be reached (or answers inside a time frame the question set).", "By when should that happen?"),
    ),
    "glossary": (
        _c("terms", "terms", R, "The reply names domain terms the founder uses.", "Which words do you use that I should use exactly your way?"),
        _c("meaning", "per term: meaning", R, "The reply says what those terms mean.", "What does that term include?"),
    ),
}
assert list(TREES) == AREA_KEYS_V2, "trees must cover the v2 areas in order"
ALL_OPTIONAL_AREAS = tuple(a for a, cs in TREES.items() if not any(c.required for c in cs))
assert set(ALL_OPTIONAL_AREAS) == {"constraints", "risks"}

# Bench context: the three personas of the pool product (ANS-02). Production
# takes them from the interviewer's structured output (personas field).
DEFAULT_PERSONAS = ("client", "groomer", "front desk")


def slug(p: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in p.strip().lower()).strip("_") or "persona"


def component_ids(area: str, personas: Iterable[str] = DEFAULT_PERSONAS, required_only: bool = False) -> list[str]:
    out = []
    for c in TREES[area]:
        if required_only and not c.required: continue
        if c.per_persona: out += [f"{area}.{slug(p)}.{c.key}" for p in personas]
        else: out.append(f"{area}.{c.key}")
    return out


def component(cid: str) -> Component:
    area, *rest = cid.split(".")
    return next(c for c in TREES[area] if c.key == rest[-1])


def followup(cid: str, personas: Iterable[str] = DEFAULT_PERSONAS) -> str:
    c = component(cid); parts = cid.split(".")
    p = next((x for x in personas if slug(x) == parts[1]), parts[1]) if c.per_persona else ""
    return c.followup.replace("{p}", p)


def na_nodes(area: str, personas: Iterable[str] = DEFAULT_PERSONAS) -> list[str]:
    """Skippable nodes the call asks about: the area, and each persona branch when the area has one."""
    nodes = [area]
    if any(c.per_persona for c in TREES[area]): nodes += [f"{area}.{slug(p)}" for p in personas]
    return nodes


def _key(node_or_cid: str, prefix: str) -> str:
    return prefix + "__" + node_or_cid.replace(".", "__")


def statements(area: str, personas: Iterable[str] = DEFAULT_PERSONAS) -> dict[str, str]:
    """All yes/no statements for one coverage call on ``area``: components, skips, mentioned."""
    personas = list(personas); out = {}
    for c in TREES[area]:
        if c.per_persona:
            for p in personas: out[_key(f"{area}.{slug(p)}.{c.key}", "comp")] = c.statement.replace("{p}", p)
        else: out[_key(f"{area}.{c.key}", "comp")] = c.statement
    for node in na_nodes(area, personas):
        what = f"the {area} topic" if node == area else f"{area} for the {next(p for p in personas if slug(p) == node.split('.')[1])}"
        out[_key(node, "na")] = f"The founder says {what} does not apply to their product at all (not a refusal to answer)."
    for other in AREA_KEYS_V2:
        if other != area: out[f"mentioned__{other}"] = f"The reply mentions anything about {other}."
    return out


def parse(answers: Mapping[str, float]) -> tuple[set[str], set[str], set[str]]:
    """(filled component ids, skipped nodes, mentioned areas) from key -> value >= THRESHOLD."""
    filled, na, mentioned = set(), set(), set()
    for k, v in answers.items():
        try: yes = float(v) >= THRESHOLD
        except (TypeError, ValueError): yes = str(v).lower() in ("yes", "true")
        if not yes: continue
        if k.startswith("comp__"): filled.add(k[6:].replace("__", "."))
        elif k.startswith("na__"): na.add(k[4:].replace("__", "."))
        elif k.startswith("mentioned__"): mentioned.add(k[11:])
    return filled, na, mentioned


def closed_by_skip(cid: str, skipped: Iterable[str]) -> bool:
    return any(cid == n or cid.startswith(n + ".") for n in skipped)


def derive_relevance(area: str, filled: Iterable[str], skipped: Iterable[str] = (), delegated: Iterable[str] = (),
                     open_components: Iterable[str] | None = None, personas: Iterable[str] = DEFAULT_PERSONAS) -> str:
    """Relevance of ONE reply to a question on ``area``. Only this reply's fills count.

    open_components: required components still open when the question was asked
    (default: all of them). A follow-up question passes just its target.
    """
    filled = {c for c in filled if c.startswith(area + ".")}; skipped = set(skipped); delegated = set(delegated)
    if area in ALL_OPTIONAL_AREAS:
        return "addressed" if filled or area in skipped or any(d.startswith(area + ".") for d in delegated) else "dodged"
    req = component_ids(area, personas, required_only=True)
    open_ = req if open_components is None else list(open_components)
    hit = [c for c in open_ if c in filled or c in delegated or closed_by_skip(c, skipped)]
    if not hit: return "dodged"
    return "addressed" if len(hit) == len(open_) else "partial"


def review_list(area_state: Mapping[str, Mapping[str, object]], personas: Iterable[str] = DEFAULT_PERSONAS) -> list[str]:
    """Review-gate lines: every open required component and every skip, 'Circling back to X'.

    area_state[area] = {"filled": set, "delegated": set, "skipped": {node: founder words}}.
    """
    lines = []
    for area in AREA_KEYS_V2:
        st = area_state.get(area, {}); filled = set(st.get("filled", ())); deleg = set(st.get("delegated", ()))
        skipped = dict(st.get("skipped", {}))
        for node, words in skipped.items(): lines.append(f"Circling back to {node}: you said it does not apply (\"{words}\"). Still true?")
        for cid in component_ids(area, personas, required_only=True):
            if cid in filled or cid in deleg or closed_by_skip(cid, skipped): continue
            lines.append(f"Circling back to {cid}: {followup(cid, personas)}")
    return lines


# Interview design (Shy 2026-09-24, for step 5 wiring - NOT WIRED): a reply that
# only uses a plain speed or size word ("fast", "small") stays DODGED in the
# labels, but the interviewer asks ONE clarifying follow-up, then moves on.
PLAIN_WORD_PROBES = {
    "performance": "How fast, roughly - a second? A minute?",
    "scalability": "How small, roughly - tens of clients? Hundreds?",
}
MAX_PLAIN_WORD_PROBES_PER_AREA = 1
