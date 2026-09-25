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


# ---------------------------------------------------------------------------
# trees-v2 statements (Shy 2026-09-25, after bench-v4): "Jev is very weak in
# reasoning. The questions should be very direct... let's dumb it down."
# Every statement asks Jev to READ, not reason: literal, checkable, and where
# judgment can't be avoided, the instruction to be explicit is in the statement.
# Only the wording changes; trees, IDs, R/O and truth are the same as v1.
# Plus one blunt asked-area gate (about__<area>): components count only when it
# says yes (option A2). Same single call.
# ---------------------------------------------------------------------------
_NO_GUESS = " Answer no if you would have to guess."
_ONLY_P = " Answer no if the reply does not mention the {p}."
STATEMENTS_V2: dict[tuple[str, str], str] = {
    ("goal", "problem"): "The reply says in plain words what is going wrong for the business today." + _NO_GUESS,
    ("goal", "who"): "The reply says which people have that problem. Answer no if people are only mentioned, not said to have the problem.",
    ("goal", "cost"): "The reply says what the problem costs today, such as lost time, money or customers.",
    ("users", "personas"): "The reply lists the kinds of people who will use the app.",
    ("users", "context"): "The reply says where or when the {p} uses the app, for example on a phone or at the desk." + _ONLY_P,
    ("users", "device"): "The reply names the device the {p} uses, such as a phone or a tablet." + _ONLY_P,
    ("users", "count"): "The reply gives how many {p} there are, as a number." + _ONLY_P,
    ("flows", "trigger"): "The reply says what the first step is - what starts it.",
    ("flows", "actors"): "The reply says which people do the steps it describes. Answer no if the reply describes no steps.",
    ("flows", "steps"): "The reply lists at least two steps in the order they happen.",
    ("flows", "outcome"): "The reply says what the last step is - how it ends.",
    ("flows", "exceptions"): "The reply says what happens when something goes differently, such as a cancellation or a no-show.",
    ("data", "what"): "The reply lists pieces of information the app must store, such as names, phone numbers or dates.",
    ("data", "source"): "The reply says where that information is kept today, such as a paper book or a spreadsheet.",
    ("data", "owner"): "The reply says who owns the information.",
    ("data", "retention"): "The reply says how long the information is kept.",
    ("data", "amount"): "The reply gives how much information there is, as a number.",
    ("integrations", "systems"): ("The reply names another company's product by its name (such as WhatsApp, Stripe or Google Calendar), "
                                  "or says the app connects to nothing. A description like 'the company we use' is not a name."),
    ("integrations", "purpose"): "The reply says what another app or service is used for.",
    ("integrations", "pricing"): "The reply says what another app or service costs, or who pays for it.",
    ("integrations", "direction"): "The reply says which way information moves between the app and another service.",
    ("performance", "moment"): "The reply names a moment where the {p} must not wait." + _ONLY_P,
    ("performance", "number"): ("The reply gives a time in seconds or minutes for the {p}, or says there is no number. "
                                "Words like 'fast' or 'instant' are not a number." + _ONLY_P),
    ("reliability", "allowed"): "The reply says when it is OK for the app to be down.",
    ("reliability", "forbidden"): "The reply says when the app must never be down.",
    ("reliability", "per_year"): "The reply gives an uptime percent or a total downtime for a year, such as 99.9% or 'a few hours a year'.",
    ("reliability", "never_lost"): "The reply says which information must never be lost.",
    ("reliability", "recovery"): "The reply says how fast the app must be back after it goes down.",
    ("security", "see"): "The reply says what the {p} may see." + _ONLY_P,
    ("security", "change"): "The reply says what the {p} may change." + _ONLY_P,
    ("security", "signin"): "The reply says how people sign in, such as with a password or a phone number.",
    ("security", "sensitive"): "The reply names information that is private or sensitive.",
    ("security", "trace"): "The reply says it must be possible to see who did what.",
    ("security", "standards"): "The reply names a law, rule or standard, such as GDPR.",
    ("scalability", "users_n"): "The reply gives a number for how many users or clients there are.",
    ("scalability", "volume"): "The reply gives a number of bookings or transactions per day, week or month.",
    ("scalability", "peak"): "The reply gives a number for the busiest time.",
    ("failures", "worst"): "The reply says in plain words what the worst failure would be." + _NO_GUESS,
    ("failures", "never"): "The reply says something must never happen, using words like 'never', 'must not' or 'can't'.",
    ("failures", "when_down"): "The reply says what people will do while the app is down.",
    ("constraints", "budget"): "The reply gives an amount of money for building the app.",
    ("constraints", "deadline"): "The reply gives a date or time by which the app must be ready.",
    ("constraints", "maintainer"): "The reply says who will look after the app once it is live, or that nobody on their side can.",
    ("constraints", "technology"): ("The reply names a technology they want or must avoid (a language, cloud or platform), or says they "
                                    "don't care. A phone or a tablet is not a technology choice."),
    ("constraints", "running_costs"): "The reply says whether they will pay a monthly fee for outside services, or how much.",
    ("risks", "assumption"): "The reply says something the founder is not sure is true, and that the app fails if it is not true." + _NO_GUESS,
    ("risks", "notice"): "The reply says how the founder would see that this is happening.",
    ("done", "measure"): "The reply names something they will count or check after launch to see if the app works. Answer no if nothing is said about after launch.",
    ("done", "target"): "The reply gives a target value for that thing, such as a number.",
    ("done", "by_when"): "The reply says by when that target should be reached, or answers inside a time the question set.",
    ("glossary", "terms"): "The reply puts a word in quotes, says 'we call it ...', or explains a word. Ordinary words like 'booking' or 'tablet' do not count.",
    ("glossary", "meaning"): "The reply says what a special word means.",
}
assert set(STATEMENTS_V2) == {(a, c.key) for a, cs in TREES.items() for c in cs}, "v2 wording must cover every component"

# Blunt asked-area gate (A2): what each area is about, in plain words.
AREA_ABOUT = {
    "goal": "the problem the app solves, or who has that problem",
    "users": "who will use the app",
    "flows": "the steps of using the app, from start to end",
    "data": "what information the app keeps",
    "integrations": "other apps or services the app connects to",
    "performance": "how fast the app must be",
    "reliability": "when the app may be down, and for how long",
    "security": "who may see or change what, or which information is private",
    "scalability": "how many users, clients or bookings the app must handle",
    "failures": "what goes wrong when the app fails, and what must never happen",
    "constraints": "budget, deadline, who looks after the app, or technology",
    "risks": "an assumption that could make the whole app a mistake",
    "done": "how the founder will know, after launch, that the app works",
    "glossary": "what special words used in the business mean",
}
assert list(AREA_ABOUT) == AREA_KEYS_V2


def statements_v2(area: str, personas: Iterable[str] = DEFAULT_PERSONAS) -> dict[str, str]:
    """trees-v2: literal component statements + the asked-area gate + skips + mentioned."""
    personas = list(personas)
    out = {f"about__{area}": f"The reply is about {AREA_ABOUT[area]}. Answer yes only if that is what the reply talks about."}
    for c in TREES[area]:
        text = STATEMENTS_V2[(area, c.key)]
        if c.per_persona:
            for p in personas: out[_key(f"{area}.{slug(p)}.{c.key}", "comp")] = text.replace("{p}", p)
        else: out[_key(f"{area}.{c.key}", "comp")] = text
    base = statements(area, personas)
    out.update({k: v for k, v in base.items() if not k.startswith("comp__")})
    return out


# trees-v3 calibration: rules remain domain-neutral; examples live here instead
# of being baked into the rule. The salon pack is the benchmark's default.
# Other domains can supply their own examples without changing the tree,
# truth, or yes/no keys. No persona-name aliasing in v3: the replay context
# only carries titles, and earlier questions are not preserved in the call.
# Future production support should persist detected entities across questions,
# resolve a name ("Bob") to a role/title when known, and prefer the title in
# statements, while still supporting systems built specifically for Bob.
DOMAIN_EXAMPLES_V3: dict[str, dict[tuple[str, str], str]] = {
    "salon": {
        ("goal", "cost"): "lost time, money or customers",
        ("users", "context"): "on a phone or at the desk",
        ("users", "device"): "a phone or a tablet",
        ("flows", "exceptions"): "a cancellation or a no-show",
        ("data", "what"): "owner and dog details",
        ("data", "source"): "a paper book or a spreadsheet",
        ("integrations", "systems"): "WhatsApp, Stripe or Google Calendar",
        ("reliability", "forbidden"): "on weekends",
        ("reliability", "per_year"): "99.9% or a few hours a year",
        ("security", "signin"): "a password or a phone number",
        ("security", "standards"): "GDPR",
        ("constraints", "technology"): "a language, cloud or platform",
        ("done", "target"): "a number",
    },
}

# Delta from v2. Keep unchanged component statements frozen so that the
# comparison measures the calibration, not an incidental rewrite.
STATEMENTS_V3 = {**STATEMENTS_V2,
    ("flows", "steps"): "The reply says at least one thing that happens in the flow. One step is enough.",
    ("data", "what"): "The reply says what information the app keeps, even in general words like {example}.",
    ("integrations", "systems"): "The reply names every outside service it mentions by product name, for example {example}. A description like 'the company we use' is not a name. Answer no if any mentioned service is only described, not named.",
    ("integrations", "purpose"): "The reply says the job an outside app or service does. Naming the job is enough, even if the reply does not name that service.",
    ("reliability", "forbidden"): "The reply says a time when the app must be up, such as {example}.",
    ("failures", "never"): "The reply says something must never happen, using words like 'never', 'must not', 'can't', 'worst case' or 'would be a disaster'.",
    ("done", "measure"): "The reply says what result will show the app works after launch. Answer no if the reply does not talk about after launch.",
}
assert set(STATEMENTS_V3) == set(STATEMENTS_V2)

# Move illustrative examples out of the inherited v2 wording. These are
# explanatory only; the rule remains in STATEMENTS_V3.
_V3_EXAMPLE_SUFFIX = {
    ("goal", "cost"): "such as lost time, money or customers.",
    ("users", "context"): "for example on a phone or at the desk.",
    ("users", "device"): "such as a phone or a tablet.",
    ("flows", "exceptions"): "such as a cancellation or a no-show.",
    ("data", "source"): "such as a paper book or a spreadsheet.",
    ("reliability", "per_year"): "such as 99.9% or 'a few hours a year'.",
    ("security", "signin"): "such as with a password or a phone number.",
    ("security", "standards"): "such as GDPR.",
    ("done", "target"): "such as a number.",
}
for _key_v3, _example_v3 in _V3_EXAMPLE_SUFFIX.items():
    _text_v3 = STATEMENTS_V3[_key_v3]
    assert _example_v3 in _text_v3, _key_v3
    STATEMENTS_V3[_key_v3] = _text_v3.replace(_example_v3, "for example {example}.")
# Constraints' parenthetical is a rule (what counts as a technology), not an
# illustrative domain-specific example, so it stays attached to the rule.

AREA_ABOUT_V3 = {**AREA_ABOUT,
    "flows": "anything a person or the app does in a booking, even one step",
    "reliability": "anything about when the app must be up or may be down",
    "glossary": "a special word the business uses, even without explaining it",
}
# The goal gate was leaky on unrelated replies. An explicit exclusion gives
# margin without broadening the other gates in this calibration run.
AREA_GATE_EXCLUSIONS_V3 = {
    "goal": "A list of who uses the app, without saying what problem they have, is not about the goal.",
}


def statements_v3(area: str, personas: Iterable[str] = DEFAULT_PERSONAS, *, domain: str = "salon") -> dict[str, str]:
    """trees-v3: v2 keys and one call, with per-domain illustrative examples."""
    if domain not in DOMAIN_EXAMPLES_V3:
        raise ValueError(f"unknown trees-v3 example domain: {domain!r}")
    examples = DOMAIN_EXAMPLES_V3[domain]
    personas = list(personas)
    gate = f"The reply is about {AREA_ABOUT_V3[area]}. Answer yes only if that is what the reply talks about."
    if area in AREA_GATE_EXCLUSIONS_V3: gate += " " + AREA_GATE_EXCLUSIONS_V3[area]
    out = {f"about__{area}": gate}
    for c in TREES[area]:
        text = STATEMENTS_V3[(area, c.key)]
        if "{example}" in text:
            if (area, c.key) not in examples:
                raise ValueError(f"missing trees-v3 example for {domain}: {area}.{c.key}")
            text = text.replace("{example}", examples[(area, c.key)])
        if c.per_persona:
            for p in personas: out[_key(f"{area}.{slug(p)}.{c.key}", "comp")] = text.replace("{p}", p)
        else: out[_key(f"{area}.{c.key}", "comp")] = text
    base = statements(area, personas)
    out.update({k: v for k, v in base.items() if not k.startswith("comp__")})
    return out


def effective_fills(area: str, answers: Mapping[str, object]) -> tuple[set[str], set[str], set[str]]:
    """(filled, skipped, mentioned) after the asked-area gate: when the call carries
    about__<area> and it is no, nothing in the asked area counts as filled.
    Without a gate key (trees-v1) this is plain parse()."""
    filled, skipped, mentioned = parse(answers)
    gate = answers.get(f"about__{area}")
    if gate is not None:
        try: open_ = float(gate) >= THRESHOLD
        except (TypeError, ValueError): open_ = str(gate).lower() in ("yes", "true")
        if not open_: filled = {c for c in filled if not c.startswith(area + ".")}
    return filled, skipped, mentioned
