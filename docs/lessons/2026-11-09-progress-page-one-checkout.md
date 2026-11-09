# 2026-11-09 - The hot progress read paid two checkouts for one logical read

2026-11-09 production turn evidence: every progress fetch cost ~2.8s at
remote-pooler latency (~300ms/statement), on every poll and every wake of every
answer. The drain read the publication row and the journal page in TWO borrows
(load_published + read_journal), each paying claims setup and a round trip.

PgSessionStore.progress_page returns the pair in ONE borrow: same rows, same
owner scoping, half the fixed cost. SessionRegistry.progress uses it; the
KeyError contract for unpublished/foreign sessions is unchanged.

The event-driven half of this track already existed: the child re-signals the
parent's wake bus on every committed journal append over a private pipe
(delivery-api child_progress.py). Verified against production evidence before
building anything: all six ingests showed source_age_max 5927-6122ms - a
constant transport cost, not polling lag.
