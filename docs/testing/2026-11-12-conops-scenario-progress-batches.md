# CONOPS scenario call and progress-write reduction

One complete scenario call requests all missing fields together. Code-derived
identity, actors, outcome and founder source turns are never taken from that
response. Each returned field is normalized and checked against its contract;
missing/invalid fields use the existing targeted field path. Existing bounded
JSON repair and depth repair remain in place. Final item validation, provenance
checks and completeness findings remain unchanged. Options still use their six
calls; option batching is separate follow-up work.

CONOPS durable progress now keeps validated item completions plus phase and
founder-decision boundaries. Failures always pass. Field/repair signals still
reach console/raw subscribers, but no longer each start their own durable
observational transaction. Heartbeats remain transient. Control/checkpoint,
Stop/fencing and provider accounting are not changed.

## Hermetic benchmark

Nine recorded-shape fake scenarios return the same trigger, ordered flow and
exceptions, and retain all derived fields/source turns. Nine scenario calls plus
six unchanged option calls = 15 rather than 33 (18 fewer, 55%). The benchmark
asserts the entire scenario content and provenance, not only call count. It
asserts targeted partial/null/depth repair and bounded malformed-object repair.
Existing weak-model per-field replies still repair successfully and existing
shallow-answer findings remain visible. A real local Postgres journal records
14 rows total: two phase boundaries, one stakeholder completion, nine scenario
completions, two option completions. All item counts are monotone; progress does
not publish control authority. This is a write-count proof, not a measured
production latency/price promise. No actual LLM, provider expense or live-user
process was used.

Local gate: 787 passed, 3 skipped. Focused graph/progress/real-PG suite: 48 passed.
Hosted CI is unverified because budget is exhausted. Local-only gate as agreed.
