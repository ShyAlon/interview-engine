# Observability is not an answer gate

The owner-selected answer path buffers diagnostics and usage, then performs one
fenced end-of-answer flush. A private per-operation local spool is best-effort
recovery evidence, never authority to replay paid work or override Stop/RLS.
Failure to persist diagnostics cannot turn a good answer into an error.

Live observational progress takes the private child channel, not a durable
claim/journal-mode/cursor transaction per stage. The parent keeps a bounded,
scoped transient stream. Normal non-answer durable progress remains unchanged.

Real local graph success measurements: 71 child barriers before these levers,
56 after diagnostic batching, 41 after transient progress. Provider, Stop and
unknown usage paths remain tested. No production activation is implied.

Checkpoint-memory reads canonicalize unpinned latest reads to checkpoint ids;
pending writes invalidate that same key. Serialize local cache access with
checkpoint writes, and populate only after commit. Tests compare actual
Postgres tuples, cover old pinned reads, failed fence writes and new savers.
The saver memory belongs to its explicitly enabled warm child, never a global
cross-session cache; non-answer saver construction keeps the durable path.
