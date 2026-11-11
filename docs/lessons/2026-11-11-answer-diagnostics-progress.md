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

Final publication has a dedicated answer-only conditional UPDATE: expected input
and sequence, live owner/task/generation/epoch/lease, and same-operation no-op.
The normal exact-byte publication path is unchanged. Entry-verified child stores
skip only the duplicate construction probe; claims/RLS remain on actual access.
Document guards join and lock the task and lease rows in one statement.

Immutable checkpoint blob versions can collide with proposed bytes. Populate
memory from already known canonical versions; when a version is not known,
fall back to durable reconstruction. A real collision test checks first-writer
bytes cannot be replaced in memory.

Answer checkpoints now stage in memory and flush inside the one final
control/pointer transaction. A wrong input, Stop, takeover or blob collision
rolls back both checkpoint and pointer. Failed publication discards staged
memory. One operation cannot stage another checkpoint after it has committed.
Pending writes preserve indexed regular writes and reserved-channel upserts.

The cold conditional claim also returns the exact published checkpoint and
schema readiness. The graph's answer-local latest begins at that published
anchor, not an unpublished saver-latest orphan. The same claim carries the
schema-binding document parts; reuse still checks owner/auth/generation/epoch/
instance, chunk order and hashes, and exact binding bytes. No RLS rule changed.

Real graph, fake provider and disposable PG: cold success 29 total barriers,
17 parent + 12 child; warm success 26, 17 parent + 9 child. Failure paths remain
higher and must be reported separately, not hidden behind the success result.

Failure-path review: WarmChildStep and SessionRegistry both released the same
lease. Registry close now returns checked release results; the warm wrapper
releases directly only for an evicted/unregistered session. It does not retry a
failed release. Failed setup also resets its identity tokens. Normal/failed/
evicted/lost-release tests pin exactly-once behavior. Refusal is now 29 whole
barriers; Stop remains 47. Do not describe this as a universal sub-30 pass.

Pending CheckpointTuple values omit DB indices. Counting regular channels loses
gaps from mixed reserved/regular writes. Staged writes now retain actual slot
keys; pre-existing pending writes load their real indices once. New checkpoints
remain query-free here. A fresh staged publication replay is rejected and rolls
back its proposed checkpoints, rather than retaining a pointer but leaving new
orphan rows. Already-committed same-context publication replay remains a no-op.
