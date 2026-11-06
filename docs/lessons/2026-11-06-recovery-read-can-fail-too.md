# 2026-11-06 - A recovery read can fail while the database is still down

The HTTP answer path already handled identical-response replay and provider
failure. But its failure handler loaded durable control without guarding the
cached session. If that recovery read also lost the database, memory retained
half-mutated acceptance state. The same HTTP answer then failed with "session
has no open input", despite the durable question still being open.

A failed reconciliation read proves nothing about whether a commit landed.
Evict the cached session when that read fails. The next request reloads durable
control. Do not automatically repeat model execution across an ambiguous commit.

Real HTTP tests use real Postgres and the real graph with fake models. Faults
cover advance, pre-commit publication, and post-commit acknowledgement loss,
each combined with a second database failure during recovery. After healing,
the same answer works without restart and committed answers never advance twice.
