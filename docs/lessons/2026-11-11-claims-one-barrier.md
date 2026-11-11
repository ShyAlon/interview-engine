# Count identity setup in whole-answer latency

A transaction-local role plus two tenant claims previously took two protocol
barriers before the actual query. Sending the two identity statements through
one psycopg pipeline keeps their order and transaction-local lifetime while
removing one barrier per transaction. Never replace these claims with startup
options or session-level state. A native real-Postgres wire test checks one
Sync, both owner/session fields, and reset after rollback and commit.

The first measurement must count frontend simple Query as well as Sync. In
implicit mode psycopg starts BEGIN separately. A queued explicit BEGIN still
reports IDLE until flushed, so executing identity in implicit mode emits a
second BEGIN and Sync. Explicit scoped transactions now temporarily enable
autocommit while a public Transaction owns BEGIN/COMMIT, restoring the pool's
original mode after rollback or commit. Nested savepoints keep their mode.
Implicit first queries combine role, claims and the query into the batch after
BEGIN. Checkpoint reads and writes use the same explicit setup helper.
