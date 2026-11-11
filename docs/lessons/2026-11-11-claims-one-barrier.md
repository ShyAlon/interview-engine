# Count identity setup in whole-answer latency

A transaction-local role plus two tenant claims previously took two protocol
barriers before the actual query. Sending the two identity statements through
one psycopg pipeline keeps their order and transaction-local lifetime while
removing one barrier per transaction. Never replace these claims with startup
options or session-level state. A native real-Postgres wire test checks one
Sync, both owner/session fields, and reset after rollback and commit.
