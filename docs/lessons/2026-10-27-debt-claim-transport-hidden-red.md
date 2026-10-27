# Debt: an opt-in checkpoint transport test is red

Open, outside child phase 2. Reproduce:

    DELIVERY_TEST_DELIVERY_SOURCE=/path/to/delivery-api .venv/bin/python -m pytest tests/test_pg_claim_transport.py -q

The checkpoint-saver case fails while inserting engine_sessions with SQLSTATE
42501. The other two cases pass. Reproduced identically on clean engine main
0d1d48ad8f2f8bd40668112494d5c14ba72865c0 against delivery migration source
2c5f19b (phase-1 merge); this is not a child-seam regression. Without the
explicit delivery checkout these three cases skip, hiding the red.

Cheap causal read: the checkpoint case creates connect_claimed(..., owner,
Auth-session), then takes conn.info.dsn and constructs a NEW managed_pool
from it. Transaction-local claims are not carried by conninfo. PgSessionStore
and FencedPostgresSaver then get the raw pool without their explicit claims,
so the first RLS insert correctly refuses. The passing production-store case
uses tenant_worker.pg_session_store(pool, owner, Auth-session), which supplies
claims per borrow. This is a test wiring lead, not yet a verified fix.

Next separate item: reproduce failing-first, bind explicit claims into both
store and saver test adapters, retain every assertion, add a negative that
unclaimed borrowers remain denied, and rerun the option-dropping transport
matrix. Do not put claims back in connection startup options or weaken RLS.
