# Root checkpoint config must include its namespace

LangGraph direct update_state eventually calls saver.put_writes with the supplied
config. SqliteSaver requires configurable.checkpoint_ns even for the root graph.
The application's config supplied only thread_id. New regression constructs the
real app with fixture models, advances once, directly updates graph state and
reads the root checkpoint. It fails KeyError checkpoint_ns in .76s before the fix.

One production-line change: add checkpoint_ns='' to the initial configurable.
No saver, checkpoint storage, restore pointer, fence or SQL behavior replaced.
New regression plus all PG application wiring:16pass21.46s. Existing item6 full
interview failure was the same exception. No old assertions edited, coverage
removed, skips/timeouts/dependencies changed, or production code deleted.
Hosted CI/full suite/deploy not run. No default switch or live DB touched.
