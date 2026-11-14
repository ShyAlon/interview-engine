# Keep future-call attribution beside the durable call cursor

Problem: answer-level settlement retained token totals and retail contributions, but a later export could not recover the provider model, logical call, field, latency or provider expense for each call. Export-time timestamp guesses would misattribute repairs and grouped calls.

The parent broker now receives optional bounded metadata from the existing call context. CONOPS supplies the exact requested field list, including grouped scenario and full option calls; field repairs name only the repaired field. Other calls keep their existing stage and any safe code-owned item label. An unavailable field/item stays absent. No prompt parsing or guessed field attribution.

Metadata does not participate in cursor identity, alter prompts, add provider calls, or change any retry, validation or checkpoint rule. The context resets at invocation exit. The checked-in child contract 0.4.0 adds an optional metadata callback; lifecycle and operation schemas are unchanged. vmodel-contracts stays 0.29.0.

Local proof: 797 fast tests passed, 3 skipped; 60 focused CONOPS/cursor/child checks passed, 2 skipped. The companion delivery tests cover a fresh-process real graph with fake HTTP and disposable PostgreSQL, preserving start/finish-only answer accounting. No live provider, runtime restart or deployment was used. Hosted CI is not claimed.

Ship with the companion delivery and generator changes, then rebuild the combined runtime. New metadata must not be sent to an old parent broker. The delivery migration records future facts only; it does not backfill historical calls.

Lesson: capture attribution at the call boundary, not by correlating export timestamps. A grouped call owns multiple fields and one charge, never a fictional charge per field.
