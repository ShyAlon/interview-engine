# 2026-10-26 - Chain errors must carry the denial detail, not just the headline

Incident: a blocked interview surfaced as `RuntimeError: all providers failed
for scope: llm: UserFacingError: stage 'scope_tiebreak' was stopped by
admission/accounting checks`. The admission denial itself
(`unresolved compute accounting requires review`, raised by the delivery
ledger and naming the exact check that fired) was captured by
`translate_llm_error` into `UserFacingError.cause` - and then dropped,
because `ProviderChain` formatted the failure as `str(exc)`, which for
`UserFacingError` is only the generic headline. The operator could not tell
which admission check blocked the run from the error alone.

Lessons:
- An exception whose `str()` is a summary is a formatting trap: every
  aggregation point (chains, retries, fallbacks) must render the structured
  detail, not the headline.
- `translate_llm_error` already did the right capture; the loss happened one
  layer up at the join point. Check the whole formatting path, not just the
  raise site.
- The regression test pins the real denial text: a chain that swallows the
  cause is a red test, not a code review comment.
