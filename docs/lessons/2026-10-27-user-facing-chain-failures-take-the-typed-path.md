# 2026-10-27 - A user-facing provider failure must keep its type to the boundary

A preview start died on an admission refusal and the UI showed only "engine
program failed during start: RuntimeError" - the refusal's what/cause/next
existed (translate_llm_error had already mapped it to UserFacingError) but
ProviderChain aggregated every provider failure into one RuntimeError, and the
HTTP boundary's minimal-disclosure 503 can only name a RuntimeError by type.

The aggregate was the information-loss point. Now: when any provider failed
with a UserFacingError, the chain raises UserFacingError ("all providers
failed for <slot>", cause = every provider's detail, next = the first
actionable step), which the existing typed 422 handler renders verbatim to the
customer. A chain of only plain bugs still raises RuntimeError - the
minimal-disclosure 503 plus sanitized operator header is the right shape for
genuine bugs, and this PR does not touch it.

Lesson: when an error taxonomy exists (UserFacingError vs everything else),
every AGGREGATION point must preserve it - str() concatenation silently
demotes typed, actionable failures to opaque ones.
