# 2026-11-09: Warm child session reuse - validate follow-ups against the live lease-fenced state

## Problem

Every interview answer paid a full cold-child lifecycle: process spawn, pool
build, claim, checkpoint restore, graph compile, then reap and release - about
26s of attach plus ~9s of reap around a ~33s model turn. Between answers the
session sat unleased and published, only to be re-claimed and re-restored from
its own last checkpoint seconds later.

## Root cause

`restored_child_step` is a one-operation context manager: claim, restore,
yield, then `registry.close()` and lease release in `finally`. No engine
primitive could hold the claim across sequential operations, so the delivery
parent had no way to keep a restored session alive between answers.

## Fix

`warm_child_step.WarmChildStep` holds the claimed, restored, heartbeating
session across operations. `child_operations.begin_warm` claims and serves the
first operation; `execute_warm` serves each follow-up; `WarmChildStep.close`
releases the lease exactly once.

Per-follow-up validation compares the operation's expected evidence with the
LIVE in-memory session (sequence, wait id, review revision, generation, stop
state) instead of re-reading the row. This is sound, not a shortcut: while the
lease is held, `publish_step` fences every durable write to that lease, so the
child is the only possible writer and the in-memory control state IS the
published row. The parent's operation evidence is built from its own post-turn
read of that same publish; matching expected_sequence binds the two views.
The checkpoint digest check adds nothing under a held lease and is documented
as deliberately omitted in `WarmChildStep.check`.

The published-op execution body (answer/route/advance) is extracted unchanged
into `_execute_published`, shared by the cold `execute_operation` and the warm
path, so the two paths can never drift apart.

## Evidence

- Fail-first: tests/test_warm_child_step.py on origin/main fails at
  collection - `ImportError: interview_engine.warm_child_step` does not exist.
- With the change: one claim serves a follow-up check with lease_epoch still
  2 and the child's placement_instance; diverged sequence/wait/review/
  generation/stop follow-ups raise PermissionError; a refused follow-up never
  costs the lease (a fresh claim succeeds after close); setup failure and
  production-factory refusal behave exactly like the cold seam.
- Full gates: 933 passed, 0 failed (three batches: 226 + 320 + 387).

## Lesson

Before building a wake/claim/read mechanism, check whether the held resource
already makes the state authoritative. Under an exclusive lease the live
object graph is the database row; the correct validation target is the graph,
and the correct fence remains the lease - not another round trip.
