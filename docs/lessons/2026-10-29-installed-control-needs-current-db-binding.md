# Installed control needs current database binding

An installed generator execution preserves its approved interview source. It
must reject new interview input without requiring a local anchor file. Read
owner-scoped installed bindings on every control access, verify canonical anchor
hash and execution identity before constructing a fence, and recheck afterwards.

Graph publication and Stop control need different capabilities. A captured
worker must fail after Stop advances generation. The original execution's
control reader must still inspect its stopped generation, but must never accept
a different installed session or supply graph write authority.

Writer quiescence must use the same cross-process PG lease as execution. An
empty directory or missing file lock does not prove a deadline/provider thread
has exited. Engine quiescence does not settle provider holds or finish running
jobs; those remain separate checks before final stopped confirmation.

PG canonical output is not proof that diagnostic storage is PG-only. Bind raw
output, verification, decision audits, shadow records/reports and model-call
metadata to the exact session document adapter too. The adapter owns lease,
auth and Stop checks on each write. Never wrap that independent connection in
an outer task publication transaction. Fail-first tests poison the outer guard,
retain exact bytes/full shadow input, reject path escape and clear nested legacy
bindings so one session cannot inherit another's backend. Real PG tests cover
lease takeover and Stop; the connected graph forbids JSONL/raw local copies.
