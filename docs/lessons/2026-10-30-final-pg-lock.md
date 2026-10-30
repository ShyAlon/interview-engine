# Retiring a backend also means regenerating the lock

The PG-only runtime had no remaining file-store implementation, but uv.lock still
resolved the checkpoint SQLite package and its two driver dependencies. Regenerate
from the current project metadata rather than deleting lock blocks by hand.
Historical lessons, old-file deny patterns and negative absence guards stay.

Validation: all 99 test files in three disjoint batches, 334 + 248 + 343 = 925
passed with delivery's real migration source enabled, no skips or deselections.
An earlier monolithic attempt was interrupted. A later batch hit disk exhaustion
from expired local pytest data; after removing only inactive disposable test
output the exact failed batch passed. Hosted CI remains unverified.
