# The child must explicitly choose live observational progress

The child Start and restore constructors inherited SessionRegistry's buffered
journal default. Buffered pages publish only after an advance returns. This
removed live signals during the longest wait, even though the independent PG
observational journal and parent watcher already existed.

Choose observational mode at both child constructors. These signals describe
work in progress, not a committed graph result. Lease/Auth/Stop fencing remains
in the journal; control and checkpoint publication remain separate and atomic.
A typed real Start fixture blocks inside advance and reads a signal from another
connection before release. Delivery's actual child/real graph/fixture HTTP Start
journey also failed with an empty page before this change and passed after it.
No raw stderr, SQLite path, paid call or live database is needed.
