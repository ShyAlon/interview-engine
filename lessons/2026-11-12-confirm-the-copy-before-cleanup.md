# Confirm the exact copy before cleanup

A successful save confirms the operations that save owned, not everything now
stored at the same local path. Freeze those operations, then compare the exact
serialized snapshot under the writer's lock before unlinking. Keep unknown or
replaced evidence. Do not use age as a substitute for confirmation, and do not
call a cleanup failure a failed save. Never hold the local lock over the database
transaction or paid work. Diagnostic copies are not execution replay authority.
