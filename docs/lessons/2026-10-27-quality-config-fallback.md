# Saved configuration corruption is not a first run

Missing saved config keeps normal defaults without a warning. Malformed or schema-
invalid saved config keeps sessions running on defaults but records fallback
status and one safe category-only warning per invalid episode. A valid read clears
the episode. Expose status beside the config API rather than changing its schema.
Do not log file bytes, unknown keys, validation text or paths. Regression failed
first and covers missing/corrupt/saved recovery.
