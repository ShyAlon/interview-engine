# 2026-11-07 - Ignore ad-hoc local diagnostics

Owner-machine scratch files (bench*.csv, categorization-shadow.*,
inspect-interview-recovery.py) showed up as untracked noise. Root-anchored
ignore rules keep them out of status without touching tracked sources.
