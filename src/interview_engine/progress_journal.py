"""Independent per-session progress journal; no control-session lock on readers.

SQLite commits each row and its cursor in one transaction. Connections are short
lived and independent so a synchronous graph advance cannot block HTTP reads.
"""
from __future__ import annotations
from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from uuid import uuid4
from vmodel_contracts.progress_signal import ProgressSignal, ProgressPage


class ProgressJournal:
    def __init__(self, root: Path, session_id: str, task_id: str, *, create: bool = True):
        self.path = root / session_id / "interview" / "progress.sqlite"
        self.task_id, self.session_id = task_id, session_id
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as db:
                db.execute("CREATE TABLE IF NOT EXISTS progress (cursor INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def append(self, *, phase_key: str, stage: str, label: str, status: str,
               detail: str | None = None, completed: int | None = None,
               total: int | None = None) -> ProgressSignal:
        # Fail closed on invalid records before touching the journal. The bus
        # may swallow observability errors, but a malformed signal never lands.
        common = dict(task_id=self.task_id, session_id=self.session_id,
                      phase_key=phase_key, stage=stage, label=label,
                      status=status, detail=detail, completed=completed, total=total,
                      timestamp=datetime.now(timezone.utc))
        signal = ProgressSignal(event_id=str(uuid4()), cursor=1, **common)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor=db.execute("SELECT COALESCE(MAX(cursor),0)+1 FROM progress").fetchone()[0]
            signal=signal.model_copy(update={"cursor":cursor})
            db.execute("INSERT INTO progress(cursor,payload) VALUES (?,?)", (cursor,signal.model_dump_json()))
        return signal

    def read(self, after: int = 0, limit: int = 100) -> ProgressPage:
        with self._connect() as db:
            rows = db.execute("SELECT payload FROM progress WHERE cursor>? ORDER BY cursor LIMIT ?", (after, limit)).fetchall()
        events = [ProgressSignal.model_validate_json(row[0]) for row in rows]
        return ProgressPage(events=events, next_cursor=events[-1].cursor if events else after)
