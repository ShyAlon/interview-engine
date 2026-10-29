"""Buffered progress journal for database-backed sessions.

File mode appends every progress signal to progress.sqlite the moment it is
emitted. In database mode a signal becomes durable ONLY with its step: signals
buffer during an advance and commit inside the step's publish_step transaction
(control + journal pages + checkpoint pointer together). A crash mid-advance
leaves no half-step of journal pages.

Cursors are allocated by the single lease holder: the head is read once at
construction and every buffered signal takes the next integer. Only the claim
holder writes journal rows under the lease, so the head cannot move underneath
us; a replayed step re-publishes identical (cursor, page) pairs and the
store's idempotent-conflict rules apply.
"""
from __future__ import annotations

from uuid import uuid4


class BufferedJournal:
    def __init__(self, store, claim, session_id: str, task_id: str):
        self._session_id, self._task_id = session_id, task_id
        self._next = store.journal_head(session_id, claim.owner_id) + 1
        self._buffer: list[dict] = []

    def append(self, *, phase_key: str, stage: str, label: str, status: str,
               detail: str | None = None, completed: int | None = None, total: int | None = None):
        from vmodel_contracts.progress_signal import ProgressSignal
        signal = ProgressSignal(event_id=str(uuid4()), cursor=self._next, task_id=self._task_id,
                                session_id=self._session_id, phase_key=phase_key, stage=stage,
                                label=label, status=status, detail=detail,
                                completed=completed, total=total)
        self._next += 1
        self._buffer.append(signal.model_dump(mode='json'))
        return signal

    def pending(self) -> list[tuple[int, dict]]:
        return [(page['cursor'], page) for page in self._buffer]

    def acknowledge(self, count: int) -> None:
        # Called only after the pages committed with their step.
        del self._buffer[:count]


class ObservationalJournal:
    """Live fenced PG signals, not evidence that a graph step committed.

    Explicit constructor opt-in only. Never shares a session's cursor stream
    with BufferedJournal; the store checks the durable mode under row lock.
    """
    def __init__(self, store, claim, session_id: str, task_id: str):
        if claim.session_id != session_id:
            raise ValueError('journal session must match lease claim')
        self._store, self._claim = store, claim
        self._session_id, self._task_id = session_id, task_id

    def append(self, *, phase_key: str, stage: str, label: str, status: str,
               detail: str | None = None, completed: int | None = None, total: int | None = None):
        from datetime import datetime, timezone
        from vmodel_contracts.progress_signal import ProgressSignal
        signal = ProgressSignal(event_id=str(uuid4()), cursor=1, task_id=self._task_id,
            session_id=self._session_id, phase_key=phase_key, stage=stage, label=label,
            status=status, detail=detail, completed=completed, total=total,
            timestamp=datetime.now(timezone.utc))
        page=signal.model_dump(mode='json')
        page['journal_mode']='observational'
        cursor=self._store.append_observational_progress(self._claim,page)
        return signal.model_copy(update={'cursor':cursor})
