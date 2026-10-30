"""Fixed service-global PG quality config over the host's existing pool.

Schema is owned by delivery-api. Runtime has SELECT only. Absence uses code
constants; invalid/unavailable PG is an error, never a file/memory fallback.
"""
from contextlib import contextmanager
import os

class PostgresQualityConfig:
    def __init__(self,pool):
        if os.environ.get('INTERVIEW_ENGINE_QUESTION_QUALITY_CONFIG'):
            raise ValueError('legacy quality file override is not allowed in PG runtime')
        self._pool=pool

    @contextmanager
    def transaction(self):
        pool=self._pool() if callable(self._pool) else self._pool
        with pool.connection(timeout=0.25) as c:
            with c.transaction():
                c.execute('SET LOCAL ROLE requi_engine_config_read')
                c.execute("SET LOCAL statement_timeout='1000ms'")
                c.execute("SET LOCAL lock_timeout='100ms'")
                yield c

    def get(self):
        from .question_config import defaults,validate
        with self.transaction() as c:
            row=c.execute("SELECT body FROM public.engine_quality_config WHERE config_key='question-quality'").fetchone()
        return defaults() if row is None else validate(row['body'])

    def status(self):return {'source':'postgres','fallback':False,'writable':False}

    def put(self,cfg):
        from .question_config import validate
        validate(cfg)
        raise PermissionError('service configuration edits require a reviewed operator write surface')
