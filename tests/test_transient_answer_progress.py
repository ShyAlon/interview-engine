from contextlib import contextmanager

def test_transient_progress_never_borrows_database():
    from interview_engine.pg_journal import ObservationalJournal,transient_progress
    from types import SimpleNamespace
    class Store:
        def append_observational_progress(self,*a):raise AssertionError('no progress database gate')
    rows=[];j=ObservationalJournal(Store(),SimpleNamespace(session_id='s'),'s','t')
    with transient_progress(rows.append):
        j.append(phase_key='intent',stage='probe',label='Probing',status='started')
        j.append(phase_key='intent',stage='probe',label='Probed',status='completed')
    assert [x['status'] for x in rows]==['started','completed']
    assert all(x['session_id']=='s' for x in rows)


def test_transient_progress_sink_failure_is_nonfatal(caplog):
    from interview_engine.pg_journal import ObservationalJournal,transient_progress
    from types import SimpleNamespace
    def broken(page):raise RuntimeError('gone')
    with transient_progress(broken):
        signal=ObservationalJournal(None,SimpleNamespace(session_id='s'),'s','t').append(phase_key='intent',stage='probe',label='Probing',status='started')
    assert signal.status=='started'
    assert 'progress.transient_lost' in caplog.text
