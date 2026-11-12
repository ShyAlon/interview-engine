"""Answer diagnostics are local/best-effort, never a per-call DB gate."""
import json
import pytest


def test_answer_buffers_diagnostics_and_usage_in_one_flush(tmp_path):
    from interview_engine.document_diagnostics import answer_batch,bind,append,write,read
    from interview_engine.pg_documents import InterviewDocuments
    from types import SimpleNamespace
    docs=InterviewDocuments(SimpleNamespace(claims=('owner','auth')),SimpleNamespace(owner_id='owner'), 'task')
    flushed=[]
    docs.diagnostic_many=lambda operations:flushed.append(operations)
    with answer_batch(tmp_path/'recovery',{'owner_id':'owner','session_id':'session','task_id':'task','generation':1,'operation_id':'answer'}):
        with bind(docs,tmp_path/'session'):
            append(tmp_path/'session/interview/log.jsonl','a\n')
            append(tmp_path/'session/interview/log.jsonl','b\n')
            write(tmp_path/'session/interview/debug/raw.txt','exact raw')
            docs.append('interview/usage.jsonl',b'usage\n')
            assert read(tmp_path/'session/interview/debug/raw.txt')=='exact raw'
            assert not flushed
            files=list((tmp_path/'recovery').glob('*.json'))
            assert len(files)==1
            assert json.loads(files[0].read_text())['identity']['operation_id']=='answer'
    assert len(flushed)==1
    assert flushed[0]==[('append','interview/log.jsonl',b'a\nb\n'),('write','interview/debug/raw.txt',b'exact raw'),('append','interview/usage.jsonl',b'usage\n')]
    assert not list((tmp_path/'recovery').glob('*.json'))


def test_flush_failure_preserves_spool_without_failing_answer(tmp_path,caplog):
    from interview_engine.document_diagnostics import answer_batch,bind,append
    class Docs:
        def diagnostic_many(self,ops):raise RuntimeError('database unavailable')
    with answer_batch(tmp_path/'recovery',{'owner_id':'a','session_id':'s','task_id':'t','generation':1,'operation_id':'op'}):
        with bind(Docs(),tmp_path/'session'):append(tmp_path/'session/interview/log.jsonl','a\n')
    assert len(list((tmp_path/'recovery').glob('*.json')))==1
    assert 'diagnostic.flush_failed' in caplog.text


def test_exception_flushes_best_effort_and_keeps_original_error(tmp_path):
    from interview_engine.document_diagnostics import answer_batch,bind,append
    class Docs:
        def diagnostic_many(self,ops):raise RuntimeError('flush failure')
    with pytest.raises(ValueError,match='original'):
        with answer_batch(tmp_path/'recovery',{'owner_id':'a','session_id':'s','task_id':'t','generation':1,'operation_id':'op'}):
            with bind(Docs(),tmp_path/'session'):
                append(tmp_path/'session/interview/log.jsonl','a\n')
                raise ValueError('original')


def test_recovery_ceiling_never_deletes_evidence_or_fails_answer(tmp_path,monkeypatch):
    from interview_engine import document_diagnostics as module
    root=tmp_path/'tenants/owner/sessions/.diagnostic-recovery'
    class Docs:
        def diagnostic_many(self,ops):raise RuntimeError('test unavailable')
    def append(operation,text):
        with module.answer_batch(root/('a'*64),{'operation_id':operation}):
            with module.bind(Docs(),tmp_path/'session'):
                module.append(tmp_path/'session/interview/run.log',text)
    monkeypatch.setattr(module,'SPOOL_TOTAL_FILES',1)
    append('old','old evidence')
    old=next(root.glob('*/*.json'));before=old.read_bytes()
    append('new','new evidence cannot evict old')
    assert list(root.glob('*/*.json'))==[old]
    assert old.read_bytes()==before
    assert old.stat().st_mode&0o777==0o600
    monkeypatch.setattr(module,'SPOOL_TOTAL_FILES',128)
    monkeypatch.setattr(module,'SPOOL_TOTAL_BYTES',len(before))
    append('third','owner byte ceiling')
    assert list(root.glob('*/*.json'))==[old]
    monkeypatch.setattr(module,'SPOOL_TOTAL_BYTES',32000000)
    monkeypatch.setattr(module,'SPOOL_FILE_BYTES',20)
    append('fourth','per-file ceiling')
    assert list(root.glob('*/*.json'))==[old]


def test_diagnostic_recovery_export_is_bounded_decoded_and_read_only(tmp_path):
    from interview_engine import document_diagnostics as module
    root=tmp_path/'tenants/owner/sessions/.diagnostic-recovery'/('a'*64)
    class Docs:
        def diagnostic_many(self,ops):raise RuntimeError('test unavailable')
    with module.answer_batch(root,{'owner_id':'owner','operation_id':'failed-diagnostic'}):
        with module.bind(Docs(),tmp_path/'session'):
            module.append(tmp_path/'session/interview/run.log','API_KEY=must-be-redacted-by-debug\nproject evidence\n')
    path=next(root.glob('*.json'));before=path.read_bytes()
    result=module.recovery_export(tmp_path)
    assert result['kind']=='diagnostic-only-not-execution-replay'
    assert result['records'][0]['diagnostic_operations'][0]['text'].endswith('project evidence\n')
    assert not result['omissions']
    assert module.recovery_export(tmp_path,max_bytes=1)['omissions']==['diagnostic export size limit']
    assert path.read_bytes()==before
    outside=tmp_path/'outside';outside.write_text('private outside source')
    (root/('b'*64+'.json')).symlink_to(outside)
    assert len(module.recovery_export(tmp_path)['records'])==1


def test_concurrent_spool_capacity_counts_crash_temporaries(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from interview_engine import document_diagnostics as module
    root=tmp_path/'.diagnostic-recovery'/('a'*64)
    root.mkdir(parents=True)
    (root/'.diagnostic-crash').write_bytes(b'left by interrupted atomic write')
    class Docs:
        def diagnostic_many(self,ops):raise RuntimeError('test unavailable')
    monkeypatch.setattr(module,'SPOOL_TOTAL_FILES',2)
    def write(index):
        with module.answer_batch(root,{'operation_id':str(index)}):
            with module.bind(Docs(),tmp_path/'session'):
                module.append(tmp_path/'session/interview/run.log','evidence')
    with ThreadPoolExecutor(6) as pool:list(pool.map(write,range(6)))
    assert len(list(root.glob('*.json')))==1
    assert (root/'.diagnostic-crash').read_bytes()==b'left by interrupted atomic write'


def test_delayed_confirmed_flush_does_not_delete_a_newer_copy(tmp_path):
    from interview_engine import document_diagnostics as module
    root=tmp_path/'recovery'
    identity={'operation_id':'same-answer'}
    newer=module._AnswerBatch(root,identity)
    class PendingDocs:
        def diagnostic_many(self,ops):raise RuntimeError('not saved yet')
    class CommittedDocs:
        def diagnostic_many(self,ops):
            assert ops==[('append','interview/run.log',b'older saved notes')]
            newer.queue(PendingDocs(),'append','interview/run.log',b'newer unsaved notes')
    with module.answer_batch(root,identity):
        with module.bind(CommittedDocs(),tmp_path/'session'):
            module.append(tmp_path/'session/interview/run.log','older saved notes')
    assert newer.path.is_file(), 'a delayed confirmed flush cannot delete a replaced copy'
    assert b'newer unsaved notes' in __import__('base64').b64decode(json.loads(newer.path.read_text())['operations'][0][2])


def test_successful_save_with_cleanup_failure_keeps_copy_and_names_cleanup(tmp_path,monkeypatch,caplog):
    from interview_engine import document_diagnostics as module
    class Docs:
        def diagnostic_many(self,ops):pass
    batch=module._AnswerBatch(tmp_path/'recovery',{'operation_id':'saved'})
    batch.queue(Docs(),'append','interview/run.log',b'saved')
    original=module.Path.unlink
    def fail(path,*args,**kwargs):
        if path==batch.path:raise PermissionError('test unlink denied')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(module.Path,'unlink',fail)
    batch.flush()
    assert batch.path.is_file()
    assert 'diagnostic.cleanup_deferred' in caplog.text
    assert 'diagnostic.flush_failed' not in caplog.text


@pytest.mark.parametrize('change',['malformed','symlink','oversize'])
def test_confirmed_cleanup_preserves_uncertain_or_unsafe_replacements(tmp_path,change):
    from interview_engine import document_diagnostics as module
    target=tmp_path/'outside';target.write_bytes(b'outside must stay')
    class Docs:
        def diagnostic_many(self,ops):
            path=batch.path
            if change=='symlink':
                path.unlink();path.symlink_to(target)
            elif change=='oversize':path.write_bytes(b'x'*(module.SPOOL_FILE_BYTES+1))
            else:path.write_bytes(b'malformed new evidence')
    batch=module._AnswerBatch(tmp_path/'recovery',{'operation_id':'saved'})
    batch.queue(Docs(),'append','interview/run.log',b'saved')
    batch.flush()
    assert batch.path.exists() and target.read_bytes()==b'outside must stay'


def test_stop_error_and_its_unconfirmed_copy_survive_cleanup(tmp_path):
    from interview_engine import document_diagnostics as module
    from interview_engine.stop_fence import StopRequested
    class Docs:
        def diagnostic_many(self,ops):raise StopRequested('lost lease')
    with pytest.raises(StopRequested,match='original stop'):
        with module.answer_batch(tmp_path/'recovery',{'operation_id':'stopped'}):
            with module.bind(Docs(),tmp_path/'session'):
                module.append(tmp_path/'session/interview/run.log','unsaved stop evidence')
                raise StopRequested('original stop')
    assert len(list((tmp_path/'recovery').glob('*.json')))==1


def test_cleanup_does_not_evict_other_old_or_temporary_copies(tmp_path):
    import os
    from interview_engine import document_diagnostics as module
    root=tmp_path/'recovery';root.mkdir()
    old=root/('0'*64+'.json');old.write_bytes(b'old unconfirmed')
    temporary=root/'.diagnostic-crash';temporary.write_bytes(b'crash temporary')
    os.utime(old,(1,1));os.utime(temporary,(1,1))
    class Docs:
        def diagnostic_many(self,ops):pass
    with module.answer_batch(root,{'operation_id':'current'}):
        with module.bind(Docs(),tmp_path/'session'):
            module.append(tmp_path/'session/interview/run.log','confirmed current')
    assert old.read_bytes()==b'old unconfirmed'
    assert temporary.read_bytes()==b'crash temporary'
    assert list(root.glob('*.json'))==[old]
