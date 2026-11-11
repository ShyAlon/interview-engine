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
