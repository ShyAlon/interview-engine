"""PG diagnostic sinks keep exact evidence without local files or outer locks."""
import json
from dataclasses import asdict
from pathlib import Path
import pytest


def test_bound_diagnostics_use_owned_documents_and_no_external_publication_lock(tmp_path,monkeypatch):
    from interview_engine.document_diagnostics import bind
    from interview_engine.raw_output import RawOutputLog
    from interview_engine.verification import VerificationLog
    from interview_engine.categorization import DecisionAudit,jsonl_audit_sink
    from interview_engine import model_config as mc,stop_fence
    rows={}
    class Documents:
        def read(self,key):return rows.get(key)
        def write(self,key,raw):rows[key]=raw
        def append(self,key,raw):rows[key]=rows.get(key,b'')+raw
    root=tmp_path/'session';out=root/'interview'
    def forbidden(call):raise AssertionError('PG documents own their task transaction')
    monkeypatch.setattr(stop_fence,'guarded_publish',forbidden)
    token=mc._session_llm_log.set(str(out/'llm-calls.jsonl'))
    audit=DecisionAudit('scope',{'answer':'accepted'},{'answer':.9},'rules','rules','none',(),'now')
    try:
        with bind(Documents(),root):
            path=RawOutputLog(out/'debug').write('conops','item','  exact raw\n')
            VerificationLog(out/'verification.jsonl').write({'score':1})
            jsonl_audit_sink(out/'decisions.jsonl')(audit)
            mc._log_llm({'event':'diagnostic'})
            with pytest.raises(ValueError):RawOutputLog(tmp_path/'foreign').write('x','x','no')
    finally:mc._session_llm_log.reset(token)
    assert rows['interview/debug/'+path.name]==b'  exact raw\n'
    assert json.loads(rows['interview/verification.jsonl'])['score']==1
    assert json.loads(rows['interview/decisions.jsonl'])==json.loads(json.dumps(asdict(audit)))
    assert json.loads(rows['interview/llm-calls.jsonl'])['event']=='diagnostic'
    assert not root.exists() and not (tmp_path/'foreign').exists()


def test_bound_shadow_keeps_full_record_and_report_without_local_files(tmp_path):
    from interview_engine.document_diagnostics import bind
    from interview_engine.categorization_shadow import ShadowCategorizer,render_report
    from test_categorization_shadow import Cat,req,result
    rows={}
    class Documents:
        def read(self,key):return rows.get(key)
        def write(self,key,raw):rows[key]=raw
        def append(self,key,raw):rows[key]=rows.get(key,b'')+raw
    root=tmp_path/'session';path=root/'interview/shadow.jsonl'
    baseline=result('rules','accepted');jev=result('typesafe','rejected')
    with bind(Documents(),root):
        shadow=ShadowCategorizer(Cat(baseline),Cat(jev),path)
        assert shadow.classify(req()) is baseline
        assert 'Decisions: 1' in render_report([path])
    row=json.loads(rows['interview/shadow.jsonl'])
    assert row['input']['state']=='secret founder input'
    assert not row['agreement'] and row['authority']=='baseline'
    assert b'Decisions: 1' in rows['interview/shadow.md']
    assert not root.exists()


def test_nested_legacy_context_cannot_inherit_another_session_diagnostics(tmp_path):
    from interview_engine.document_diagnostics import bind,configured,append
    rows={}
    class Documents:
        def append(self,key,raw):rows[key]=raw
    with bind(Documents(),tmp_path):
        assert configured()
        with bind(None,tmp_path/'legacy'):assert not configured()
        assert configured();append(tmp_path/'interview/a.jsonl','a\n')
    assert not configured() and rows=={'interview/a.jsonl':b'a\n'}
