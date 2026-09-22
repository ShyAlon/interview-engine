import json
from interview_engine.categorization import ClassificationAnswer,ClassificationRequest,ProviderResult
from interview_engine.categorization_shadow import ShadowCategorizer,render_report

class Cat:
    def __init__(self,result=None,error=None):self.result,self.error=result,error
    def active(self,slot):return True
    def classify(self,request):
        if self.error:raise self.error
        return self.result

def req():return ClassificationRequest('scope','secret founder input',{})
def result(provider,value,tokens=(0,0),latency=.1,cost=None):
    return ProviderResult({'answer':ClassificationAnswer(value,.8)},provider,provider+'-model',tokens[0],tokens[1],latency,cost)

def test_shadow_never_changes_baseline_and_appends_detailed_record(tmp_path):
    path=tmp_path/'shadow.jsonl';baseline=result('llm','accepted',(10,4),.2,.001);jev=result('typesafe','rejected',(3,1),.05,None)
    out=ShadowCategorizer(Cat(baseline),Cat(jev),path).classify(req())
    assert out is baseline
    row=json.loads(path.read_text())
    assert row['slot']=='scope' and not row['agreement']
    assert row['baseline']['verdict']=={'answer':'accepted'} and row['jev']['verdict']=={'answer':'rejected'}
    assert row['baseline']['input_tokens']==10 and row['jev']['output_tokens']==1
    assert row['input_ref'].startswith('sha256:') and 'secret founder input' not in path.read_text()
    assert (tmp_path/'shadow.md').is_file() and 'Decisions: 1' in (tmp_path/'shadow.md').read_text()

def test_shadow_error_is_observed_but_baseline_returns(tmp_path):
    path=tmp_path/'shadow.jsonl';baseline=result('rules','accepted')
    assert ShadowCategorizer(Cat(baseline),Cat(error=TimeoutError('down')),path).classify(req()) is baseline
    row=json.loads(path.read_text());assert row['jev_error_count']==1 and row['jev']['status']=='error'
    assert row['baseline']['cost_usd']==0.0

def test_report_rolls_up_slots_and_lists_disagreements(tmp_path):
    path=tmp_path/'shadow.jsonl';baseline=result('llm','accepted',(10,4),.2,.001);jev=result('typesafe','rejected',(3,1),.05,None)
    ShadowCategorizer(Cat(baseline),Cat(jev),path).classify(req())
    md=render_report([path])
    assert 'Agreement: 0/1 (0.0%)' in md and 'Input tokens: 10' in md
    assert 'Disagreements for human review' in md and 'Unpriced calls: 1' in md


def test_observability_io_failure_never_changes_authoritative_result(tmp_path,monkeypatch):
    baseline=result('rules','accepted')
    shadow=ShadowCategorizer(Cat(baseline),Cat(result('typesafe','accepted')),tmp_path/'shadow.jsonl')
    monkeypatch.setattr('interview_engine.categorization_shadow.os.fsync',lambda _fd: (_ for _ in ()).throw(OSError('disk full')))
    assert shadow.classify(req()) is baseline

def test_shadow_record_pins_session_schema_version(tmp_path):
    path=tmp_path/'shadow.jsonl';baseline=result('llm','accepted');jev=result('typesafe','accepted')
    ShadowCategorizer(Cat(baseline),Cat(jev),path,session_id='s1',schema_id='interview.coverage',schema_version='1.0.0').classify(req())
    row=json.loads(path.read_text())
    assert row['session_id']=='s1' and row['schema_id']=='interview.coverage' and row['schema_version']=='1.0.0'
