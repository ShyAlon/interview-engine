import json
from uuid import uuid4
import pyarrow.parquet as pq
import pytest
from vmodel_contracts.facets import (
    COVERAGE_SCHEMA_ID,COVERAGE_SCHEMA_VERSION,DeadLetterEntry,DeadLetterReason,
    EvidenceDisposition,EvidenceRecord,PromotionReview,PromotionStatus,SessionSchemaBinding,
)
from interview_engine.evidence import EvidenceLake


def evidence():
    return EvidenceRecord(evidence_id=uuid4(),run_id='r1',session_id='s1',sequence=1,
        schema_id=COVERAGE_SCHEMA_ID,schema_version=COVERAGE_SCHEMA_VERSION,facet_key='goal',option_id=30,
        disposition=EvidenceDisposition.CLASSIFIED,raw_text='Dispatchers lose calls.',provider='typesafe',
        confidence=.8,occurred_at='2026-09-22T08:00:00Z')


def test_evidence_is_raw_append_and_immutable_parquet_part(tmp_path):
    lake=EvidenceLake(tmp_path,'r1');part=lake.append_evidence(evidence())
    assert part.is_file() and pq.ParquetFile(part).read().num_rows==1
    row=json.loads((part.parent/'events.jsonl').read_text())
    assert row['raw_text']=='Dispatchers lose calls.' and row['schema_version']==COVERAGE_SCHEMA_VERSION


def test_session_binding_is_idempotent_but_cannot_drift(tmp_path):
    lake=EvidenceLake(tmp_path,'r1');first=lake.bind_session('s1');assert lake.bind_session('s1')==first
    with pytest.raises(ValueError,match='immutable'):
        lake.bind_session('s1',SessionSchemaBinding(session_id='s1',schema_id=COVERAGE_SCHEMA_ID,schema_version='2.0.0'))


def test_dlq_keeps_unsolicited_sla_and_cannot_promote_itself(tmp_path):
    lake=EvidenceLake(tmp_path,'r1');eid=uuid4();did=uuid4()
    part=lake.append_dead_letter(DeadLetterEntry(dlq_id=did,evidence_id=eid,run_id='r1',session_id='s1',
        schema_id=COVERAGE_SCHEMA_ID,schema_version=COVERAGE_SCHEMA_VERSION,reason=DeadLetterReason.UNSOLICITED_FACET,
        raw_text='Our SLA must be 99.95%.',proposed_facet_key='service_level',created_at='2026-09-22T08:00:00Z'))
    assert pq.ParquetFile(part).read().to_pylist()[0]['proposed_facet_key']=='service_level'
    with pytest.raises(ValueError,match='reviewed'):
        lake.append_promotion(PromotionReview(proposal_id=uuid4(),source_dlq_ids=[did],status=PromotionStatus.PROPOSED,
            target_schema_id=COVERAGE_SCHEMA_ID,target_schema_version='1.1.0'))

def test_unbucketed_answer_is_preserved_and_sent_to_dlq(tmp_path):
    from interview_engine.categorization import ClassificationAnswer,ProviderResult
    from interview_engine.evidence import record_founder_classification
    lake=EvidenceLake(tmp_path,'r1');lake.bind_session('s1')
    result=ProviderResult({'relevance':ClassificationAnswer('dodged',.9),
        'covers_goal':ClassificationAnswer(.1,None)},'typesafe','jev-latest')
    record,dead=record_founder_classification(lake,session_id='s1',sequence=2,
        raw_text='By the way, do you support an SLA?',target='goal',result=result)
    assert record.disposition is EvidenceDisposition.DEAD_LETTER and dead is not None
    assert dead.raw_text=='By the way, do you support an SLA?'
    assert (lake.paths.run/'dlq'/'events.jsonl').is_file()


def test_classified_answer_is_not_dead_lettered(tmp_path):
    from interview_engine.categorization import ClassificationAnswer,ProviderResult
    from interview_engine.evidence import record_founder_classification
    lake=EvidenceLake(tmp_path,'r1')
    result=ProviderResult({'relevance':ClassificationAnswer('addressed',.9),
        'covers_goal':ClassificationAnswer(.8,None)},'typesafe','jev-latest')
    record,dead=record_founder_classification(lake,session_id='s1',sequence=2,
        raw_text='Dispatchers lose calls.',target='goal',result=result)
    assert record.disposition is EvidenceDisposition.CLASSIFIED and dead is None

def test_missing_pyarrow_degrades_to_jsonl_without_blocking(tmp_path,monkeypatch,caplog):
    import builtins
    real_import=builtins.__import__
    def without_arrow(name,*args,**kwargs):
        if name=='pyarrow':raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_arrow)
    lake=EvidenceLake(tmp_path,'r1')
    returned=lake.append_evidence(evidence())
    assert returned.name=='events.jsonl' and returned.is_file()
    assert json.loads(returned.read_text())['raw_text']=='Dispatchers lose calls.'
    marker=returned.parent/'parquet-errors.jsonl';assert marker.is_file()
    assert json.loads(marker.read_text())['error_type']=='ModuleNotFoundError'
    assert 'parquet projection degraded' in caplog.text


def test_parquet_writer_failure_degrades_to_jsonl_without_blocking(tmp_path,monkeypatch):
    import pyarrow.parquet as parquet
    monkeypatch.setattr(parquet,'write_table',lambda *a,**k: (_ for _ in ()).throw(OSError('disk full')))
    lake=EvidenceLake(tmp_path,'r1');returned=lake.append_evidence(evidence())
    assert returned.name=='events.jsonl' and returned.is_file()
    assert json.loads((returned.parent/'parquet-errors.jsonl').read_text())['error_type']=='OSError'
