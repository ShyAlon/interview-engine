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
