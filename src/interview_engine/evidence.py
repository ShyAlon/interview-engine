"""Append-only analytical evidence adapter.

JSONL is the immutable recovery/source record. Each append also emits an
immutable Parquet part for typed analytical scans. Runtime code can append DLQ
proposals, but only reviewed PromotionReview contracts can enter promotions.
"""
from __future__ import annotations
import json, logging, os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

log=logging.getLogger("interview_engine.evidence")

from vmodel_contracts.facets import (
    COVERAGE_SCHEMA, DeadLetterEntry, EvidenceRecord, PromotionReview,
    PromotionStatus, SessionSchemaBinding,
)


@dataclass(frozen=True)
class EvidencePaths:
    root: Path
    run_id: str
    @property
    def run(self): return self.root / f"run_id={self.run_id}"


class EvidenceLake:
    def __init__(self, root: str | Path, run_id: str):
        self.paths=EvidencePaths(Path(root),run_id);self.paths.run.mkdir(parents=True,exist_ok=True)
        self._lock=Lock()

    def bind_session(self,session_id:str,binding:SessionSchemaBinding|None=None)->SessionSchemaBinding:
        binding=binding or SessionSchemaBinding(session_id=session_id,schema_id=COVERAGE_SCHEMA.schema_id,
                                                schema_version=COVERAGE_SCHEMA.schema_version)
        path=self.paths.run/f"session_id={session_id}"/"schema-binding.json";path.parent.mkdir(parents=True,exist_ok=True)
        body=binding.model_dump_json(indent=2)
        with self._lock:
            if path.exists():
                existing=SessionSchemaBinding.model_validate_json(path.read_text())
                if existing != binding:
                    from .binding_compat import BindingMigrationRequired,BindingIdentityMismatch,identity
                    if identity(existing)==identity(binding):raise BindingMigrationRequired(path,existing,binding)
                    raise BindingIdentityMismatch()
            else:
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'w') as f:f.write(body);f.flush();os.fsync(f.fileno())
        return binding

    def append_evidence(self,record:EvidenceRecord)->Path:
        return self._append("evidence",record.model_dump(mode="json"))

    def append_dead_letter(self,entry:DeadLetterEntry)->Path:
        return self._append("dlq",entry.model_dump(mode="json"))

    def append_promotion(self,review:PromotionReview)->Path:
        if review.status not in (PromotionStatus.ACCEPTED,PromotionStatus.REJECTED,PromotionStatus.RELEASED):
            raise ValueError("promotion log accepts reviewed decisions only")
        return self._append("promotions",review.model_dump(mode="json"))

    def _append(self,stream:str,row:dict)->Path:
        from .stop_fence import guarded_publish
        return guarded_publish(lambda:self._append_unfenced(stream,row))

    def _append_unfenced(self,stream:str,row:dict)->Path:
        directory=self.paths.run/stream;directory.mkdir(parents=True,exist_ok=True)
        line=json.dumps(row,separators=(",",":"),sort_keys=True,default=str)
        with self._lock:
            raw=directory/"events.jsonl"
            with raw.open('a') as f:f.write(line+'\n');f.flush();os.fsync(f.fileno())
            part=directory/f"part-{uuid4()}.parquet"
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table=pa.Table.from_pylist([{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in row.items()}])
                tmp=part.with_suffix('.tmp');pq.write_table(table,tmp,compression='zstd');tmp.replace(part)
            except Exception as exc:
                # Raw JSONL is authoritative. Parquet is a rebuildable
                # analytical projection and cannot block interview control flow.
                marker=directory/"parquet-errors.jsonl"
                try:
                    with marker.open('a') as f:
                        f.write(json.dumps({'part':part.name,'error_type':type(exc).__name__,
                                            'error':str(exc)})+'\n')
                        f.flush();os.fsync(f.fileno())
                except OSError as marker_exc:
                    log.error("parquet projection and degradation log failed stream=%s part=%s projection_error=%s marker_error=%s",
                              stream,part.name,exc,marker_exc,exc_info=True)
                else:
                    log.warning("parquet projection degraded to authoritative JSONL stream=%s part=%s error=%s: %s",
                                stream,part.name,type(exc).__name__,exc)
                return raw
        return part


def record_founder_classification(lake: EvidenceLake, *, session_id: str, sequence: int,
                                  raw_text: str, target: str | None, result) -> tuple[EvidenceRecord, DeadLetterEntry | None]:
    """Persist one complete founder answer and route an unbucketed answer to DLQ.

    This observes classifier output only. It never updates graph coverage or a
    facet schema; graph code remains the sole owner of accumulation/state.
    """
    from datetime import datetime, timezone
    from vmodel_contracts.facets import DeadLetterReason, EvidenceDisposition
    answers=result.answers
    matched=[key.removeprefix('covers_') for key,value in answers.items()
             if key.startswith('covers_') and float(value.value)>=.5]
    now=datetime.now(timezone.utc).isoformat();eid=uuid4()
    disposition=EvidenceDisposition.CLASSIFIED if matched else EvidenceDisposition.DEAD_LETTER
    confidence=max((value.confidence for key,value in answers.items()
                    if key.startswith('covers_') and value.confidence is not None),default=None)
    record=EvidenceRecord(evidence_id=eid,run_id=lake.paths.run_id,session_id=session_id,sequence=sequence,
        schema_id=COVERAGE_SCHEMA.schema_id,schema_version=COVERAGE_SCHEMA.schema_version,
        facet_key=matched[0] if len(matched)==1 else target,option_id=30 if matched else None,
        disposition=disposition,raw_text=raw_text,
        extracted_value={'matched_facets':matched,'verdict':{k:v.value for k,v in answers.items()}},
        provider=result.provider,confidence=confidence,occurred_at=now)
    lake.append_evidence(record)
    dead=None
    if not matched:
        dead=DeadLetterEntry(dlq_id=uuid4(),evidence_id=eid,run_id=lake.paths.run_id,session_id=session_id,
            schema_id=COVERAGE_SCHEMA.schema_id,schema_version=COVERAGE_SCHEMA.schema_version,
            reason=DeadLetterReason.NO_MATCH,raw_text=raw_text,proposed_facet_key=None,created_at=now)
        lake.append_dead_letter(dead)
    return record,dead
