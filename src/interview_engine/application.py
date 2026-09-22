"""Application service for one HTTP-driven interview session.

No console or transport concerns live here. The service advances the graph until
it reaches a typed founder-input boundary or completion.
"""
from __future__ import annotations
import json,sqlite3
from pathlib import Path
from uuid import uuid4
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from vmodel_contracts import AppArchetype,ConOpsDocument,IntakeSubmission,IntentModel
from vmodel_contracts.facets import COVERAGE_SCHEMA
from vmodel_contracts.interactive import InputKind,InputOption,InputRequest,InputResponse,NextStep,ProcessStatus
from vmodel_contracts.runtime import artifacts_root
from .conops_stage import founder_decision,render_conops
from .graph import build_graph
from .progress import ProgressBus,stderr_printer
from .telemetry import UsageLog,jsonl_sink

class InterviewApplication:
    """Stateful graph driver. HTTP supplies typed responses; graph owns domain state."""
    def __init__(self,session_id:str,task_id:str,idea:str,*,artifact_root:str|None=None):
        self.session_id,self.task_id=session_id,task_id
        self.outdir=artifacts_root(artifact_root)/session_id/'interview';self.outdir.mkdir(parents=True,exist_ok=True)
        from .evidence import EvidenceLake
        self.evidence=EvidenceLake(artifacts_root(artifact_root)/'evidence-lake',task_id)
        self.schema_binding=self.evidence.bind_session(session_id)
        db=self.outdir/'checkpoints.sqlite';conn=sqlite3.connect(str(db),check_same_thread=False)
        saver=SqliteSaver(conn,serde=JsonPlusSerializer(allowed_msgpack_modules=[AppArchetype]))
        log=UsageLog(session_id,sink=jsonl_sink(self.outdir/'usage.jsonl'))
        progress=ProgressBus();progress.subscribe(stderr_printer())
        self.graph=build_graph(log,checkpointer=saver,progress=progress,conops_enabled=True,raw_output_dir=self.outdir/'debug',
            session_id=session_id,schema_id=self.schema_binding.schema_id,schema_version=self.schema_binding.schema_version,evidence_lake=self.evidence)
        self.cfg={'configurable':{'thread_id':session_id}}
        submission=IntakeSubmission(free_text=idea,founder_id='http-founder',submission_id=session_id)
        self.state={'messages':[HumanMessage(content=idea)],'submission':submission.model_dump(mode='json')}
        self.result=None;self.open_input=None;self.finished=False

    def advance(self,response:InputResponse|None=None):
        if response is not None:
            if self.open_input is None or response.input_id!=self.open_input.input_id:raise ValueError('stale input')
            value=response.value
            if self.open_input.kind==InputKind.APPROVAL:
                if value == 1:self._finish();return None
                if value == 0:raise ValueError('intent rejection requires a new interview generation')
                raise ValueError('approval response must use a declared option value')
            answer=str(value)
            if self.open_input.rationale_required:
                if not response.rationale:raise ValueError('typed input requires a rationale')
                answer=f"{value} - {response.rationale}"
            self.state={'messages':[HumanMessage(content=answer)]}
        while not self.finished:
            self.result=self.graph.invoke(self.state,self.cfg);self.state=None;self.open_input=None
            verdict=self.result.get('verdict')
            if verdict and not verdict['accepted']:
                self.finished=True;raise ValueError(verdict['decline_message'])
            if 'conops' in self.result:
                self.open_input=InputRequest(input_id=uuid4(),kind=InputKind.APPROVAL,prompt='Approve this intent model?',
                    options=[InputOption(value=1,label='Approve'),InputOption(value=0,label='Reject')],context={'artifact':'intent_model.json'})
                return self.open_input
            last=self.result['messages'][-1].content
            if '[INTERVIEW COMPLETE]' in last:self.state={};continue
            if self.result.get('conops_draft'):
                decision=founder_decision(self.result['conops_draft'])
                self.open_input=InputRequest(input_id=uuid4(),kind=InputKind.CHOICE,prompt=decision['prompt'],
                    options=[InputOption.model_validate(o) for o in decision['options']],rationale_required=decision.get('rationale_required',False))
            else:self.open_input=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt=last)
            return self.open_input
        return None

    def _finish(self):
        intent=IntentModel.model_validate(self.result['intent_model']);conops=ConOpsDocument.model_validate(self.result['conops'])
        (self.outdir/'intent_model.json').write_text(intent.model_dump_json(indent=2));(self.outdir/'conops.json').write_text(conops.model_dump_json(indent=2))
        (self.outdir/'conops.md').write_text(render_conops(conops));self.finished=True;self.open_input=None
