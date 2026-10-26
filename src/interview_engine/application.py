"""Application service for one HTTP-driven interview session.

No console or transport concerns live here. The service advances the graph until
it reaches a typed founder-input boundary or completion.
"""
from __future__ import annotations
import json,logging,sqlite3,time
from contextlib import nullcontext, contextmanager
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
from . import session_models, model_config

_log=logging.getLogger("interview_engine.application")


class InterviewApplication:
    """Stateful graph driver. HTTP supplies typed responses; graph owns domain state."""
    def __init__(self,session_id:str,task_id:str,idea:str,*,artifact_root:str|None=None,model_route:str|None=None,pinned_models:dict|None=None,initial_model:str|None=None,pg_store=None,pg_claim=None):
        self.produced_checkpoint_id=None
        self.session_id,self.task_id=session_id,task_id
        from .session_runtime import SessionRuntime
        self.runtime=SessionRuntime()
        self.model_route=model_route
        self.model_map=pinned_models if pinned_models is not None else session_models.resolve(model_route,initial_model) if model_route else None
        if self.model_map is not None:session_models.validate(self.model_map,model_route,preflight=pinned_models is None)
        self.outdir=artifacts_root(artifact_root)/session_id/'interview';self.outdir.mkdir(parents=True,exist_ok=True)
        from .evidence import EvidenceLake
        self.evidence=EvidenceLake(artifacts_root(artifact_root)/'evidence-lake',task_id)
        self.schema_binding=self.evidence.bind_session(session_id)
        if pg_store is not None:
            # Database mode: the graph checkpoints into Postgres through the
            # fenced saver; every mutation carries the session's lease fence.
            # Tenant claims ride each transaction (pg_claims): poolers drop
            # startup-option GUCs, so the conninfo never carries them.
            import psycopg
            from .pg_checkpoint import FencedPostgresSaver, LeaseFence
            from .pg_claims import connect_claimed
            t_pg=time.monotonic();_log.info("pg.connect.begin session=%s",session_id)
            pg_conn=connect_claimed(pg_store.conn.info.dsn,pg_claim.owner_id,pg_claim.session_id,autocommit=True)
            _log.info("pg.connect.end session=%s dur_ms=%d",session_id,int((time.monotonic()-t_pg)*1000))
            from . import model_config as _mc
            self._staging=_mc.StagingContext(session_id,pg_store,pg_claim)
            _log.info("staging.ready session=%s",session_id)
            saver=FencedPostgresSaver(pg_conn,
                lease=LeaseFence(session_id,pg_claim.generation,pg_claim.epoch),
                serde=JsonPlusSerializer(allowed_msgpack_modules=[AppArchetype]),
                claims=(pg_claim.owner_id,pg_claim.session_id))
            t_setup=time.monotonic();_log.info("saver.setup.begin session=%s",session_id)
            saver.setup()
            _log.info("saver.setup.end session=%s dur_ms=%d",session_id,int((time.monotonic()-t_setup)*1000))
        else:
            db=self.outdir/'checkpoints.sqlite';conn=sqlite3.connect(str(db),check_same_thread=False)
            from .stop_fence import check_execution,guarded_publish
            class FencedSqliteSaver(SqliteSaver):
                def put(self,*args,**kwargs):
                    return guarded_publish(lambda:super(FencedSqliteSaver,self).put(*args,**kwargs))
                def put_writes(self,*args,**kwargs):
                    return guarded_publish(lambda:super(FencedSqliteSaver,self).put_writes(*args,**kwargs))
            saver=FencedSqliteSaver(conn,serde=JsonPlusSerializer(allowed_msgpack_modules=[AppArchetype]))
        log=UsageLog(session_id,sink=jsonl_sink(self.outdir/'usage.jsonl'))
        self.usage_log=log
        progress=ProgressBus();progress.subscribe(stderr_printer())
        if pg_store is not None:
            from .pg_journal import BufferedJournal
            self.progress_journal=BufferedJournal(pg_store,pg_claim,session_id,task_id)
        else:
            from .progress_journal import ProgressJournal
            self.progress_journal=ProgressJournal(artifacts_root(artifact_root),session_id,task_id)
        self._pg_store=pg_store;self._pg_claim=pg_claim
        if not hasattr(self,'_staging'):self._staging=None
        def journal_signal(ev):
            # The interview graph emits intent and ConOps stages through one bus.
            phase = "conops" if "conops" in ev.stage.lower() or "concept" in ev.stage.lower() else "intent"
            self.progress_journal.append(phase_key=phase,stage=ev.stage,label=ev.label,
                status=ev.status if ev.status in ("started","completed","failed","heartbeat","item_started","item_completed") else "heartbeat",detail=ev.detail,completed=ev.completed,total=ev.total)
        progress.subscribe(journal_signal)
        def heartbeat_signal(stage, label, elapsed):
            self.progress_journal.append(phase_key="conops" if "conops" in stage.lower() else "intent",
                stage=stage,label=label,status="heartbeat",detail=f"Model call still running, {elapsed:.0f}s elapsed")
        self._heartbeat_signal=heartbeat_signal
        with self._runtime_context():
            from .verification import VerificationLog
            self.graph=build_graph(log,checkpointer=saver,progress=progress,verification_log=VerificationLog(self.outdir/'verification.jsonl'),conops_enabled=True,raw_output_dir=self.outdir/'debug',
                session_id=session_id,schema_id=self.schema_binding.schema_id,schema_version=self.schema_binding.schema_version,evidence_lake=self.evidence)
        self.checkpointer=saver;self.saver=saver
        self.cfg={'configurable':{'thread_id':session_id}}
        submission=IntakeSubmission(free_text=idea,founder_id='http-founder',submission_id=session_id)
        self.state={'messages':[HumanMessage(content=idea)],'submission':submission.model_dump(mode='json')}
        self.result=None;self.open_input=None;self.finished=False

    @contextmanager
    def _runtime_context(self):
        token=model_config._session_llm_log.set(str(self.outdir/'llm-calls.jsonl'))
        # PR B staged provider responses: DB-backed sessions stage every usable
        # provider response durably before interpretation (replay, never rebill).
        staging_token=None
        # _staging is optional (None outside DB-backed sessions) and absent on
        # instances built without __init__ (isolated unit constructions).
        if getattr(self, '_staging', None) is not None:
            staging_token=model_config._response_staging.set(self._staging)
        heartbeat_token=model_config._progress_heartbeat.set(self._heartbeat_signal)
        from .stop_fence import _stopped_usage
        def stopped_usage(response,*,stage,provider,model):
            usage=getattr(response,'usage_metadata',None)
            if not isinstance(usage,dict):
                # Missing usage must remain explicit operator evidence, not 0.
                model_config._log_llm({'event':'stopped_usage_unknown','stage':stage,'provider':provider,'model':model})
                return
            self.usage_log.record(stage,provider,model,usage)
        usage_token=_stopped_usage.set(stopped_usage)
        try:
            from .session_runtime import use
            with use(self.runtime), session_models.use(self.model_map) if self.model_map is not None else nullcontext():
                yield
        finally:
            if staging_token is not None:model_config._response_staging.reset(staging_token)
            _stopped_usage.reset(usage_token)
            model_config._progress_heartbeat.reset(heartbeat_token)
            model_config._session_llm_log.reset(token)

    def advance(self,response:InputResponse|None=None):
        from .stop_fence import check_execution
        check_execution()
        _log.info("advance.enter session=%s finished=%s response=%s",self.session_id,self.finished,'yes' if response is not None else 'no')
        rework_backup=None;rework_input=None
        if response is not None:
            if self.open_input is None or response.input_id!=self.open_input.input_id:raise ValueError('stale input')
            value=response.value
            if self.open_input.context.get('decision_purpose')=='semantic_review':
                from vmodel_contracts.semantic_review import SemanticReview
                original=SemanticReview.model_validate(self.result['semantic_review'])
                payload=json.loads(str(value))
                if payload.get('revision')!=original.revision:raise ValueError('semantic review changed')
                reviewed=SemanticReview.model_validate({**original.model_dump(),'decisions':payload.get('decisions',[])})
                if any(d.actor!=response.actor_id for d in reviewed.decisions):raise ValueError('semantic decision actor mismatch')
                resolved=reviewed.resolved_candidate(IntentModel.model_validate)
                self.graph.update_state(self.cfg,{'intent_model':resolved,'semantic_review':reviewed.model_dump(mode='json'),'semantic_review_resolved':True})
                __import__('interview_engine.stop_fence',fromlist=['guarded_publish']).guarded_publish(lambda:(self.outdir/'semantic-review.json').write_text(reviewed.model_dump_json(indent=2)))
                self.result={**self.result,'intent_model':resolved,'semantic_review':reviewed.model_dump(mode='json'),'semantic_review_resolved':True}
                self.open_input=None;self.state={}
            if self.open_input is not None and self.open_input.kind==InputKind.APPROVAL:
                if response.founder_inputs and value in (0,1):
                    # Founder answers to open inputs are recorded for both approval and rework.
                    from vmodel_contracts.founder_register import FounderRegister,apply_resolutions
                    register=FounderRegister.model_validate(self.result['intent_model']['founder_register'])
                    updated=apply_resolutions(register,response.founder_inputs,response.actor_id)
                    self.result['intent_model']={**self.result['intent_model'],'founder_register':updated.model_dump(mode='json')}
                    self.graph.update_state(self.cfg,{'intent_model':self.result['intent_model']})
                if value == 1:
                    self._finish();return None
                if value == 0:
                    # Archive before changing the checkpoint. Rework ConOps only;
                    # the reviewed intent and its source transcript remain intact.
                    archive=self.outdir/'review-history';archive.mkdir(exist_ok=True)
                    from .stop_fence import guarded_publish
                    guarded_publish(lambda:(archive/f'{self.open_input.input_id}.json').write_text(json.dumps(
                        {'intent_model':self.result['intent_model'],'conops':self.result['conops'],
                         'guidance':response.rationale or ''},ensure_ascii=False,indent=2)))
                    rework_backup=dict(self.result);rework_input=self.open_input
                    self.graph.update_state(self.cfg,{'conops':None,'conops_draft':None,'conops_decision_attempts':0})
                    self.state={'messages':[HumanMessage(content=response.rationale or 'Regenerate the ConOps concepts for review.')]}
                    self.open_input=None
                elif value != 1:raise ValueError('approval response must use a declared option value')
            answer=str(value)
            if self.open_input is not None and self.open_input.rationale_required:
                if not response.rationale:raise ValueError('typed input requires a rationale')
                answer=f"{value} - {response.rationale}"
            if self.open_input is not None:self.state={'messages':[HumanMessage(content=answer)]}
        while not self.finished:
            try:
                with self._runtime_context():
                    if getattr(self, '_staging', None) is not None:
                        # Anchor for this advance: the pinned checkpoint when
                        # restored, else the session's published pointer
                        # ('bootstrap' first). The super-step half of the
                        # execution point is read from the pregel scratchpad at
                        # call time - deterministic given the anchor, and immune
                        # to the async-checkpoint commit race.
                        anchor = self.cfg['configurable'].get('checkpoint_id')
                        if anchor is None:
                            published = self._pg_store.load_published(self.session_id, self._pg_claim.owner_id)
                            anchor = published['published_checkpoint_id'] if published else None
                        self._staging.set_execution_point(anchor)
                    # 'exit' durability: the turn's checkpoints are durable
                    # when invoke returns (default 'async' commits in the
                    # background, so the step publish could run before the
                    # turn's final checkpoint landed - observed 2026-10-23 as
                    # the published pointer predating the turn's state).
                    t_invoke=time.monotonic();_log.info("advance.turn.begin session=%s",self.session_id)
                    self.result=self.graph.invoke(self.state,self.cfg,durability='exit')
                    _log.info("advance.turn.end session=%s dur_ms=%d",self.session_id,int((time.monotonic()-t_invoke)*1000))
                    # Publish provenance: record the checkpoint this advance
                    # produced (its writes are durable at 'exit'). The step
                    # publish references exactly this checkpoint - a
                    # saver-latest read at publish time could otherwise select
                    # an unpublished orphan committed by a crashed prior
                    # epoch's turn.
                    if hasattr(self.graph,'get_state'):
                        cfg_view=self.cfg.get('configurable',{}) if isinstance(self.cfg,dict) else {}
                        latest=self.graph.get_state({'configurable':{'thread_id':cfg_view.get('thread_id'),
                            'checkpoint_ns':cfg_view.get('checkpoint_ns','')}})
                        snap_cfg=(getattr(latest,'config',None) or {}).get('configurable',{}) if latest is not None else {}
                        # getattr: unit constructions (__new__, no __init__) carry no attribute.
                        self.produced_checkpoint_id=snap_cfg.get('checkpoint_id') or getattr(self,'produced_checkpoint_id',None)
                    # Continue from what this turn produced: a restored cfg is
                    # pinned to the restore anchor, and invoke resumes a pinned
                    # config from that checkpoint - so without advancing the
                    # pin, every later turn on a reclaimed host replays from
                    # the anchor (2026-10-23: conops extraction replayed per
                    # answer until the LLM call budget stopped the run). A
                    # fresh host's unpinned cfg gets the same resume-latest
                    # semantics; a failed invoke leaves the pin at the last
                    # produced checkpoint, so a retry replays only that turn.
                    if getattr(self,'produced_checkpoint_id',None) and isinstance(self.cfg,dict):
                        self.cfg.setdefault('configurable',{})['checkpoint_id']=self.produced_checkpoint_id
                    check_execution()
            except Exception:
                check_execution()  # Stop must never restore a previous actionable input.
                if rework_backup is not None:
                    # Registry restores the old approval on failure. Restore its
                    # checkpoint too, so a retry/restart cannot accept a cleared
                    # or partly regenerated ConOps under that old review ID.
                    if hasattr(self.graph,'get_state'):
                        # Checkpoint restore requires a real compiled graph; stub
                        # graphs (unit constructions) carry no checkpoints.
                        from langchain_core.messages import RemoveMessage
                        current=self.graph.get_state(self.cfg).values
                        old_ids={m.id for m in rework_backup.get('messages',[]) if getattr(m,'id',None)}
                        remove=[RemoveMessage(id=m.id) for m in current.get('messages',[])
                                if getattr(m,'id',None) and m.id not in old_ids]
                        restored_cfg=self.graph.update_state(self.cfg,{'conops':rework_backup.get('conops'),
                            'conops_draft':rework_backup.get('conops_draft'),
                            'conops_decision_attempts':rework_backup.get('conops_decision_attempts',0),
                            'messages':remove+rework_backup.get('messages',[])})
                        # The rollback write is this owner's produced state; the
                        # failed rework's checkpoints are unpublished orphans.
                        if isinstance(restored_cfg,dict):
                            self.produced_checkpoint_id=(restored_cfg.get('configurable') or {}).get('checkpoint_id') or getattr(self,'produced_checkpoint_id',None)
                            if getattr(self,'produced_checkpoint_id',None) and isinstance(self.cfg,dict):
                                self.cfg.setdefault('configurable',{})['checkpoint_id']=self.produced_checkpoint_id
                    self.result=rework_backup;self.open_input=rework_input
                    self.state=None
                raise
            self.state=None;self.open_input=None
            verdict=self.result.get('verdict')
            if verdict and not verdict['accepted']:
                self.finished=True;raise ValueError(verdict['decline_message'])
            if self.result.get('semantic_review') and not self.result.get('semantic_review_resolved'):
                review=self.result['semantic_review']
                self.open_input=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Review these interview findings before continuing.',
                    context={'decision_purpose':'semantic_review','evidence_version':1,'session_id':self.session_id,
                             'evidence_revision':__import__('hashlib').sha256(json.dumps({'semantic_review':review},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
                             'evidence':{'semantic_review':review}})
                _log.info("advance.return session=%s kind=review",self.session_id)
                return self.open_input
            if self.result.get('conops'):
                if not self.result['intent_model'].get('founder_register'):
                    from .completeness import inventory
                    register=inventory(self.session_id,self.result['intent_model'],self.result.get('messages',[]),self.result['conops'])
                    self.result['intent_model']={**self.result['intent_model'],'founder_register':register}
                    self.graph.update_state(self.cfg,{'intent_model':self.result['intent_model']})
                self.open_input=InputRequest(input_id=uuid4(),kind=InputKind.APPROVAL,prompt='Approve this intent model?',
                    options=[InputOption(value=1,label='Proceed'),InputOption(value=0,label='Edit')],
                    context=self._decision_context('interview_approval'))
                _log.info("advance.return session=%s kind=approval",self.session_id)
                return self.open_input
            last=self.result['messages'][-1].content
            if '[INTERVIEW COMPLETE]' in last:self.state={};continue
            if self.result.get('conops_draft'):
                decision=founder_decision(self._decision_context('conops_concept')['evidence']['conops_draft'])
                self.open_input=InputRequest(input_id=uuid4(),kind=InputKind.CHOICE,prompt=decision['prompt'],
                    options=[InputOption.model_validate(o) for o in decision['options']],rationale_required=decision.get('rationale_required',False),
                    context=self._decision_context('conops_concept'))
            else:self.open_input=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt=last)
            _log.info("advance.return session=%s kind=question",self.session_id)
            return self.open_input
        _log.info("advance.return session=%s kind=none finished=%s result_keys=%s",self.session_id,self.finished,sorted(self.result.keys()) if isinstance(self.result,dict) else type(self.result).__name__)
        return None

    def _decision_context(self,purpose:str)->dict:
        """Code-owned, bounded review evidence. Never infer purpose from prompts."""
        import hashlib
        if purpose=='interview_approval':
            evidence={'intent_model':IntentModel.model_validate(self.result['intent_model']).model_dump(mode='json'),
                      'conops':ConOpsDocument.model_validate(self.result['conops']).model_dump(mode='json')}
        else:
            from .conops_stage import validate_draft
            founder_turns={i for i,m in enumerate(self.result.get('messages',[])) if getattr(m,'type',None)=='human'}
            evidence={'conops_draft':validate_draft(self.result['conops_draft'],founder_turns)}
        raw=json.dumps(evidence,sort_keys=True,separators=(',',':'),ensure_ascii=False)
        if len(raw.encode('utf-8'))>1_000_000:
            raise ValueError('decision review evidence exceeds 1 MB')
        return {'decision_purpose':purpose,'evidence_version':1,'session_id':self.session_id,
                'evidence_revision':hashlib.sha256(raw.encode('utf-8')).hexdigest(),'evidence':evidence}

    def _finish(self):
        from .stop_fence import guarded_publish
        return guarded_publish(self._finish_outputs)

    def _finish_outputs(self):
        intent=IntentModel.model_validate(self.result['intent_model']);conops=ConOpsDocument.model_validate(self.result['conops'])
        if intent.founder_register:
            (self.outdir/'founder-decisions.json').write_text(intent.founder_register.model_dump_json(indent=2))
        (self.outdir/'intent_model.json').write_text(intent.model_dump_json(indent=2));(self.outdir/'conops.json').write_text(conops.model_dump_json(indent=2))
        (self.outdir/'conops.md').write_text(render_conops(conops));self.finished=True;self.open_input=None
