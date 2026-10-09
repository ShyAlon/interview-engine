import hashlib,json
import pytest
from vmodel_contracts.execution_anchor import ExecutionAnchor
from interview_engine.installed_execution import load_installed_execution_control
from interview_engine.stop_fence import StopRequested


def fixture(tmp_path):
    anchor=ExecutionAnchor('00000000-0000-0000-0000-000000000001','task','execution2','semantic',2,'a'*64,'test-install')
    root=tmp_path/'tenants'/anchor.owner_id/'sessions'/anchor.execution_session_id;(root/'interview').mkdir(parents=True)
    raw=json.dumps(anchor.document(),sort_keys=True,indent=2).encode();(root/'execution-generation.json').write_bytes(raw)
    return root,anchor,hashlib.sha256(raw).hexdigest()


def test_installed_engine_control_preserves_semantic_id_and_blocks_interview_mutation(tmp_path):
    root,anchor,sha=fixture(tmp_path)
    control=load_installed_execution_control(root,expected_anchor=anchor,expected_anchor_sha256=sha)
    assert control.fence.generation==2 and control.anchor.semantic_session_id=='semantic'
    with pytest.raises(ValueError,match='new interview input'):control.reject_interview_mutation()
    control.fence.request_stop()
    assert control.fence.generation==3
    with pytest.raises(StopRequested):control.fence.check(2)


def test_engine_control_never_infers_anchor_or_ignores_hash_drift(tmp_path):
    root,anchor,sha=fixture(tmp_path)
    with pytest.raises(ValueError):load_installed_execution_control(root,expected_anchor=None,expected_anchor_sha256=sha)
    with pytest.raises(ValueError):load_installed_execution_control(root,expected_anchor=anchor,expected_anchor_sha256='b'*64)
    control=load_installed_execution_control(root,expected_anchor=anchor,expected_anchor_sha256=sha)
    (root/'execution-generation.json').write_text(json.dumps(anchor.document()))
    with pytest.raises(ValueError,match='bytes changed'):control.validate()


def test_private_registry_reads_owner_binding_each_time_and_never_restores_graph(tmp_path):
    from interview_engine.installed_execution import InstalledExecutionRegistry
    from interview_engine.tenant_worker import _owner
    root,anchor,sha=fixture(tmp_path);active=[True];seen=[]
    def read(owner,sid):
        seen.append((owner,sid))
        return {'anchor':anchor,'anchor_sha256':sha} if active[0] else None
    registry=InstalledExecutionRegistry(tmp_path,read_binding=read)
    with pytest.raises(PermissionError):registry.get(anchor.execution_session_id)
    token=_owner.set(anchor.owner_id)
    try:
        assert registry.get(anchor.execution_session_id).fence.generation==2
        with pytest.raises(ValueError,match='new interview input'):registry.respond(anchor.execution_session_id,{})
        assert registry.stop(anchor.execution_session_id)=='STOPPED'
        assert registry.reconcile_stopped(anchor.execution_session_id)=='STOPPED'
        assert len(seen)>=5
        assert not (root/'interview/http-session.json').exists()
        active[0]=False
        with pytest.raises(PermissionError):registry.get(anchor.execution_session_id)
    finally:_owner.reset(token)


def test_registry_foreign_owner_and_namespace_fail_before_control(tmp_path):
    from interview_engine.installed_execution import InstalledExecutionRegistry
    from interview_engine.tenant_worker import _owner
    root,anchor,sha=fixture(tmp_path)
    registry=InstalledExecutionRegistry(tmp_path,read_binding=lambda *a:{'anchor':anchor,'anchor_sha256':sha})
    token=_owner.set('00000000-0000-0000-0000-000000000002')
    try:
        with pytest.raises(PermissionError,match='owner/execution'):registry.get(anchor.execution_session_id)
        with pytest.raises(ValueError):registry.get('../escape')
        assert not (root/'interview/stop-request.json').exists()
    finally:_owner.reset(token)


def test_registry_revoked_between_reads_never_writes_stop(tmp_path):
    from interview_engine.installed_execution import InstalledExecutionRegistry
    from interview_engine.tenant_worker import _owner
    root,anchor,sha=fixture(tmp_path);calls=[]
    def read(*a):
        calls.append(1)
        return {'anchor':anchor,'anchor_sha256':sha} if len(calls)==1 else None
    registry=InstalledExecutionRegistry(tmp_path,read_binding=read)
    token=_owner.set(anchor.owner_id)
    try:
        with pytest.raises(PermissionError):registry.stop(anchor.execution_session_id)
        assert not (root/'interview/stop-request.json').exists()
    finally:_owner.reset(token)
