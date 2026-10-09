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
