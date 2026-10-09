"""Private installed-lineage control reader. No session start/input route wiring.

Expected anchor comes from authenticated installed control-plane records, never a
request body/file inference. Preserved interview source cannot accept new input.
"""
from dataclasses import dataclass
from pathlib import Path
from vmodel_contracts.execution_anchor import ExecutionAnchor,read_execution_anchor
from .stop_fence import StopFence


@dataclass(frozen=True)
class InstalledExecutionControl:
    anchor:ExecutionAnchor
    anchor_sha256:str
    root:Path
    fence:StopFence

    def validate(self):
        receipt=read_execution_anchor(self.root,expected=self.anchor)
        if receipt['document_sha256']!=self.anchor_sha256:raise ValueError('installed engine anchor bytes changed')
        return receipt

    def reject_interview_mutation(self):
        self.validate()
        raise ValueError('lineage execution preserves approved interview source; new interview input is unavailable')


def load_installed_execution_control(root,*,expected_anchor,expected_anchor_sha256):
    if not isinstance(expected_anchor,ExecutionAnchor):raise ValueError('trusted installed engine anchor required')
    root=Path(root);receipt=read_execution_anchor(root,expected=expected_anchor)
    if receipt['document_sha256']!=expected_anchor_sha256:raise ValueError('installed engine anchor differs from database digest')
    fence=StopFence(root/'interview'/'stop-request.json',execution_anchor=expected_anchor)
    return InstalledExecutionControl(expected_anchor,expected_anchor_sha256,root,fence)


class InstalledExecutionRegistry:
    """Private owner-scoped control registry, never installed by the HTTP factory.

    Reader must return current authenticated DB binding on every access. No
    request/body/path inference, interview graph restore, publication or start.
    """
    def __init__(self,base,*,read_binding):
        if not callable(read_binding):raise ValueError('authenticated installed binding reader required')
        self.base=Path(base).resolve();self.read_binding=read_binding

    def get(self,execution_session_id):
        from .tenant_worker import current_owner
        from uuid import UUID
        import re
        owner=current_owner()
        if owner is None or str(UUID(owner))!=owner:raise PermissionError('verified installed owner scope required')
        if not isinstance(execution_session_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',execution_session_id):
            raise ValueError('safe installed execution namespace required')
        binding=self.read_binding(owner,execution_session_id)
        if not isinstance(binding,dict) or set(binding)!={'anchor','anchor_sha256'}:raise PermissionError('current installed database binding required')
        anchor=binding['anchor']
        if not isinstance(anchor,ExecutionAnchor) or anchor.owner_id!=owner or anchor.execution_session_id!=execution_session_id:
            raise PermissionError('installed registry owner/execution mismatch')
        root=self.base/'tenants'/owner/'sessions'/execution_session_id
        if root.resolve()!=root:raise PermissionError('unsafe installed control namespace')
        return load_installed_execution_control(root,expected_anchor=anchor,expected_anchor_sha256=binding['anchor_sha256'])

    def stop(self,execution_session_id):
        control=self.get(execution_session_id)
        with control.fence.marker_guard():
            # Re-read revocation/hash evidence immediately before writing Stop.
            latest=self.get(execution_session_id)
            if latest.anchor!=control.anchor or latest.anchor_sha256!=control.anchor_sha256:
                raise PermissionError('installed binding changed before Stop')
            return control.fence.request_stop()

    def reconcile_stopped(self,execution_session_id):
        control=self.get(execution_session_id)
        if not control.fence.requested:raise ValueError('durable installed Stop required')
        return control.fence.confirm_quiescent_after_recovery(no_live_worker=True)

    def respond(self,execution_session_id,response):
        self.get(execution_session_id).reject_interview_mutation()
