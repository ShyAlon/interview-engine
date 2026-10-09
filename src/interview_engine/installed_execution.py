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
