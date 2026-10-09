import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from vmodel_contracts.interactive import StartSessionRequest,InputRequest,InputKind,InputResponse,ProcessStatus
from interview_engine.http_service import SessionRegistry,ProtocolFailure


def test_registry_stop_is_durable_nonblocking_and_discards_late_pending(tmp_path):
    entered=threading.Event();release=threading.Event()
    class Program:
        finished=False
        def advance(self,response=None):
            if response is not None:entered.set();release.wait(2)
            return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='fake')
    registry=SessionRegistry(lambda *a:Program(),state_dir=tmp_path)
    session,_=registry.start(StartSessionRequest(task_id='fake',idea='fake',model_route='local'))
    answer=InputResponse(input_id=session.open_input.input_id,value='fake',actor_id='fake')
    with ThreadPoolExecutor() as pool:
        future=pool.submit(registry.respond,session.session_id,answer)
        assert entered.wait(1)
        state=registry.stop(session.session_id)
        assert state.process_status==ProcessStatus.STOP_REQUESTED and state.open_input is None
        release.set();_,state=future.result()
    assert state.process_status==ProcessStatus.STOPPED and state.open_input is None
    assert registry.snapshot(session.session_id).process_status==ProcessStatus.STOPPED
    with pytest.raises(ProtocolFailure):registry.respond(session.session_id,answer)
    restored=SessionRegistry(lambda *a:pytest.fail('no auto restore'),state_dir=tmp_path)
    # Restart itself does not prove the previous worker is gone.
    assert restored.snapshot(session.session_id).process_status==ProcessStatus.STOP_REQUESTED
