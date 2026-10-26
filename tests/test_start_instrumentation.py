"""Start-path triage instrumentation for the 2026-10-25 "Starting Interview" stall.

The live stall (Meditator 5 and 6) shows a completed first turn followed by
total silence: no question surfaces, no error logs. The start path emitted
almost nothing between construction and return. These tests pin the marker
trail the instrumentation must emit - a stall localizes to the span whose
begin marker has no matching end marker."""
import logging
from vmodel_contracts.interactive import ProcessStatus
from test_lease_heartbeat_fence import FakePgStore,FakeProgram,registry,start

LOGGER="interview_engine.http"

def messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name==LOGGER]


def test_happy_path_emits_ordered_start_trail(caplog):
    store=FakePgStore()
    with caplog.at_level("DEBUG"):
        session,_=start(registry(store,FakeProgram()))
    msgs=messages(caplog)
    def idx(token):
        hits=[i for i,m in enumerate(msgs) if m.startswith(token)]
        assert hits,"missing log marker %s in %s"%(token,msgs)
        return hits[0]
    order=["session.start.accept","pg.registry.created","pg.lease.claimed",
        "session.published","program.construct.begin","program.construct.end",
        "session.saved","advance.begin","advance.end","control_commit.enter",
        "open_input.set","control_commit.exit","start.return"]
    positions=[idx(t) for t in order]
    assert positions==sorted(positions),"markers out of order: %s"%(msgs,)
    assert any(m.startswith("advance.end") and "pending=present" in m and "dur_ms=" in m for m in msgs)
    assert any(m.startswith("session.saved") and "outcome=published" in m and "dur_ms=" in m for m in msgs)
    assert any(m.startswith("control_commit.enter") and "fenced=False" in m for m in msgs)
    assert any(m.startswith("start.return") and "open_input=set" in m and "status=30" in m for m in msgs)
    assert session.status==ProcessStatus.WAITING_INPUT


def test_none_advance_trail_shows_pending_none_and_null_return(caplog):
    # The exact live 2026-10-25 shape: the trail must show WHERE it went null.
    store=FakePgStore()
    with caplog.at_level("DEBUG"):
        session,_=start(registry(store,FakeProgram(result=None)))
    msgs=messages(caplog)
    assert any(m.startswith("advance.end") and "pending=none" in m for m in msgs)
    assert any(m.startswith("start.return") and "open_input=null" in m and "status=20" in m for m in msgs)
    assert not any(m.startswith("open_input.set") for m in msgs)


def test_fenced_publish_warns_instead_of_swallowing(caplog):
    # Lease lapses mid-advance (heartbeat gated shut): the post-advance publish
    # fences. The rejection must now be a WARNING, not a silent pg_stopped flip.
    store=FakePgStore(block_heartbeat=True)
    with caplog.at_level("DEBUG"):
        session,_=start(registry(store,FakeProgram(delay=0.8),ttl=0.3))
        store.gate.set()
    warnings=[r.getMessage() for r in caplog.records if r.name==LOGGER and r.levelno>=logging.WARNING]
    assert any(m.startswith("session.save.fenced") for m in warnings),"fenced publish stayed silent: %s"%messages(caplog)
    assert session.pg_stopped is True
    assert session.snapshot().process_status==ProcessStatus.STOP_REQUESTED
