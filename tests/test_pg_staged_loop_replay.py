"""Staged replay alignment proven against a REAL LangGraph execution: loops
(same node twice in one advance), parallel nodes, and a crash after a super-step
committed but before the next node's call completed (partial checkpoint). The
execution point is anchor:super-step; a resume from the SAME anchor with the
SAME input (the application's real resume shape) re-executes the SAME
super-steps, so staged rows line up: completed calls replay (no rebill), the
crashed call pauses with attempt evidence."""
import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')
from conftest import managed_pg_server, managed_pool
import psycopg
import psycopg.rows
from typing import Annotated
from typing_extensions import TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

import interview_engine.attempt_admission as aa
import interview_engine.model_config as mc
from interview_engine.pg_checkpoint import FencedPostgresSaver, LeaseFence, bootstrap_checkpoint_schema
from interview_engine.pg_session_store import PgSessionStore
from test_pg_session_store import MIRRORED_REGISTRY_DDL, MIRRORED_STAGING_DDL
from test_pg_staged_responses import FakeGateway
from interview_engine.errors import UserFacingError

OWNER = '00000000-0000-0000-0000-000000000001'


class S(TypedDict):
    log: Annotated[list, operator.add]
    limit: int  # replace-reducer: each turn's input sets how long log may grow


def make_store(tmp_path):
    server = managed_pg_server(str(tmp_path / 'pgdata'))
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.execute(MIRRORED_STAGING_DDL)
    conn.commit()
    pool = managed_pool(server.get_uri())
    store = PgSessionStore(pool)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=600)
    assert claim is not None
    saver_conn = psycopg.connect(server.get_uri(), autocommit=True)
    bootstrap_checkpoint_schema(saver_conn)
    saver_conn.close()
    return store, claim, pool


def make_saver(pool, claim):
    return FencedPostgresSaver(pool, lease=LeaseFence('s1', claim.generation, claim.epoch))


def latest_ckpt(saver):
    return saver.get_tuple({'configurable': {'thread_id': 's1'}}).config['configurable']['checkpoint_id']


def mk_node(calls, name, payload, count, boom_at=None):
    def node(state):
        idx = count['n']
        count['n'] += 1
        def call():
            if boom_at is not None and idx == boom_at:
                raise RuntimeError(f'crash at call {idx}')
            calls.append((name, idx) if count is not None else name)
            return AIMessage(content=f'{payload}-{idx}' if '{idx}' in payload or payload in ('a','b') else payload, usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2})
        resp = aa.admitted_invoke(call, stage='loop_stage', provider='fake', model='fake-1',
                                  messages=[], max_output_tokens=10)
        return {'log': [resp.content]}
    return node


def loop_graph(saver, calls, boom_at=None):
    """a -> b -> a -> b per invoke (until log reaches `limit` entries)."""
    count = {'n': 0}
    g = StateGraph(S)
    g.add_node('a', mk_node(calls, 'a', 'a', count, boom_at))
    g.add_node('b', mk_node(calls, 'b', 'b', count, boom_at))
    g.add_edge(START, 'a')
    g.add_edge('a', 'b')
    g.add_conditional_edges('b', lambda s: 'a' if len(s['log']) < s['limit'] else END, {'a': 'a', END: END})
    return g.compile(checkpointer=saver)


def test_loop_crash_after_partial_checkpoint_resume_replays_and_pauses(tmp_path):
    """Turn 2 loops a,b,a,b and crashes during the 3rd call (second visit of node
    a), AFTER the a,b super-steps committed. Resume from the same pinned anchor
    with the same input (the application's real resume shape): the completed
    calls replay WITHOUT provider calls or new ledger attempts; the crashed call
    pauses with its attempt id."""
    store, claim, pool = make_store(tmp_path)
    gw = FakeGateway()
    staging = mc.StagingContext('s1', store, claim)
    saver = make_saver(pool, claim)
    app = loop_graph(saver, calls := [])
    cfg = {'configurable': {'thread_id': 's1'}}
    token = mc._response_staging.set(staging)
    with aa.admission_gateway(gw, required=True):
        staging.set_execution_point('anchor-T1', 0)
        app.invoke({'log': [], 'limit': 4}, cfg)  # turn 1 completes: a,b,a,b
        anchor = latest_ckpt(saver)               # the published pointer at turn 1's end
        calls.clear()
        # fix payload per idx now that count restarts for turn 2
        app2 = loop_graph(saver, calls, boom_at=2)
        staging.set_execution_point(anchor, 0)
        with pytest.raises(RuntimeError, match='crash at call 2'):
            app2.invoke({'log': ['turn-2 input'], 'limit': 9},
                        {'configurable': {'thread_id': 's1', 'checkpoint_id': anchor}})
    mc._response_staging.reset(token)
    assert calls == [('a', 0), ('b', 1)]  # two calls completed, third crashed
    assert len(gw.reserved) == 4 + 3 and len(gw.uncertain_calls) == 1

    # Resume: fresh process state (new StagingContext), same anchor, same input.
    staging2 = mc.StagingContext('s1', store, claim)
    saver2 = make_saver(pool, claim)
    app3 = loop_graph(saver2, calls2 := [])
    token = mc._response_staging.set(staging2)
    with aa.admission_gateway(gw, required=True):
        staging2.set_execution_point(anchor, 0)
        with pytest.raises(UserFacingError, match='Reconcile|result is missing'):
            app3.invoke({'log': ['turn-2 input'], 'limit': 9},
                        {'configurable': {'thread_id': 's1', 'checkpoint_id': anchor}})
    mc._response_staging.reset(token)
    assert calls2 == [], f'completed calls must replay, not re-invoke: {calls2}'
    assert len(gw.reserved) == 7, 'no new ledger attempt during replay+pause'
    # The crashed call's staged row (dispatched, no response) is the pause evidence.
    with store._borrow() as c:
        open_rows = c.execute("SELECT node, attempt_id FROM staged_provider_responses WHERE session_id='s1' AND response IS NULL").fetchall()
    assert len(open_rows) == 1 and open_rows[0]['node'] == 'a' and open_rows[0]['attempt_id'] == gw.uncertain_calls[0]


def test_parallel_nodes_stage_and_replay_under_their_own_node_identity(tmp_path):
    """Two parallel nodes (same super-step, same stage) stage under distinct node
    identities; a resume of the SAME turn replays each under its own key with no
    provider calls and no new ledger attempts - no cross-node mix."""
    store, claim, pool = make_store(tmp_path)
    gw = FakeGateway()
    staging = mc.StagingContext('s1', store, claim)
    saver = make_saver(pool, claim)

    def parallel_graph(sv, calls):
        g = StateGraph(S)
        g.add_node('x', mk_node(calls, 'x', 'x-result', {'n': 0}))
        g.add_node('y', mk_node(calls, 'y', 'y-result', {'n': 0}))
        g.add_edge(START, 'x')
        g.add_edge(START, 'y')
        g.add_edge('x', END)
        g.add_edge('y', END)
        return g.compile(checkpointer=sv)

    calls = []
    app = parallel_graph(saver, calls)
    cfg = {'configurable': {'thread_id': 's1'}}
    token = mc._response_staging.set(staging)
    with aa.admission_gateway(gw, required=True):
        staging.set_execution_point('anchor-T1', 0)
        app.invoke({'log': []}, cfg)              # turn 1: x, y
        anchor = latest_ckpt(saver)
        staging.set_execution_point(anchor, 0)
        app.invoke({'log': ['turn-2 input']}, {'configurable': {'thread_id': 's1', 'checkpoint_id': anchor}})
    mc._response_staging.reset(token)
    assert len(calls) == 4
    # Each (execution_point, node) pair staged exactly once per turn: no collision.
    with store._borrow() as c:
        rows = c.execute("SELECT execution_point, node, count(*) AS n FROM staged_provider_responses GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    assert len(rows) == 4 and all(r['n'] == 1 for r in rows)
    assert sorted(r['node'] for r in rows) == ['x', 'x', 'y', 'y']

    # Resume turn 2 from the same anchor with the same input: both replay.
    staging2 = mc.StagingContext('s1', store, claim)
    saver2 = make_saver(pool, claim)
    calls2 = []
    app2 = parallel_graph(saver2, calls2)
    token = mc._response_staging.set(staging2)
    with aa.admission_gateway(gw, required=True):
        staging2.set_execution_point(anchor, 0)
        result = app2.invoke({'log': ['turn-2 input']}, {'configurable': {'thread_id': 's1', 'checkpoint_id': anchor}})
    mc._response_staging.reset(token)
    assert calls2 == [], f'parallel calls must replay, not re-invoke: {calls2}'
    assert sorted(result['log'][-2:]) == ['x-result', 'y-result']
    assert len(gw.reserved) == 4, 'no new ledger attempts on replay'
