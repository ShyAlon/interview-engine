"""Each worker job installs its own SDK-client binding. A chat model must be built inside the job that uses it, never carried over from
the job that prepared the interview program, and never shared between jobs."""
import re
from pathlib import Path

import pytest

from interview_engine import wire_binding
from interview_engine.attempt_admission import AdmissionUnavailable
from interview_engine.wire_binding import per_job_model, register_model, wire_clients, wire_model

SRC=Path(__file__).resolve().parents[1]/'src'/'interview_engine'


def factory(counter):
    def build():
        counter.append(1)
        return register_model(object(),stage='probing',provider='deepseek',model='m')
    return build


def job():
    return wire_clients(object(),lambda **kw:None)


def accepted(model):
    return wire_model(model,stage='probing',provider='deepseek',model='m')


def test_prepared_in_one_job_and_used_in_the_next_is_the_failure_this_fixes():
    built=[]
    with job():
        get=per_job_model(factory(built))      # the program is prepared here
        assert built==[]                       # nothing is built, nothing can be carried over
    with job():                                # the first answer step runs as a new job
        model=get()
        assert accepted(model) is True


def test_a_model_built_once_and_carried_into_a_new_job_is_refused():
    with job():
        stale=register_model(object(),stage='probing',provider='deepseek',model='m')
    with job(), pytest.raises(AdmissionUnavailable,match='not bound'):
        accepted(stale)


def test_each_job_gets_its_own_model_and_never_the_other_jobs():
    built=[]
    with job():
        get=per_job_model(factory(built))
        first=get();assert get() is first and len(built)==1       # cached inside one job
    with job():
        second=get()
        assert second is not first and len(built)==2
        assert accepted(second) is True
        with pytest.raises(AdmissionUnavailable):accepted(first)  # the old job's model is not usable here


def test_without_a_binding_it_builds_once_up_front():
    built=[];get=per_job_model(factory(built))
    assert len(built)==1
    first=get();assert get() is first and len(built)==1


def test_build_errors_without_a_binding_surface_at_build_time_as_before():
    def broken():raise RuntimeError('no key')
    with pytest.raises(RuntimeError,match='no key'):per_job_model(broken)


def test_graph_no_longer_builds_models_once_at_program_build():
    s=(SRC/'graph.py').read_text()
    body=s.split('def build_graph(')[1].split('\n    def ')[0]
    assert not re.search(r'^\s+\w+_model = model_config\.build_chat_model\(',body,re.M)
    for name in ('probing_model','categorization_model','distill_model','conops_model','tiebreak_model'):
        assert f'{name} = per_job_model(' in body
        # every use calls the getter inside the job
        for hit in re.finditer(r'\b'+name+r'\b(?!\s*=)',s):
            tail=s[hit.end():hit.end()+2]
            assert tail=='()', (name,s[hit.start()-40:hit.end()+20])
