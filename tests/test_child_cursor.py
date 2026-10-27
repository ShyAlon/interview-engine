import pytest
from interview_engine.child_cursor import broker_call_id,broker_invocation
from interview_engine import model_config


def test_no_cursor_is_not_a_random_paid_call():
    with pytest.raises(PermissionError):broker_call_id('intent')


def test_durable_graph_point_node_item_ordinal_replay():
    class Claim:generation=1
    a=model_config.StagingContext('session',None,Claim());b=model_config.StagingContext('session',None,Claim())
    def call(ctx,item='item'):
        token=model_config._response_staging.set(ctx)
        try:
            with broker_invocation('intent',item):return broker_call_id('intent')
        finally:model_config._response_staging.reset(token)
    first=call(a);assert call(b)==first
    assert call(a)!=first
    b.set_execution_point('new-checkpoint');assert call(b)!=first
    with broker_invocation('intent','item'):
        with pytest.raises(PermissionError):broker_call_id('conops')
