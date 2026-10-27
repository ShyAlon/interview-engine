import asyncio
import socket
import pytest
from requi_child_contract.broker import BrokerClient,BrokerTransport,AsyncBrokerTransport
from interview_engine import wire_binding as module


def test_explicit_broker_binding_default_unchanged():
    parent,child=socket.socketpair();client=BrokerClient(child.fileno(),next_call_id=lambda stage:'call-1')
    try:
        assert module.model_clients(stage='intent',provider='deepseek',model='m') is None
        with module.broker_clients(client):
            pair=module.model_clients(stage='intent',provider='deepseek',model='m')
            assert isinstance(pair['http_client']._transport,BrokerTransport)
            assert isinstance(pair['http_async_client']._transport,AsyncBrokerTransport)
            obj=object();module.register_model(obj,stage='intent',provider='deepseek',model='m')
            assert module.wire_model(obj,stage='intent',provider='deepseek',model='m')
            with pytest.raises(module.AdmissionUnavailable):module.wire_model(obj,stage='wrong',provider='deepseek',model='m')
            pair['http_client'].close();asyncio.run(pair['http_async_client'].aclose())
        assert module.model_clients(stage='intent',provider='deepseek',model='m') is None
    finally:client.close();parent.close();child.close()


def test_per_job_model_rebuilds_for_each_broker_not_old_model():
    parent,child=socket.socketpair();client=BrokerClient(child.fileno(),next_call_id=lambda stage:'call-1');made=[]
    try:
        def build():
            model=object();made.append(model);return model
        get=module.per_job_model(build)
        local=get()
        with module.broker_clients(client):one=get();assert one is get()
        with module.broker_clients(client):two=get();assert two is get()
        assert local is not one and one is not two and len(made)==3
    finally:client.close();parent.close();child.close()
