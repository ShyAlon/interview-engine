"""Direct review writes publish their own pointer, never the prior graph turn."""
from interview_engine.application import InterviewApplication


def test_direct_state_write_advances_produced_and_resume_pointer():
    app=InterviewApplication.__new__(InterviewApplication)
    app.cfg={'configurable':{'thread_id':'s','checkpoint_ns':'','checkpoint_id':'prior'}}
    app.produced_checkpoint_id='prior'
    class Graph:
        def update_state(self,cfg,values):
            assert cfg['configurable']['checkpoint_id']=='prior'
            assert values=={'intent_model':{'founder_register':{'items':[]}}}
            return {'configurable':{'thread_id':'s','checkpoint_ns':'','checkpoint_id':'review'}}
    app.graph=Graph()
    result=app._update_checkpoint({'intent_model':{'founder_register':{'items':[]}}})
    assert result['configurable']['checkpoint_id']=='review'
    assert app.produced_checkpoint_id==app.cfg['configurable']['checkpoint_id']=='review'
