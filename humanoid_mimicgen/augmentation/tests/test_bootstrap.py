import numpy as np
import pytest
from humanoid_mimicgen.augmentation.bootstrap import press_segments, validate_source_clock


def test_press_requires_continuous_native_condition_and_shared_boundaries():
    times=np.arange(100)*.02
    pressing=np.zeros(100,bool); pressing[20:80]=True
    result=press_segments(times, pressing, 50)
    assert result['status']=='ready'
    assert result['segments'][0]['end_frame']==result['segments'][1]['start_frame']==20
    assert result['segments'][1]['end_frame']==result['segments'][2]['start_frame']==79
    assert result['segments'][2]['end_frame']==99


def test_disconnected_taps_are_not_successful_press():
    mask=np.ones(100,bool); mask[::20]=False
    assert press_segments(np.arange(100)*.02,mask,50)['status']=='rejected'


def test_precontrol_clock_rejects_missing_frames_or_wrong_initial():
    validate_source_clock(np.arange(10)*.02+2.005,2.005,.02,'pre_control')
    with pytest.raises(ValueError): validate_source_clock([2.005,2.045],2.005,.02,'pre_control')
    with pytest.raises(ValueError): validate_source_clock([2.005,2.025],2,.02,'pre_control')
    with pytest.raises(ValueError): validate_source_clock([2.005,2.025],2.005,.02,'post_control')


def test_dispatch_rejects_out_of_scope_episode_and_unread_skill(tmp_path):
    from humanoid_mimicgen.augmentation.bootstrap_agent import dispatch
    c={'root':tmp_path,'config':{'episodes':[0]}}
    with pytest.raises(ValueError,match='budget'):dispatch('replay_physical',{'episode':1},c)
    with pytest.raises(ValueError,match='Read skill'):dispatch('replay_physical',{'episode':0},c)


def test_interrupted_physics_is_not_silently_repeated(tmp_path):
    from humanoid_mimicgen.augmentation.bootstrap_agent import dispatch
    c={'root':tmp_path,'config':{'episodes':[0]}}
    stage=tmp_path/'tools';stage.mkdir()
    for name in ['read_skill','replay_saved_000000','replay_physical_000000.started']:
        (stage/(name+'.json')).write_text('{}')
    with pytest.raises(RuntimeError,match='silently rerun'):dispatch('replay_physical',{'episode':0},c)


def test_mode1_object_shift_changes_only_wrist_position_channels():
    from types import SimpleNamespace
    from humanoid_mimicgen.augmentation.bootstrap import encode_reference
    class Encoder:
        def get_inputs(self):return [SimpleNamespace(name='in')]
        def run(self,_,feed):return [np.zeros((1,64),np.float32)]
    s=SimpleNamespace(qids=np.arange(7,36),vids=np.arange(6,35),encoder=Encoder())
    q=np.zeros((20,36));q[:,3]=1;v=np.zeros((20,35))
    poses=np.zeros((20,3,7));poses[:,:,3]=1
    d=SimpleNamespace(qpos=q[0])
    _,a=encode_reference(s,d,q,v,poses,0,np.array([0.,0.]))
    _,b=encode_reference(s,d,q,v,poses,0,np.array([.01,-.02]))
    changed=np.flatnonzero(a!=b)
    assert changed.tolist()==[901,902,904,905]
    assert a[0]==b[0]==1


def test_failed_gateway_response_is_retained_and_retried_without_tools(tmp_path):
    from humanoid_mimicgen.augmentation.bootstrap_agent import fetch_response
    from types import SimpleNamespace
    import json
    folder=tmp_path/'round';folder.mkdir()
    (folder/'response.json').write_text(json.dumps({'status':'failed','error':{'type':'upstream_error'}}))
    calls=[]
    def respond(*args):
        calls.append(1)
        return {'status':'completed','output':[], 'usage':{'input_tokens':2,'output_tokens':1,'total_tokens':3}}
    result=fetch_response(SimpleNamespace(respond=respond),[],folder,tmp_path)
    assert result['status']=='completed' and len(calls)==1
    assert list(folder.glob('response.failed.*.json'))


def test_contact_tail_uses_native_hold_duration_and_preserves_source():
    import pandas as pd
    from humanoid_mimicgen.augmentation.bootstrap import extend_contact_tail
    df=pd.DataFrame({'sim.time':[2.,2.02], 'sim.qvel':[np.ones(3),np.ones(3)], 'action.motion_token':[np.zeros(64),np.ones(64)]})
    out=extend_contact_tail(df,50,.02,.02)
    assert len(df)==2 and len(out)==52
    assert out['source_frame'].tolist()==[0]+[1]*51
    assert np.all(out['sim.qvel'].iloc[-1]==0)
    assert out['sim.time'].iloc[-1]==pytest.approx(3.02)
    assert np.array_equal(out['action.motion_token'].iloc[-1],df['action.motion_token'].iloc[-1])


def test_truncated_stream_can_retry_but_no_tool_is_executed(tmp_path,monkeypatch):
    from humanoid_mimicgen.augmentation import bootstrap_agent as a
    from humanoid_mimicgen.workflow.contracts import WorkflowError
    from types import SimpleNamespace
    calls=[]
    def respond(*args):
        calls.append(1)
        if len(calls)==1:raise WorkflowError('invalid_api_response','Stream ended without terminal response')
        return {'status':'completed','output':[],'usage':None}
    monkeypatch.setattr(a.time,'sleep',lambda _:None)
    folder=tmp_path/'round';folder.mkdir()
    assert a.fetch_response(SimpleNamespace(respond=respond),[],folder,tmp_path)['status']=='completed'
    assert len(calls)==2
