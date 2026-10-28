import numpy as np
import pytest
from cap_sim.contact_prior import extract_prior, map_prior


def test_contact_mapping_uses_object_frame_rotation_scale_and_current_feedback():
    prior = {'object_key':'target','reference_scale':1.,'contacts':[
        {'hand_body':'thumb','position_object_m':[.1,0,0],'position_wrist_m':[.02,0,0]},
        {'hand_body':'index','position_object_m':[-.1,0,0],'position_wrist_m':[-.02,0,0]}],
        'wrist_quaternion_object_wxyz':[1,0,0,0]}
    q=np.sqrt(.5)
    observation={'objects':{'target':[1,2,3,q,0,0,q]},'contacts':{'events':[
        {'bodies':['thumb','apple'],'position':[1,2.12,3],'force_N':2.}]}}
    mapped=map_prior(prior,observation,scale=1.2)
    assert np.allclose(mapped['target_contacts'][0]['position_world_m'],[1,2.12,3])
    assert np.allclose(mapped['target_contacts'][1]['position_world_m'],[1,1.88,3])
    assert mapped['target_contacts'][0]['nearest_measured_error_m']==pytest.approx(0)
    assert mapped['target_contacts'][1]['nearest_measured_error_m'] is None
    assert mapped['force_semantics']=='observations_only_not_control_targets'


def test_extraction_rejects_unaccepted_reference(tmp_path):
    import json
    (tmp_path/'trajectory.json').write_text(json.dumps({'acceptance':{'success':False}}))
    with pytest.raises(ValueError,match='successful'):
        extract_prior(tmp_path/'trajectory.json',object_body='apple',wrist_body='wrist')


def test_invalid_scale_and_pose_are_not_silently_used():
    prior={'object_key':'target','reference_scale':1.,'contacts':[],
           'wrist_quaternion_object_wxyz':[1,0,0,0]}
    with pytest.raises(ValueError):map_prior(prior,{'objects':{'target':[0,0,0,1,0,0,0]}},scale=-1)
    with pytest.raises(ValueError):map_prior(prior,{'objects':{'target':[0,0,0,0,0,0,0]}},scale=1)


def test_feedback_refreshes_live_targets_but_preserves_reset_anchor():
    from cap_sim.contact_prior import enrich_request
    prior={'object_key':'target','object_body':'apple','reference_scale':1.,'contacts':[
        {'hand_body':'hand_thumb','position_object_m':[.03,0,0],'position_wrist_m':[.01,0,0]}],
        'wrist_quaternion_object_wxyz':[1,0,0,0]}
    req={'task':{'scene':{}},'observation':{'objects':{'target':[0,0,0,1,0,0,0]}}}
    enrich_request(req,[prior]);req['observation']['objects']['target'][0]=.1
    enrich_request(req,[prior])
    assert req['contact_privileged']['initial_targets'][0]['target_contacts'][0]['position_world_m'][0]==pytest.approx(.03)
    assert req['contact_privileged']['current_targets'][0]['target_contacts'][0]['position_world_m'][0]==pytest.approx(.13)


def test_core_supplies_contact_payload_to_initial_and_feedback_plans(monkeypatch,tmp_path):
    import copy
    from cap_sim.core import run_task
    from cap_sim.io import write,read
    from test_core import Backend,task
    prior={'object_key':'target','object_body':'apple','reference_scale':1.,'contacts':[
        {'hand_body':'hand_thumb','position_object_m':[.03,0,0],'position_wrist_m':[.01,0,0]}],
        'wrist_quaternion_object_wxyz':[1,0,0,0]}
    monkeypatch.setattr('cap_sim.core.extract_prior',lambda *args,**kwargs:copy.deepcopy(prior))
    t=task();t['budget'].update(attempts=1,optimization_attempts=0);t['contact_prior']={'object_body':'apple','wrist_body':'wrist'}
    path=tmp_path/'reference.json';write(path,{'actions':[]});t['references']=[str(path)]
    class B(Backend):
        def describe(self,t):return {'objects':{'target':[0,0,0,1,0,0,0]}}
        def execute(self,a,remaining):
            result=super().execute(a,remaining);result['observation']={'objects':{'target':[.1,0,0,1,0,0,0]}};return result
    class P:
        def __init__(self):self.requests=[]
        def plan(self,q):
            self.requests.append(copy.deepcopy(q));feedback=q.get('feedback_index',0)>0
            return {'analysis':{'evidence':['measured contact prior'],'hypothesis':'test payload'},
                    'actions':[{'name':'win' if feedback else 'reach','replan_after':not feedback}]}
    p=P();assert run_task(t,p,B(),tmp_path/'run')['success']
    assert len(p.requests)==2
    first,second=[q['contact_privileged'] for q in p.requests]
    assert second['initial_targets']==first['initial_targets']
    assert second['current_targets'][0]['target_contacts'][0]['position_world_m'][0]==pytest.approx(.13)
    assert read(tmp_path/'run/experience.json')['contact_priors']==[prior]
    assert read(tmp_path/'run/attempts/001/feedback_001_request.json')['contact_privileged']==second


def test_extract_local_points_from_moving_rotating_object(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    from cap_sim.io import write,digest
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.005"/><worldbody>
      <body name="apple"><freejoint/><geom type="sphere" size=".1" mass=".1"/></body>
      <body name="wrist" pos="0 0 .3"><geom type="sphere" size=".01"/></body>
      </worldbody></mujoco>''')
    mujoco.mj_saveModel(model,str(tmp_path/'scene.mjb'),None)
    from cap_sim.contact_prior import rotation
    states=[];contacts=[]
    for i in range(45):
        angle=i*.02;q=np.array([np.cos(angle/2),0,0,np.sin(angle/2)]);pos=np.array([i*.001,0,.3]);events=[]
        for name,x in [('hand_thumb',.03),('hand_index',-.03)]:
            events.append({'bodies':[name,'apple'],'force_N':1.,'position':(pos+rotation(q)@np.array([x,0,0])).tolist()})
        states.append(np.r_[(i+1)*.005,pos,q,np.zeros(model.nv+model.nu)]);contacts.append({'events':events})
    np.savez(tmp_path/'states.npz',samples=states);write(tmp_path/'contacts.json',contacts);write(tmp_path/'samples.json',[])
    write(tmp_path/'independent_validation.json',{'success':True})
    write(tmp_path/'trajectory.json',{'acceptance':{'success':True},'task':{'scene':{}},'samples_sha256':digest([]),'source_attempt':str(tmp_path)})
    p=extract_prior(tmp_path/'trajectory.json',object_body='apple',wrist_body='wrist')
    byname={c['hand_body']:c for c in p['contacts']}
    assert np.allclose(byname['hand_thumb']['position_object_m'],[.03,0,0],atol=1e-10)
    assert byname['hand_thumb']['local_scatter_max_m']<1e-10
    write(tmp_path/'contacts.json',contacts[:-1])
    with pytest.raises(ValueError,match='alignment'):extract_prior(tmp_path/'trajectory.json',object_body='apple',wrist_body='wrist')
