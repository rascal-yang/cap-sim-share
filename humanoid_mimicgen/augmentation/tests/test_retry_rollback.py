from collections import deque
from types import SimpleNamespace
import numpy as np
import pytest


def test_phase_budget_refresh_respects_total_cap():
    from humanoid_mimicgen.augmentation.retry_state import phase_deadline,budgets
    assert budgets({},3)==(900,1200,3300)
    assert phase_deadline(100,4000,1200)==1300
    assert phase_deadline(3500,4000,1200)==4000
    with pytest.raises(ValueError):budgets({'retry_wall_budget_s':-1},3)


def test_checkpoint_selection_requires_offered_completed_case():
    from humanoid_mimicgen.augmentation.retry_state import validate_resume
    offer={'case':'/case','last_frame':12}
    assert validate_resume({'kind':'checkpoint','case':'/case','frame':8,'rationale':'before contact'},[offer])['frame']==8
    for case,frame in [('/else',8),('/case',13),('/case',-1),('/case',True)]:
        with pytest.raises(ValueError):validate_resume({'kind':'checkpoint','case':case,'frame':frame,'rationale':'why'},[offer])
    with pytest.raises(ValueError):validate_resume({'kind':'restart','case':'/case','frame':0,'rationale':'why'},[offer])


def test_checkpoint_restores_integration_history_and_acceptance_without_step():
    import mujoco
    from humanoid_mimicgen.augmentation.retry_state import capture,restore
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="slide"/><geom type="sphere" size=".1" mass="1"/></body></worldbody><actuator><motor joint="0"/></actuator></mujoco>'.replace('<joint type="slide"/>','<joint name="j" type="slide"/>').replace('joint="0"','joint="j"'))
    d=mujoco.MjData(m);d.time=2.;d.qpos[:]=.3;d.qvel[:]=.7;d.qacc_warmstart[:]=.11;d.ctrl[:]=.9
    s=SimpleNamespace(history=deque([{'q':np.array([.2],dtype=np.float32)}],maxlen=10),last=np.array([.4],dtype=np.float32),tick=7)
    task=SimpleNamespace(_init_target_height=.5,_press_hold_count=4,_contact_started=True,reward=.8)
    a=SimpleNamespace(servo=np.array([.01]),previous=np.array([.2]),hold=4,shadow_ever=False)
    b=SimpleNamespace(m=m,d=d,e=SimpleNamespace(task=task))
    state,meta=capture(b,s,a,7,False)
    d.qpos[:]=9;d.qacc_warmstart[:]=0;s.last[:]=8;s.history.clear();task._press_hold_count=0;a.servo[:]=0
    restore(b,s,a,state,meta)
    assert d.time==2 and d.qpos[0]==.3 and d.qvel[0]==.7 and d.ctrl[0]==.9 and d.qacc_warmstart[0]==.11
    assert task._press_hold_count==4 and task._contact_started
    assert s.last.dtype==np.float32 and s.history[0]['q'].dtype==np.float32 and s.tick==7
    assert a.servo[0]==.01 and a.previous[0]==.2


def test_retry_receives_review_history_checkpoints_and_fresh_budget(tmp_path,monkeypatch):
    import json
    from humanoid_mimicgen.augmentation import contact_review as c,contact_delivery
    baseline=tmp_path/'baseline';baseline.mkdir();(baseline/'result.json').write_text('{"source_frames":20}')
    params={'episode':0,'contact_start':2,'contact_end':10,'safety_guard':None}
    spec={'task':'test','baseline':str(baseline),'annotation':'annotation','dataset':'dataset'}
    config={'output_root':str(tmp_path/'study'),'task':spec,'object_shift_xy_m':[0,0]}
    monkeypatch.setattr(c,'install',lambda:None)
    monkeypatch.setattr(c,'verify_origin',lambda p:{'decision':params})
    monkeypatch.setattr(c,'require_prerequisites',lambda *a:None)
    monkeypatch.setattr(c,'validate_binding',lambda *a:None)
    monkeypatch.setattr(c,'compare',lambda *a:{'replay':{'robot_bodies':['hand'],'joints':[{'name':'elbow'}]}})
    monkeypatch.setattr(c,'configured_provider',lambda *a:SimpleNamespace(config={}))
    monkeypatch.setattr(c,'check_frozen',lambda *a:None)
    monkeypatch.setattr(c,'inspect_contact_case',lambda *a:{'current':True,'robot_bodies':['hand'],'joints':[{'name':'elbow'}]})
    monkeypatch.setattr(c,'offer',lambda p:{'case':str(p),'last_frame':10})
    monkeypatch.setattr(contact_delivery,'deliver',lambda *a:None)
    def inspection(case,out):
        out.mkdir();(out/'inspection.json').write_text('{"rendering_and_inspection_wall_s":0}')
    monkeypatch.setattr(c,'inspect_case',inspection)
    contexts=[];cases=[];deadlines=[]
    (tmp_path/'plan').write_text('{}')
    def decide(provider,root,context,episodes,**kw):
        stage=context['stage']
        if stage=='explore':return {'action':'proceed','rationale':'sufficient evidence','query':None},tmp_path/'explore'
        if stage=='criteria_audit':return {'assessment':'native_valid','rationale':'native checker valid'},tmp_path/'audit'
        if stage in ('contact_plan','contact_retry'):
            contexts.append(context);deadlines.append(provider.deadline)
            resume={'kind':'restart','case':'','frame':0,'rationale':'first'} if len(contexts)==1 else {'kind':'checkpoint','case':str(cases[0]),'frame':8,'rationale':'before loss of contact'}
            return {'contact_start':2,'episode':0,'action':'run','resume':resume},tmp_path/'plan'
        return {'action':'retry' if len(cases)==1 else 'accept','rationale':'Recover from before contact loss'},tmp_path/'review'
    monkeypatch.setattr(c,'api_decide',decide)
    def physics(*a,**kw):
        case=tmp_path/f'case_{len(cases)}';case.mkdir();cases.append(case)
        return {'experiment_path':str(case),'success':len(cases)==2,'independent_success':len(cases)==2,'coverage':1,'failure':None,'simulation_wall_s':0}
    monkeypatch.setattr(c,'physical_case',physics)
    result=c.run(config,{})
    assert result['accepted']==1 and result['physical_attempts']==2
    assert contexts[1]['previous_review']['action']=='retry'
    assert len(contexts[1]['attempt_history'])==1 and contexts[1]['rollback_checkpoints'][0]['case']==str(cases[0])
    assert contexts[0]['phase_wall_budget_s']==900 and contexts[1]['phase_wall_budget_s']==1200
    assert deadlines[1]>deadlines[0]
    assert 'remaining_phase_wall_s' not in contexts[0]  # resume cache cannot contain changing wall-clock context

    assert 'diagnostic' not in contexts[0]['evidence']
    assert contexts[0]['previous_decision'] is None
    assert contexts[0]['initial_parameters']['contact_start']==2
