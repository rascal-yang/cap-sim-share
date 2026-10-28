import json
from pathlib import Path
import numpy as np
import pytest
from humanoid_mimicgen.augmentation.verified_seed import exact_revalidation,validate_identity


def test_seed_rejects_new_conditions():
    result={'task':'task','episode':1,'object_delta_xy_m':[.005,0],'mode':'token_prefix_mode1','success':True,'independent_success':True,'coverage':1.,'failure':None}
    plan={'episode':1,'reference_case':'ref','contact_start':3,'resume':{'kind':'restart'},'safety_guard':None}
    spec={'task':'task','baseline':'ref'};params={'episode':1,'contact_start':3,'safety_guard':None}
    validate_identity(result,plan,spec,params,[.005,0])
    with pytest.raises(ValueError):validate_identity(result,plan,spec,params,[.01,0])
    with pytest.raises(ValueError):validate_identity(result|{'success':False},plan,spec,params,[.005,0])
    with pytest.raises(ValueError):validate_identity(result,plan|{'resume':{'kind':'checkpoint'}},spec,params,[.005,0])


def test_identical_trace_does_not_override_native_failure(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b'
    for p in (a,b):
        (p/'execution').mkdir(parents=True);(p/'scene.mjb').write_bytes(b'same scene')
        np.savez(p/'execution/trajectory.npz',state=np.zeros((4,2)),next_state=np.ones((4,2)),commands=np.zeros((4,3)))
    good={'experiment_path':str(b),'success':True,'independent_success':True,'coverage':1.,'failure':None}
    assert exact_revalidation(a,good)['passed']
    assert not exact_revalidation(a,good|{'independent_success':False})['passed']
    np.savez(b/'execution/trajectory.npz',state=np.ones((4,2)),next_state=np.ones((4,2)),commands=np.zeros((4,3)))
    assert not exact_revalidation(a,good)['passed']


@pytest.mark.parametrize('match',[True,False])
def test_local_seed_gate_skips_api_only_on_exact_pass(tmp_path,monkeypatch,match):
    from types import SimpleNamespace
    from humanoid_mimicgen.augmentation import contact_review as c,contact_delivery
    baseline=tmp_path/'baseline';baseline.mkdir();(baseline/'result.json').write_text('{"source_frames":20}')
    params={'action':'run','rationale':'proven','hypothesis':'revalidate','episode':0,'tail_s':0,'wrist_delta_m':[0,0,0],'contact_start':2,'contact_end':10,'safety_guard':None}
    control={'contact_bodies':['hand'],'force_range_N':[0,10],'ramp_s':.1,'joints':[],'max_offset_rad':.1,'max_rate_rad_s':.5,'force_stop_N':20}
    plan=params|{'contact_control':control,'reference_case':str(baseline),'criteria_decision':str(tmp_path/'audit'),'resume':{'kind':'restart','case':'','frame':0,'rationale':'fresh'}}
    dp=tmp_path/'plan';dp.write_text('{}');(tmp_path/'audit').write_text('{}')
    seed={'case':str(baseline),'plan':plan,'decision_path':str(dp),'source_review':'original_api_review'}
    spec={'task':'test','baseline':str(baseline),'annotation':'annotation','dataset':'dataset','verified_seed':'accepted_item'}
    monkeypatch.setattr(c,'install',lambda:None);monkeypatch.setattr(c,'load_seed',lambda *a:seed)
    monkeypatch.setattr(c,'verify_origin',lambda *a:{'decision':params});monkeypatch.setattr(c,'require_prerequisites',lambda *a:None);monkeypatch.setattr(c,'validate_binding',lambda *a:None)
    monkeypatch.setattr(c,'compare',lambda *a:{'replay':{'robot_bodies':['hand'],'joints':[]}})
    monkeypatch.setattr(c,'reusable_audit',lambda *a:({'assessment':'native_valid','rationale':'native checker verified'},tmp_path/'audit'))
    monkeypatch.setattr(c,'configured_provider',lambda *a:SimpleNamespace(config={}))
    monkeypatch.setattr(c,'check_frozen',lambda *a:None);monkeypatch.setattr(c,'offer',lambda *a:None)
    monkeypatch.setattr(c,'inspect_contact_case',lambda *a:{})
    def inspect(case,out):out.mkdir();(out/'inspection.json').write_text('{"rendering_and_inspection_wall_s":0}')
    monkeypatch.setattr(c,'inspect_case',inspect)
    def physics(*a,**kw):
        case=tmp_path/'new_case';case.mkdir()
        return {'experiment_path':str(case),'success':True,'independent_success':True,'coverage':1.,'failure':None,'simulation_wall_s':0}
    monkeypatch.setattr(c,'physical_case',physics);monkeypatch.setattr(c,'exact_revalidation',lambda *a:{'passed':match})
    monkeypatch.setattr(contact_delivery,'deliver',lambda *a:None);calls=[]
    def decide(provider,root,context,*a,**kw):
        calls.append(context['stage']);return {'action':'reject','rationale':'fresh evidence requires review'},tmp_path/'review'
    monkeypatch.setattr(c,'api_decide',decide)
    result=c.run({'output_root':str(tmp_path/'study'),'task':spec,'object_shift_xy_m':[0,0]}, {})
    assert result['physical_attempts']==1
    assert calls==([] if match else ['contact_review'])
    assert result['accepted']==int(match)
    if match:assert result['attempts'][0]['review']['authority']=='local_exact_seed_revalidation'
