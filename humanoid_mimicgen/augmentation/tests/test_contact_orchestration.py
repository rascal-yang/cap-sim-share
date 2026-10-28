import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from humanoid_mimicgen.augmentation import contact_ablation as c


def test_legacy_guard_equivalence():
    old={'episode':0,'tail_s':0,'wrist_delta_m':[0,0,0],'contact_start':2,'contact_end':4}
    c.validate_matched(old|{'safety_guard':{'root_height_min_m':.3,'root_tilt_max_deg':50}},old)
    with pytest.raises(ValueError):c.validate_matched(old|{'safety_guard':{'root_height_min_m':.5,'root_tilt_max_deg':50}},old)


def setup_run(tmp_path,monkeypatch):
    params={'action':'run','episode':0,'tail_s':0,'wrist_delta_m':[0,0,0],'contact_start':2,'contact_end':4,'safety_guard':None}
    baseline=tmp_path/'baseline';baseline.mkdir();(baseline/'result.json').write_text('{"success":true}')
    config={'output_root':str(tmp_path/'study'),'tasks':[{'task':'test','annotation':'old.json','baseline':str(baseline),'dataset':'unused'}]}
    monkeypatch.setattr(c,'install',lambda:None);monkeypatch.setattr(c,'configured_provider',lambda *a:None)
    monkeypatch.setattr(c,'verify_origin',lambda *a:{'decision':params})
    def decide(provider,root,context,episodes):
        return params|{'action':'run' if context['stage'].endswith('plan') else 'reject'},root/'decision.json'
    monkeypatch.setattr(c,'api_decide',decide)
    def physics(dataset,ep,root,mode,**kw):
        case=root/mode;case.mkdir(exist_ok=True);return {'experiment_path':str(case),'success':False,'original_token_steps':0}
    physical=Mock(side_effect=physics);monkeypatch.setattr(c,'physical_case',physical)
    monkeypatch.setattr(c,'diagnose',lambda *a:{'native_success_at_end':False})
    def inspect(case,out):
        out.mkdir();r={'review_images':[]};(out/'inspection.json').write_text(json.dumps(r));return r
    monkeypatch.setattr(c,'inspect_case',inspect)
    return config,physical


def test_failed_review_does_not_suppress_other_arm(tmp_path,monkeypatch):
    config,physical=setup_run(tmp_path,monkeypatch)
    def diag(case,dataset):
        if case.name=='cartesian_source_legs':raise RuntimeError('read-only evidence failure')
        return {'native_success_at_end':False}
    monkeypatch.setattr(c,'diagnose',diag)
    result=c.run(config,{})
    assert physical.call_count==2
    task=result['tasks'][0]
    assert task['status']=='error' and len(task['arms'])==2
    assert task['arms'][0]['physical_result']['experiment_path']
    assert task['arms'][1]['review']['action']=='reject'


def test_resume_reuses_completed_physics(tmp_path,monkeypatch):
    config,physical=setup_run(tmp_path,monkeypatch)
    monkeypatch.setattr(c,'diagnose',Mock(side_effect=RuntimeError('diagnostic failure')))
    c.run(config,{})
    assert physical.call_count==2
    monkeypatch.setattr(c,'diagnose',lambda *a:{'native_success_at_end':False})
    result=c.run(config,{},resume_errors=True)
    assert result['tasks'][0]['status']=='measured'
    assert physical.call_count==2
    assert list((Path(config['output_root'])/'test').glob('result.error.*.json'))


def test_api_accept_cannot_override_physical_failure(tmp_path,monkeypatch):
    config,_=setup_run(tmp_path,monkeypatch)
    original=c.api_decide
    def accept_review(provider,root,context,episodes):
        d,path=original(provider,root,context,episodes)
        return d|{'action':'run' if context['stage'].endswith('plan') else 'accept'},path
    monkeypatch.setattr(c,'api_decide',accept_review)
    result=c.run(config,{})
    assert all(not arm['quality_accepted'] for arm in result['tasks'][0]['arms'])
