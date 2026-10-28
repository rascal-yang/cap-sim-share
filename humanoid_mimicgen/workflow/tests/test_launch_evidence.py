import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
import pytest
from humanoid_mimicgen.workflow.contracts import WorkflowError, file_hash
from humanoid_mimicgen.workflow.launch_evidence import navigation_admission, recover_template_evidence
from humanoid_mimicgen.workflow.native import validate_physics


def artifact_fixture(tmp_path):
    from humanoid_mimicgen.workflow.parking import fit_stop_response
    p=tmp_path/'measured.json';p.write_text('fixture measurements')
    return dict(schema_version=2,kind='zero_intercept_velocity_response',scope={'old':True},
                samples=[dict(velocity_xy_m_s=[.1,0],displacement_xy_m=[.01,0]),
                         dict(velocity_xy_m_s=[.2,0],displacement_xy_m=[.02,0])],
                model=fit_stop_response([[.1,0],[.2,0]],[[.01,0],[.02,0]]),evidence_sha256={str(p):file_hash(p)})


def test_calibration_default_is_strict_but_explicit_diagnostic_records_failure(tmp_path):
    args=(artifact_fixture(tmp_path), {}, {'minimum_samples':2,'max_residual_m':.025}, {})
    with pytest.raises(WorkflowError):navigation_admission({},*args)
    result=navigation_admission({'navigation_execution':'uncalibrated_diagnostic','diagnostic_reason':'Measure current fixed controller'},*args)
    assert result['diagnostic_only'] and not result['calibrated'] and result['reason_code']=='needs_calibration'
    with pytest.raises(WorkflowError):navigation_admission({'navigation_execution':'uncalibrated_diagnostic'},*args)
    with pytest.raises(WorkflowError):navigation_admission({'navigation_execution':'typo'},*args)


def test_diagnostic_cannot_suppress_calibration_integrity_failure(monkeypatch,tmp_path):
    import humanoid_mimicgen.workflow.navigation.control as control
    def tampered(*args):raise WorkflowError('integrity_error','Tampered file')
    monkeypatch.setattr(control,'validate_route_calibration',tampered)
    with pytest.raises(WorkflowError,match='Tampered'):
        navigation_admission({'navigation_execution':'uncalibrated_diagnostic','diagnostic_reason':'Measure'}, artifact_fixture(tmp_path), {}, {}, {})


def test_changed_code_recovers_exact_archive_but_assets_and_wrong_archive_fail(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();code=repo/'runner.py';code.write_text('original')
    snapshot=tmp_path/'snapshot';snapshot.mkdir();(snapshot/'runner.py').write_text('original')
    template={'preflight':{'evidence_sha256':{str(code):file_hash(code)}},'pipeline':{'git_commit_and_dirty_patch':{'source_snapshot':str(snapshot)}}}
    code.write_text('new version')
    restored,changes=recover_template_evidence(template,repo)
    assert str(code) not in restored and restored[str(snapshot/'runner.py')]==file_hash(snapshot/'runner.py')
    assert len(changes)==1
    (snapshot/'runner.py').write_text('bad archive')
    with pytest.raises(WorkflowError):recover_template_evidence(template,repo)
    asset=repo/'scene.mjb';asset.write_text('changed asset')
    template['preflight']['evidence_sha256']={str(asset):'not current'}
    with pytest.raises(WorkflowError):recover_template_evidence(template,repo)


def test_flat_physics_requires_identical_model_pd_timestep_and_reward_stride():
    m=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005"/><worldbody><body><joint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
    sonic=SimpleNamespace(kp=np.array([10.]),kd=np.array([1.]))
    flat=dict(body_mass=m.body_mass.tolist(),kp=[10.],kd=[1.],physics_dt_s=.005,reward_stride=4)
    validate_physics(m,flat,sonic=sonic,reward_stride=4)
    for key in flat:
        bad=copy.deepcopy(flat)
        if key=='body_mass':bad[key][-1]+=1
        elif key in ('kp','kd'):bad[key][0]+=1
        else:bad[key]*=2
        with pytest.raises(WorkflowError):validate_physics(m,bad,sonic=sonic,reward_stride=4)
    with pytest.raises(WorkflowError):validate_physics(m,flat)


def test_wrapped_physics_still_checks_options_and_model_identity():
    from humanoid_mimicgen.workflow.native import validate_model_identity
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
    options={'dt':'timestep','gravity':'gravity','integrator':'integrator','solver':'solver','iterations':'iterations',
             'tolerance':'tolerance','cone':'cone','disableflags':'disableflags','enableflags':'enableflags'}
    manifest={key:np.asarray(getattr(m.opt,value)).tolist() for key,value in options.items()};manifest['arrays']={'body_mass':m.body_mass.tolist()}
    validate_model_identity(m,m);validate_physics(m,manifest)
    manifest['iterations']+=1
    with pytest.raises(WorkflowError):validate_physics(m,manifest)


def test_diagnostic_checks_tampering_even_when_calibration_scope_differs(tmp_path):
    p=tmp_path/'measured.json';p.write_text('modified')
    artifact=artifact_fixture(tmp_path);artifact['evidence_sha256']={str(p):'wrong'}
    with pytest.raises(WorkflowError,match='integrity'):
        navigation_admission({'navigation_execution':'uncalibrated_diagnostic','diagnostic_reason':'Measure'},artifact,{}, {}, {})


@pytest.mark.parametrize('broken',['empty','schema','model','samples','evidence'])
def test_diagnostic_rejects_malformed_or_internally_tampered_artifact(tmp_path,broken):
    artifact=artifact_fixture(tmp_path)
    if broken=='empty':artifact={}
    if broken=='schema':artifact['schema_version']=3
    if broken=='model':artifact['model']['tau_s']+=1
    if broken=='samples':artifact['samples'][0]['velocity_xy_m_s'][0]+=1
    if broken=='evidence':artifact['evidence_sha256']={}
    with pytest.raises(WorkflowError):
        navigation_admission({'navigation_execution':'uncalibrated_diagnostic','diagnostic_reason':'Measure'},artifact,{}, {}, {})
