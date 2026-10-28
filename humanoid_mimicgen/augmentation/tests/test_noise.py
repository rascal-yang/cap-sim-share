import numpy as np
import pytest
from humanoid_mimicgen.augmentation.noise import JointNoise


def test_noise_reproducible_held_between_policy_ticks_and_no_accumulation():
    a=JointNoise([1,3],.01,17,[-1,-1],[1,1]);b=JointNoise([1,3],.01,17,[-1,-1],[1,1])
    target=np.zeros(5)
    first=a.apply(target,True)
    assert np.array_equal(first,b.apply(target,True))
    assert np.array_equal(first,a.apply(target,False))
    assert np.array_equal(target,np.zeros(5))
    assert np.array_equal(first[[0,2,4]],target[[0,2,4]])
    assert not np.array_equal(first,a.apply(target,True))


def test_noise_respects_joint_ranges():
    a=JointNoise([0],1.,1,[-.1],[.1])
    for _ in range(10):assert abs(a.apply(np.zeros(2),True)[0])<=.1


def test_noise_declared_before_contract_write_and_context_restored(tmp_path,monkeypatch):
    from humanoid_mimicgen.augmentation.noise import freeze_noise
    from humanoid_mimicgen.workflow import service
    captured=[]
    original=lambda p,v:captured.append(v)
    monkeypatch.setattr(service,'write_new',original)
    config={'seed':3,'sigma_rad':.01,'interface':'post_sonic_joint_target','resample':'policy_tick','joint_group':'both_arms'}
    with freeze_noise(config):service.write_new(tmp_path/'experiment.json',{'control':{},'pipeline':{}})
    assert service.write_new is original
    assert captured[0]['control']['motion_noise']['seed']==3
    assert 'execute-case' in captured[0]['pipeline']['entrypoint']


def test_export_keeps_nominal_and_executed_actions_separate(tmp_path):
    import json
    from humanoid_mimicgen.augmentation.tests.test_export import trace
    from humanoid_mimicgen.augmentation.export import export_case
    from humanoid_mimicgen.workflow.contracts import file_hash,WorkflowError
    root=trace(tmp_path);out=root/'execution'
    c=json.loads((root/'experiment.json').read_text());config={'seed':3,'sigma_rad':.01,'interface':'post_sonic_joint_target','resample':'policy_tick','joint_group':'both_arms'}
    c['control']={'motion_noise':config};(root/'experiment.json').write_text(json.dumps(c));(root/'experiment.sha256').write_text(file_hash(root/'experiment.json'))
    body=np.zeros((2,14));body[:,0]=.01
    cmds=json.loads((out/'commands.json').read_text())
    for i,x in enumerate(cmds):x['body_target_rad']=body[i].tolist()
    for p in ('commands.json','attempted_commands.json'):(out/p).write_text(json.dumps(cmds))
    arrays=dict(time_s=np.array([0,.01]),nominal_body_target_rad=np.zeros((2,14)),executed_body_target_rad=body,
                nominal_ctrl_Nm=np.zeros((2,2)),executed_ctrl_Nm=np.array([[1,2],[3,4]]),policy_tick=np.array([True,False]))
    np.savez(out/'motion_noise.npz',**arrays)
    metadata={'config':config,'arm_target_indices':list(range(14)),'npz_sha256':file_hash(out/'motion_noise.npz')}
    (out/'motion_noise.json').write_text(json.dumps(metadata))
    export_case(root,tmp_path/'export')
    with np.load(tmp_path/'export/trajectory.npz') as x:
        assert np.all(x['nominal_body_target_rad']==0)
        assert np.all(x['body_target_noise_rad'][:,0]==.01)
    arrays['time_s']+=.1;np.savez(out/'motion_noise.npz',**arrays);metadata['npz_sha256']=file_hash(out/'motion_noise.npz');(out/'motion_noise.json').write_text(json.dumps(metadata))
    with pytest.raises(WorkflowError,match='alignment_error'):export_case(root,tmp_path/'bad')
