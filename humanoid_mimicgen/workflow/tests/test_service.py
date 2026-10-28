import pytest
from humanoid_mimicgen.workflow.service import input_context, check_contract
from humanoid_mimicgen.workflow.contracts import WorkflowError


def test_missing_evidence_does_not_get_default(tmp_path):
    with pytest.raises(WorkflowError):input_context({'targets':str(tmp_path/'missing')})
    p=tmp_path/'a';p.write_text('a');c=input_context({'targets':str(p)})
    p.write_text('b');assert input_context({'targets':str(p)})!=c


def test_unfrozen_contract_cannot_launch(tmp_path):
    p=tmp_path/'experiment.json';p.write_text('{}')
    with pytest.raises(WorkflowError):check_contract(p)


def test_cli_keeps_native_diagnostics_off_json_stdout(monkeypatch,capfd):
    import os,json
    from humanoid_mimicgen.workflow import service
    from humanoid_mimicgen.workflow.__main__ import main
    def noisy(_):
        print('python diagnostic');os.write(1,b'native diagnostic\n')
        return {'status':'prepared'}
    monkeypatch.setattr(service,'inspect',noisy)
    assert main(['inspect','--experiment','.'])==0
    captured=capfd.readouterr()
    assert json.loads(captured.out)=={'status':'prepared'}
    assert 'native diagnostic' in captured.err and 'python diagnostic' in captured.err


def test_legacy_record_without_inputs_reaches_native_preflight(tmp_path,monkeypatch):
    import json
    import numpy as np
    from humanoid_mimicgen.workflow import service
    import humanoid_mimicgen.workflow.native as native
    import humanoid_mimicgen.cli as cli
    files={}
    for name in ('model','initial','targets','reference_states'):
        path=tmp_path/(name+'.npz');np.savez(path,qpos=np.zeros(7));files[name]=str(path)
    record=tmp_path/'record.json';record.write_text(json.dumps({'preflight':{'evidence_sha256':{}}}));files['record']=str(record)
    seed=tmp_path/'controller_seed.npz';np.savez(seed,last_action=np.zeros(29))
    task=tmp_path/'task.json';task.write_text(json.dumps({'resources':{'initial_state':str(seed)}}));files['task']=str(task)
    keys=('context','bindings','phases','policy','command_pose_offset','source_frame_count','adapter')
    strategy={key:{} for key in keys};strategy.update(provenance={'implementation_sha256':{}},proposal={'stance':{'id':'chosen'}})
    monkeypatch.setattr(service.Store,'load',lambda *args:{'strategy':strategy})
    monkeypatch.setattr(service,'_strategy',lambda request:strategy)
    monkeypatch.setattr(service,'_geometry',lambda *args:{'candidates':[{'id':'chosen','feasible':True}]})
    output=tmp_path/'prepared'
    def create(*args):output.mkdir();return output
    monkeypatch.setattr(cli,'create_experiment',create)
    def stop_before_physics(*args):
        assert args[0]['resources']['initial_state']==str(seed)
        assert all((output/(name+'.npz')).exists() for name in ('model','initial','targets','reference_states'))
        raise WorkflowError('test_stop_at_native','No physics in fixture')
    monkeypatch.setattr(native,'NativeAdapter',stop_before_physics)
    with pytest.raises(WorkflowError,match='test_stop_at_native'):
        service.prepare('fixture',dict(strategy_root=str(tmp_path),experiment_root=str(tmp_path),files=files))
