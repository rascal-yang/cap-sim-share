import json
import os
from pathlib import Path
from humanoid_mimicgen.augmentation.__main__ import main
from humanoid_mimicgen.augmentation.jobs import JobStore
from humanoid_mimicgen.augmentation import tools, agent
from humanoid_mimicgen.augmentation.tests.test_worker import fixture
from humanoid_mimicgen.augmentation.tests.test_export import trace
from humanoid_mimicgen.augmentation.tests.test_agent import reply, call, Scripted
from humanoid_mimicgen.workflow import service


def test_json_stdout_and_stderr_diagnostics(tmp_path, monkeypatch, capfd):
    p=tmp_path/'job.json'; p.write_text('{}')
    monkeypatch.setattr('humanoid_mimicgen.augmentation.__main__.load_job',lambda p:{})
    def noisy(j):
        print('python diagnostic'); os.write(1,b'native diagnostic\n')
        return {'status':'validated','physics_steps':0}
    monkeypatch.setattr('humanoid_mimicgen.augmentation.__main__.validate_job',noisy)
    assert main(['validate','--job',str(p)])==0
    output=capfd.readouterr()
    assert json.loads(output.out)['physics_steps']==0
    assert 'native diagnostic' in output.err and 'python diagnostic' in output.err


def test_cli_mock_agent_to_dataset_and_resubmit_has_zero_new_physics(tmp_path, monkeypatch, capfd):
    store, ident, source=fixture(tmp_path,monkeypatch)
    j=store.get(ident)['job']; j['request_id']='api_job'; j['export']=True
    p=tmp_path/'job.json'; p.write_text(json.dumps(j))
    from humanoid_mimicgen.workflow.contracts import file_hash
    monkeypatch.setattr(tools,'validate_job',lambda j:{'status':'validated','input_hashes':{str(source):file_hash(source)},'physics_steps':0})
    from humanoid_mimicgen.augmentation.worker import run_job
    monkeypatch.setattr(tools,'start_job',lambda s,i: {'job_id':i,'status':run_job(s,i)['status']})
    def execute(path):
        import shutil
        saved=trace(Path(path)/'fixture')
        for item in saved.iterdir():
            target=Path(path)/item.name
            if item.is_dir(): shutil.copytree(item,target)
            else: shutil.copyfile(item,target)
        with (tmp_path/'executed').open('a') as f:f.write(Path(path).name+'\n')
        return json.loads((Path(path)/'result.json').read_text())
    monkeypatch.setattr(service,'execute',execute)
    config={'protocol':'responses','base_url':'https://api.example.test/v1','model':'test-model','key_env':'UNSET_TEST',
            'timeout_s':1,'max_rounds':4,'max_output_tokens':1000,'wall_s':20}
    provider=tmp_path/'provider.json'; provider.write_text(json.dumps(config))
    real_run=agent.run_agent
    transport=Scripted([reply(call('read','read_skill',{'name':'augmentation'})),reply(call('submit','submit_job',{})),reply()])
    monkeypatch.setattr(agent,'run_agent',lambda p,c,s:real_run(p,c,s,transport=transport))
    args=['agent','--job',str(p),'--provider',str(provider),'--store',str(store.root)]
    assert main(args)==0
    result=json.loads(capfd.readouterr().out)
    index=json.loads(Path(result['job']['details']['export']['index_path']).read_text())
    assert index['case_count']==2 and index['successes']==2
    assert main(args)==0
    capfd.readouterr()
    assert (tmp_path/'executed').read_text()=='a\nb\n'
