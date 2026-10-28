import json
import subprocess
import sys
from humanoid_mimicgen.experience_dataset.store import Dataset
from humanoid_mimicgen.experience_dataset.collector import collect_once, run_command


def test_collection_writes_persistent_health_and_retries(tmp_path):
    source=tmp_path/'exp'; source.mkdir(); case=source/'case'; case.mkdir()
    (case/'result.json').write_text('{')
    ds=Dataset(tmp_path/'ds'); report=collect_once(ds,[source])
    assert report['counts']['error']==1
    (case/'result.json').write_text('{"status":"error","error_type":"TimeoutError"}')
    assert collect_once(ds,[source])['counts']['partial']==1
    state=json.loads((ds.root/'collector/status.json').read_text())
    assert state['status']=='idle'
    assert state['last_report']['discovered']==1


def test_run_wrapper_records_failed_process_without_physical_failure(tmp_path):
    ds=Dataset(tmp_path/'ds'); source=tmp_path/'exp'; source.mkdir()
    code=run_command(ds,[source],[sys.executable,'-c','raise SystemExit(3)'])
    assert code==3
    events=list((ds.root/'collector/runs').glob('*.json'))
    assert len(events)==1
    event=json.loads(events[0].read_text())
    assert event['exit_code']==3
    assert 'command' not in event  # Do not persist credentials in arbitrary command arguments.


def test_cli_ingest_snapshot_query(tmp_path):
    source=tmp_path/'exp'; source.mkdir()
    ds=tmp_path/'ds'
    command=[sys.executable,'-m','humanoid_mimicgen.experience_dataset','--dataset',str(ds)]
    result=subprocess.run(command+['ingest',str(source)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['discovered']==0
    subprocess.run(command+['snapshot','test'],capture_output=True,check=True)
    result=subprocess.run(command+['query','--snapshot','test','--text','苹果'],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['results']==[]
