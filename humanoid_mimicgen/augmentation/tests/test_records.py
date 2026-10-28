from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import pytest
from humanoid_mimicgen.augmentation.records import create_record, index_history


def test_new_record_uses_shanghai_date_and_preserves_collision(tmp_path):
    clock=datetime(2026,10,21,16,7,tzinfo=ZoneInfo('Asia/Shanghai'))
    first=create_record(tmp_path,'test',clock);(first/'keep').write_text('original')
    second=create_record(tmp_path,'test',clock)
    assert first.relative_to(tmp_path).as_posix()=='2026-10/21/20261021_1607_test'
    assert second.name=='20261021_1607_test_01'
    assert (first/'keep').read_text()=='original'
    assert json.loads((first/'creation.json').read_text())['created_at'].endswith('+08:00')


def test_history_index_preserves_paths_bytes_and_date_only_names(tmp_path):
    old=tmp_path/'20261021_old';old.mkdir();(old/'frozen.json').write_bytes(b'untouched')
    unknown=tmp_path/'apple_pipeline_v5';unknown.mkdir()
    x=index_history(tmp_path)
    alias=tmp_path/'2026-10/21'/old.name
    assert alias.is_symlink() and alias.resolve()==old
    assert old.is_dir() and not old.is_symlink()
    assert (old/'frozen.json').read_bytes()==b'untouched'
    assert x['unclassified']==['apple_pipeline_v5']
    assert index_history(tmp_path)['created_links']==0


def test_conflicting_history_entry_is_never_overwritten(tmp_path):
    (tmp_path/'20261021_old').mkdir()
    target=tmp_path/'2026-10/21/20261021_old';target.mkdir(parents=True)
    with pytest.raises(FileExistsError):index_history(tmp_path)
    assert not target.is_symlink()


def test_relocation_verifies_original_bytes_and_creates_only_derived_files(tmp_path):
    from humanoid_mimicgen.augmentation.relocation import rebase_baseline
    from humanoid_mimicgen.workflow.contracts import file_hash,WorkflowError
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir();data=new/'data';data.write_text('frozen')
    task=new/'task.json';task.write_text(json.dumps({'resources':{'initial_state':str(old/'data')}}))
    record=new/'record.json';record.write_text(json.dumps({'pipeline':{},'preflight':{'evidence_sha256':{str(old/'data'):file_hash(data)}}}))
    request=new/'request.json';request.write_text(json.dumps({'files':{'task':str(old/'task.json'),'record':str(old/'record.json')}}))
    hashes={p:file_hash(p) for p in (data,task,record,request)}
    mapping=tmp_path/'map.json';mapping.write_text(json.dumps({str(old):str(new)}))
    out=rebase_baseline(request,tmp_path/'derived',mapping)
    assert out.exists() and all(file_hash(p)==h for p,h in hashes.items())
    derived=json.loads((out.parent/'task.json').read_text());assert derived['resources']['initial_state']==str(data)
    data.write_text('corrupted')
    with pytest.raises(WorkflowError,match='integrity_error'):rebase_baseline(request,tmp_path/'rejected',mapping)
    assert not (tmp_path/'rejected').exists()
