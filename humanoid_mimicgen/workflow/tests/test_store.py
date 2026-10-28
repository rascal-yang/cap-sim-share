import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from humanoid_mimicgen.workflow.store import Store
from humanoid_mimicgen.workflow.contracts import WorkflowError, validate_strategy


def strategy():
    return {'schema_version':1,'parent_id':None,'kind':'trajectory_retargeting','adapter':'g1_sonic_native_v1',
            'context':{'scene':'a'},'bindings':{'object':'fruit','hands':['left']},
            'phases':[{'id':'pick','start':4,'end':9,'requires_contact':False,'requires_clearance':False},
                      {'id':'carry','start':9,'end':13,'requires_contact':True,'requires_clearance':True}],
            'command_pose_offset':-1,'source_frame_count':20,'policy':{},'proposal':{'stance':None},
            'evidence':{},'provenance':{'original_sequences_accessible':True}}


def test_atomic_save_and_load_copies_artifacts(tmp_path):
    artifact=tmp_path/'trace';artifact.write_bytes(b'trace')
    store=Store(tmp_path/'store');s=strategy()
    with ThreadPoolExecutor(max_workers=3) as pool:
        ids=list(pool.map(lambda _:store.save(s,{'source/trace':artifact}),range(3)))
    assert len(set(ids))==1
    artifact.unlink()
    loaded=store.load(ids[0]);assert loaded['strategy']['bindings']['hands']==['left']
    assert (loaded['path']/'artifacts/source/trace').read_bytes()==b'trace'
    (loaded['path']/'artifacts/source/trace').write_bytes(b'changed')
    with pytest.raises(WorkflowError,match='integrity'):store.load(ids[0])


def test_context_changes_identity_and_unsafe_paths_rejected(tmp_path):
    store=Store(tmp_path);s=strategy();a=store.save(s,{})
    s['context']['scene']='b';assert store.save(s,{})!=a
    with pytest.raises(WorkflowError):store.load('../escape')
    with pytest.raises(WorkflowError):store.save(s,{'../escape':__file__})


def test_nonfinite_and_invalid_phase_rejected():
    s=strategy();s['context']['bad']=float('nan')
    with pytest.raises(WorkflowError):validate_strategy(s)
    s=strategy();s['phases'][1]['start']=8
    with pytest.raises(WorkflowError):validate_strategy(s)
    s=strategy();s['phases'][0]['start']=0
    with pytest.raises(WorkflowError):validate_strategy(s)


def test_modified_manifest_rejected(tmp_path):
    store=Store(tmp_path);ident=store.save(strategy(),{})
    p=tmp_path/ident/'manifest.json';manifest=json.loads(p.read_text());manifest['strategy']['context']['scene']='changed';p.write_text(json.dumps(manifest))
    with pytest.raises(WorkflowError,match='integrity'):store.load(ident)
