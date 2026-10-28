import copy
import json
import pytest
from humanoid_mimicgen.experience_dataset.schema import new_record
from humanoid_mimicgen.experience_dataset.store import Dataset, file_hash
from humanoid_mimicgen.experience_dataset.retrieval import query_experience, get_experience
from humanoid_mimicgen.experience_dataset.planner import ExperiencePlanner


def add(ds, key, text, success=True, family=None):
    r = new_record(key, task='apple', producer='test')
    r['context'].update(robot='G1', hand='Dex3', interface='mode1', controller_version='v1', actual_modes=[1])
    r['orchestration'] = [{'instruction_id': key, 'instruction': {'goal': text}}]
    r['control'] = [{'instruction_id': key, 'response_class': 'opposite', 'measurements': {'wrist_dz': -.01}}]
    r['outcome']['task_success'] = success
    if family: r['relations']['family'] = family
    ev = ds.root/(key+'.json'); ev.write_text('{}')
    r['evidence'] = [{'path': str(ev), 'sha256': file_hash(ev), 'role': 'result'}]
    ds.put(r)
    return r


def test_chinese_search_and_control_compatibility(tmp_path):
    ds = Dataset(tmp_path); a=add(ds, 'a', '苹果抬升不足'); add(ds, 'b', '导航停稳')
    ds.snapshot('s')
    q = {'snapshot': 's', 'text': '抬升不足', 'kind': 'orchestration'}
    result = query_experience(ds, q)
    assert result['results'][0]['attempt_id'] == a['attempt_id']
    assert result['results'][0]['control'] == []
    assert query_experience(ds, dict(q, kind='control'))['results'] == []
    control = dict(q, kind='control', robot='G1', hand='Dex3', interface='mode1', controller_version='v1', actual_modes=[1])
    assert query_experience(ds, control)['results']
    assert query_experience(ds, dict(control, interface='mode0'))['results'] == []
    assert query_experience(ds, dict(control, hand='Revo2'))['results'] == []


def test_snapshot_exclusions_and_family_diversity(tmp_path):
    ds=Dataset(tmp_path); a=add(ds,'a','抓取',family='shared'); add(ds,'b','抓取',family='shared')
    ds.snapshot('s'); add(ds,'c','抓取')
    result=query_experience(ds, {'snapshot':'s','kind':'orchestration','text':'抓取'})
    assert len(result['results']) == 1
    assert query_experience(ds, {'snapshot':'s','excluded_families':['shared']})['results'] == []
    assert query_experience(ds, {'snapshot':'s','current_attempt_id':a['attempt_id']})['results'] == []


def test_cross_experiment_lineage_is_excluded_as_a_group(tmp_path):
    ds=Dataset(tmp_path); a=add(ds,'a','抓取',family='one'); b=add(ds,'b','抓取',family='two')
    b['relations']['parent']=a['attempt_id']; ds.put(b)
    ds.snapshot('s')
    assert query_experience(ds, {'snapshot':'s','excluded_families':['one']})['results']==[]


def test_small_budget_keeps_useful_compact_evidence(tmp_path):
    ds=Dataset(tmp_path); add(ds,'a','抓取'); ds.snapshot('s')
    result=query_experience(ds, {'snapshot':'s','text':'抓取','max_chars':1500})
    assert result['results']
    assert len(json.dumps(result,ensure_ascii=False))<=1500


def test_chinese_query_finds_english_native_task(tmp_path):
    ds=Dataset(tmp_path); r=add(ds,'a','increase lift height')
    r['task_id']='G1WholebodyTableApplePickTeleop'; ds.put(r); ds.snapshot('s')
    assert query_experience(ds,{'snapshot':'s','text':'苹果抓取'})['results']


def test_snapshot_get_cannot_read_excluded_or_future_revision(tmp_path):
    ds=Dataset(tmp_path); r=add(ds,'a','抓取',family='excluded'); ds.snapshot('s',excluded_families=['excluded'])
    with pytest.raises(ValueError,match='outside'):
        get_experience(ds,r['attempt_id'],1,snapshot='s')
    ds.snapshot('included'); r['outcome']['task_success']=False; ds.put(r)
    with pytest.raises(ValueError,match='outside'):
        get_experience(ds,r['attempt_id'],2,snapshot='included')


def test_many_replays_do_not_hide_independent_experience(tmp_path):
    ds=Dataset(tmp_path)
    for i in range(31): add(ds,str(i),'抓取 接触',family='replays')
    independent=add(ds,'independent','抓取',family='independent')
    ds.snapshot('s')
    results=query_experience(ds,{'snapshot':'s','text':'抓取 接触'})['results']
    assert any(r['attempt_id']==independent['attempt_id'] for r in results)


def test_same_source_can_supply_success_and_failure_but_not_duplicate_replays(tmp_path):
    ds=Dataset(tmp_path)
    add(ds,'success','抓取',True,family='source')
    add(ds,'replay','抓取',True,family='source')
    add(ds,'failure','抓取',False,family='source')
    ds.snapshot('s')
    results=query_experience(ds,{'snapshot':'s','text':'抓取'})['results']
    assert len(results)==2
    assert {r['outcome']['task_success'] for r in results}=={True,False}


def test_mutated_evidence_excluded_and_get_never_returns_arbitrary_path(tmp_path):
    ds=Dataset(tmp_path); r=add(ds,'a','抓取'); ds.snapshot('s')
    (tmp_path/'a.json').write_text('{"changed":true}')
    assert query_experience(ds, {'snapshot':'s'})['results'] == []
    with pytest.raises(ValueError): get_experience(ds,'../../secret',1)


def test_context_budget_and_snapshot_planner(tmp_path):
    ds=Dataset(tmp_path); add(ds,'a','抓取'*5000); ds.snapshot('s')
    q={'snapshot':'s', 'kind':'orchestration', 'max_chars':2000}
    r=query_experience(ds,q)
    assert len(json.dumps(r, ensure_ascii=False)) <= 2000
    class Delegate:
        def plan(self, request): return request
    adapter=ExperiencePlanner(Delegate(),ds,'s',{'kind':'orchestration'})
    original={'attempt':1, 'task':{}, 'observation':{'problem':'抓取'}}
    saved=copy.deepcopy(original); response=adapter.plan(original)
    assert original == saved
    assert response['experience_context']['snapshot'] == 's'
    add(ds,'new','抓取')
    assert adapter.plan(original)['experience_context']['snapshot_sha256'] == response['experience_context']['snapshot_sha256']
