"""Revalidate an immutable, API-accepted solution for an identical condition."""
import json
from pathlib import Path
import numpy as np
from .quality_suite import verify_origin
from .suite_physics import validate_guard
from ..workflow.contracts import file_hash


def native_pass(r):
    return r.get('success') is True and r.get('independent_success') is True and r.get('coverage')==1 and not r.get('failure')


def validate_identity(result,plan,spec,params,shift):
    if not native_pass(result):raise ValueError('Seed must have full independent native success')
    if result['task']!=spec['task'] or result['episode']!=params['episode'] or result['object_delta_xy_m']!=list(shift) or result['mode']!='token_prefix_mode1':raise ValueError('Seed condition differs; use API planning for new conditions')
    if plan['episode']!=params['episode'] or Path(plan['reference_case']).resolve()!=Path(spec['baseline']).resolve() or plan['contact_start']!=params['contact_start'] or validate_guard(plan.get('safety_guard'))!=validate_guard(params.get('safety_guard')):raise ValueError('Seed reference/annotation/guard differs')
    if plan.get('resume',{}).get('kind')!='restart':raise ValueError('Only restart seeds can be reused without inheriting historical execution')


def load_seed(path,spec,params,shift):
    item=json.loads(Path(path).read_text())
    if item.get('accepted') is not True:raise ValueError('Unaccepted candidate cannot seed execution')
    result=item['result'];case=Path(result['experiment_path']);stored=json.loads((case/'result.json').read_text())
    if result!=stored:raise ValueError('Seed result differs from physical case')
    review=verify_origin(item['review_path'])
    if review['decision'].get('action')!='accept':raise ValueError('Seed needs original API acceptance, not a local or incomplete review')
    request=json.loads(Path(review['origin']['request']).read_text());content=request['input'][0]['content'];ctx=json.loads(next(x['text'] for x in content if x['type']=='input_text') if isinstance(content,list) else content)
    if ctx['criteria_audit']['assessment']!='native_valid':raise ValueError('Seed must use unchanged native criteria')
    evidence=ctx['evidence']['source_hashes']
    for p in (case/'result.json',case/'scene.mjb',case/'execution/trajectory.npz'):
        if evidence.get(str(p))!=file_hash(p):raise ValueError('Accepted seed evidence changed')
    contract=case/'experiment.json'
    if file_hash(contract)!=(case/'experiment.sha256').read_text().strip():raise ValueError('Seed contract changed')
    for p,digest in json.loads(contract.read_text())['physics']['acceptance_source_and_hash'].items():
        if file_hash(p)!=digest:raise ValueError('Native acceptance code changed since seed')
    dp=Path(result['decision_path']);plan=verify_origin(dp)['decision']
    if ctx['executed_parameters']!=plan:raise ValueError('Seed plan differs from reviewed execution')
    validate_identity(result,plan,spec,params,shift)
    return {'case':str(case),'plan':plan,'decision_path':str(dp),'source_review':item['review_path'],'source_item':str(path),'source_item_sha256':file_hash(path),'source_result_sha256':file_hash(case/'result.json'),'evidence_files':[str(path),str(case/'result.json'),str(case/'scene.mjb'),str(case/'execution/trajectory.npz'),item['review_path']]}


def exact_revalidation(source_case,result):
    a=Path(source_case);b=Path(result['experiment_path']);scene=file_hash(a/'scene.mjb')==file_hash(b/'scene.mjb')
    with np.load(a/'execution/trajectory.npz',allow_pickle=False) as old,np.load(b/'execution/trajectory.npz',allow_pickle=False) as new:
        equal={k:np.array_equal(old[k],new[k]) for k in ('state','next_state','commands')}
    return {'passed':native_pass(result) and scene and all(equal.values()),'native_and_independent_passed':native_pass(result),'identical_scene':scene,'identical_arrays':equal,
            'source_case':str(a),'new_case':str(b),'acceptance_authority':'local_exact_seed_revalidation','new_api_review':False}
