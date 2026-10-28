"""Bind standalone navigation evidence to existing frozen workflow execution."""
import json
from pathlib import Path
import numpy as np
from ..contracts import WorkflowError, file_hash
from .archive import verify_evidence


def validate_bound_plan(plan, files, strategy):
    try:
        if plan['kind']!='replay_navigation_plan' or plan['status']!='completed' or not plan['selected']['operation_checked']:
            raise WorkflowError('preflight_failed','Navigation plan is not completed')
        verify_evidence(plan['evidence_sha256'])
        for key in ('model','initial'):
            if file_hash(files[key])!=plan['scene_input_sha256'][key]:
                raise WorkflowError('context_changed','Navigation scene changed: '+key)
        if file_hash(files['reference_states'])!=plan['source_trajectory_sha256']:
            raise WorkflowError('context_changed','Navigation source reference changed')
        entry=strategy['phases'][0]['start']+strategy['command_pose_offset']
        if entry!=plan['prior']['operation_interval'][0]:raise WorkflowError('context_changed','Navigation phase alignment')
        if strategy['bindings']['object']!=plan['prior']['bindings']['object']:
            raise WorkflowError('context_changed','Navigation object binding')
        nav=strategy['policy']['navigation']
        if nav.get('profile')!='replay_navigation':raise WorkflowError('invalid_input','Explicit replay_navigation profile required')
        if nav['route_limits']['corridor_radius_m']>plan['native_options']['corridor_margin_m']:
            raise WorkflowError('preflight_failed','Runtime corridor exceeds checked margin')
        if nav.get('navigation_only') is False:
            validate_operation_binding(plan,strategy)
            if plan.get('reference_mode')=='apple_lateral_v1':
                from .lateral_execution import validate_generated_binding
                validate_generated_binding(plan,strategy)
        elif nav.get('navigation_only') is not True:
            raise WorkflowError('invalid_input','Explicit navigation_only boolean required')
        if strategy['policy'].get('grasp_trigger'):
            raise WorkflowError('invalid_input','Navigation execution must not invoke GPT')
        return plan['selected']
    except (KeyError,TypeError) as exc:
        raise WorkflowError('preflight_failed','Incomplete navigation plan binding: '+str(exc)) from None


def measured_handoff(plan,current,output_dir=None):
    """Recheck the whole reference from measured stance without stepping physics."""
    if plan.get('reference_mode')=='apple_lateral_v1':
        from .lateral_execution import regenerate_handoff
        if output_dir is None:raise WorkflowError('invalid_input','Derived handoff output required')
        return regenerate_handoff(plan,current,output_dir)
    import mujoco
    from .archive import load_archive
    from .native import NativeEvaluator
    from .prior import yaw
    verify_evidence(plan['evidence_sha256'])
    trace,bindings,_=load_archive(plan['source_manifest'])
    model=mujoco.MjModel.from_binary_path(plan['scene_files']['model'])
    evaluator=NativeEvaluator(model,current,trace['qpos'],plan['prior'],plan['scene']['task_frame_xyzyaw'],bindings,plan['native_options'])
    result=evaluator({'id':'measured_handoff','xyyaw':[*current[:2],float(yaw(current[:7]))]})
    return result


def execute_plan(plan_path,experiment):
    from ..service import check_contract
    from ..runner import execute_native
    path=Path(plan_path).resolve();root=Path(experiment).resolve();contract=root/'experiment.json'
    c=check_contract(contract);plan=json.loads(path.read_text())
    if c['preflight']['evidence_sha256'].get(str(path))!=file_hash(path):
        raise WorkflowError('preflight_failed','Navigation plan not bound in frozen contract')
    selected=validate_bound_plan(plan,c['inputs'],c['workflow']['strategy'])
    if c['workflow']['selection']['xyyaw']!=selected['xyyaw'] or c['workflow']['selection']['navigation_path']!=selected['navigation_path']:
        raise WorkflowError('context_changed','Frozen navigation route differs from plan')
    if c['workflow']['strategy']['policy']['navigation'].get('plan_file')!=str(path):
        raise WorkflowError('preflight_failed','Frozen navigation plan_file mismatch')
    audit=execute_native(contract)
    operation=c['workflow']['strategy']['policy']['navigation']['navigation_only'] is False
    success=bool(audit['completed'] and audit.get('native_task_success')) if operation else None
    result={'status':'completed' if audit['completed'] else 'failed',
            'scope':'whole_body_operation_native' if operation else 'navigation_handoff_only',
            'experiment_id':root.name,'success':success,'task_success':success,
            'navigation_success':audit.get('navigation_handoff_ready',audit['completed']),
            'native_success':audit.get('native_task_success') if operation else None,
            'navigation_admission':c['workflow']['strategy']['provenance'].get('navigation_admission'),
            'training_ready':False,'audit':audit}
    from ..service import write_new
    write_new(root/'result.json',result)
    return result


def load_navigation_source(path,plan):
    manifest=json.loads(Path(plan['source_manifest']).read_text())
    with np.load(path,allow_pickle=False) as archive:
        source=archive[manifest.get('state_key','state')].copy()
    if source.ndim!=2 or not np.isfinite(source).all():
        raise WorkflowError('invalid_input','Navigation reference states')
    return source


def validate_operation_binding(plan,strategy):
    from .operation import validate_operation
    phases=strategy['phases'];start=phases[0]['start'];end=phases[-1]['end'];offset=strategy['command_pose_offset']
    config=validate_operation(strategy['policy']['whole_body_operation'],start,end)
    if [start+offset,end-1+offset]!=plan['prior']['operation_interval']:
        raise WorkflowError('context_changed','Whole operation range must match checked reference')
    required={strategy['bindings']['root_body'],strategy['bindings']['torso'],*strategy['bindings']['feet'].values(),*plan['prior']['bindings']['operation_bodies']}
    if not required<=set(config['alignment_bodies']):
        raise WorkflowError('invalid_input','Alignment must include root, both feet and operation bodies')
    if any(p.get('requires_clearance',False) for p in phases):
        raise WorkflowError('unsupported_capability','Whole-body clearance phase gates require a task-specific clearance observer; use native acceptance')
    gates={g['frame'] for g in config['contact_gates']}
    if phases[0]['requires_contact'] or any(p['requires_contact'] and p['start'] not in gates for p in phases[1:]):
        raise WorkflowError('invalid_input','Contact phases require explicit body-group gates')


def validate_operation_inputs(plan,files,strategy,physics_dt):
    source=load_navigation_source(files['reference_states'],plan)
    start=strategy['phases'][0]['start'];end=strategy['phases'][-1]['end']
    if len(source)<end or source.shape[1]<8 or not np.allclose(np.diff(source[:,0]),physics_dt,atol=1e-9,rtol=0):
        raise WorkflowError('unsupported_capability','Operation source resampling is not implicit')
    with np.load(files['targets'],allow_pickle=False) as archive:fingers=archive['fingers']
    if fingers.shape!=(len(source),14) or not np.isfinite(fingers).all():
        raise WorkflowError('invalid_input','Source fingers must align with each reference row')
    if not np.allclose(fingers[start],strategy['policy']['whole_body_operation']['open_fingers_rad'],atol=1e-7,rtol=0):
        raise WorkflowError('invalid_input','Operation entry must use declared open fingers')
    if plan.get('reference_mode')=='apple_lateral_v1':
        from .lateral_execution import validate_generated_binding
        from .lateral_prior import load_lateral_source
        bundle=validate_generated_binding(plan,strategy)
        _,_,recorded,_=load_lateral_source(plan['source_manifest'],plan['lateral_options'])
        rows=bundle['source_rows']
        if (not np.array_equal(bundle['fingers'],recorded[rows]) or not np.array_equal(fingers[rows],recorded[rows]) or
            not np.allclose(bundle['time_s'],source[rows,0],atol=1e-9,rtol=0)):
            raise WorkflowError('context_changed','Generated/source command or timing mismatch')
