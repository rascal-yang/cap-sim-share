"""Complete-state phase checkpoints for replay navigation, with immutable lineage.

Only navigation-start and parked/pre-transition boundaries are exposed. No active
operation/recovery is partially reconstructed from a trace.
"""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import mujoco
from .retry_state import pack, unpack, task_state
from ..workflow.contracts import file_hash

SONIC_FIELDS=('last','tick','ref','ref_tick','serial','navigating','handoff_until_tick','anchor')
LOOP_FIELDS=('initial_height','poses','desired','source_desired','program_start','stage','native','navigation_done','velocity','face','latched_fingers','source_row','retry_index','attempts','attempt_native','recovery_start','normal_q','translation','object_anchor','integral_bias','events')
PARKING_FIELDS=('waypoint','elapsed','started','quiet','corrections','previous_error','reference_xy','events','plans','done','hold_poses','normal_q')


def options(navigation):
    value=navigation.get('checkpoint',{})
    return value if value.get('enabled',True) else {}


def validate_inputs(parent,child):
    for name in ('scene.mjb','physics.json','task.json','reference.npz','initial.npz','controller_initial.npz','offline_plan.json'):
        if file_hash(Path(child)/name)!=file_hash(Path(parent)/name):raise ValueError('Checkpoint context differs: '+name)


def validate_code(parent,child,parent_repository,child_repository):
    def logical(evidence,repository):
        root=Path(repository)
        return {str(Path(p).relative_to(root)):h for p,h in evidence.items() if Path(p).is_relative_to(root) and Path(p).suffix=='.py'}
    if logical(parent,parent_repository)!=logical(child,child_repository):raise ValueError('Checkpoint executable source hashes differ; create a new seed')


def restore_phase_state(parking,integral,value):
    if value['parking'] is not None and parking is not None:
        for k,v in value['parking']['fields'].items():setattr(parking,k,v)
        parking.profile.velocity=value['parking']['velocity']
    if value['integral'] is not None and integral is not None:integral.bias=value['integral']


def capture_adapter(a):
    kind=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(a.m,kind));mujoco.mj_getState(a.m,a.d,state,kind)
    return dict(integration=state,sonic={k:copy.deepcopy(getattr(a.sonic,k)) for k in SONIC_FIELDS if hasattr(a.sonic,k)},history=copy.deepcopy(list(a.sonic.history)),last_target=copy.deepcopy(a.last_target),task=task_state(a.b.e.task))


def restore_adapter(a,saved):
    kind=mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(a.m,a.d,saved['integration'],kind)
    warm=a.d.qacc_warmstart.copy();mujoco.mj_forward(a.m,a.d);a.d.qacc_warmstart[:]=warm
    for k in SONIC_FIELDS:
        if k in saved['sonic']:setattr(a.sonic,k,copy.deepcopy(saved['sonic'][k]))
        elif hasattr(a.sonic,k):delattr(a.sonic,k)
    a.sonic.history.clear();a.sonic.history.extend(copy.deepcopy(saved['history']))
    a.last_target=copy.deepcopy(saved['last_target'])
    for k,v in saved['task'].items():setattr(a.b.e.task,k,copy.deepcopy(v))
    actual=np.empty_like(saved['integration']);mujoco.mj_getState(a.m,a.d,actual,kind)
    if not np.array_equal(actual,saved['integration']):raise ValueError('Checkpoint integration restore mismatch')


def validate_change(parent,child,name):
    if name not in ('navigation_start','parked'):raise ValueError('Unsupported navigation checkpoint boundary')
    p=copy.deepcopy(parent);c=copy.deepcopy(child)
    for v in (p,c):
        v.pop('repository',None);v['navigation'].pop('checkpoint',None)
    if name=='navigation_start':return
    # Feedback and transition begin only after this measured parking boundary.
    for v in (p,c):
        v.pop('feedback_gain',None)
        for k in ('feedback_gain','orientation_feedback_gain','transition_s','wrist_integral'):v['navigation'].pop(k,None)
    if p!=c:raise ValueError('Checkpoint inherits changed navigation or earlier policy inputs')


def validate_models(parent,child):
    if parent!=child:raise ValueError('Checkpoint model hashes differ')


def verify_boundary(states,nexts,restored):
    if len(states) and (not np.array_equal(nexts[-1],restored) or not np.array_equal(np.asarray(states)[1:],np.asarray(nexts)[:-1])):
        raise ValueError('Checkpoint state boundary is not continuous')


def capture(case,name,a,scope):
    folder=Path(case)/'execution/checkpoints';folder.mkdir(exist_ok=True)
    path=folder/(name+'.json.gz')
    if path.exists():return
    if scope['program'] is not None or scope['recovery'] is not None:raise ValueError('Only pre-operation checkpoints supported')
    parking=scope['parking']
    value=dict(version=1,name=name,step=scope['step'],adapter=capture_adapter(a),loop={k:copy.deepcopy(scope[k]) for k in LOOP_FIELDS},program=None,recovery=None,
               parking=None if parking is None else dict(fields={k:copy.deepcopy(getattr(parking,k)) for k in PARKING_FIELDS},velocity=parking.profile.velocity.copy()),
               integral=None if scope['integral'] is None else copy.deepcopy(scope['integral'].bias),
               prefix_counts=dict(tokens=len(a.sonic.records),planner=len(getattr(a.sonic,'planner_records',[])),navigation=len(scope['navigation'])))
    with path.open('xb') as f:f.write(gzip.compress(json.dumps(pack(value),separators=(',',':')).encode(),mtime=0))


def bind_contract(case,contract,cfg):
    option=options(cfg['navigation'])
    if not option:return
    helper=Path(__file__).resolve();snapshot=Path(case)/'source_snapshot/navigation_checkpoint.py';snapshot.write_bytes(helper.read_bytes())
    evidence=contract['preflight']['evidence_sha256'];evidence[str(helper)]=file_hash(helper);evidence[str(snapshot)]=file_hash(snapshot)
    contract['control']['navigation_checkpoints']=dict(boundaries=['navigation_start','parked'],boundary_kind='pre_control_50Hz; parked after parking update before first transition command',complete_state='mjSTATE_INTEGRATION, SONIC history/reference/serial/handoff, task counters, adapter target, parking and loop phase state',active_program_and_recovery='None at supported boundaries',body_and_hands='Inherited prefix plus newly executed suffix; all original action fields retained')
    replay=option.get('token_replay_case')
    if replay:
        parent=Path(replay)
        for name in ('scene.mjb','physics.json','task.json','reference.npz','initial.npz','controller_initial.npz'):
            if file_hash(Path(case)/name)!=file_hash(parent/name):raise ValueError('Token replay context differs: '+name)
        for p in (parent/'execution/sonic_actions.npz',parent/'execution/trajectory.npz',parent/'result.json',parent/'runtime.json'):
            evidence[str(p)]=file_hash(p)
        validate_models(json.loads((parent/'runtime.json').read_text())['actual_loaded_model_files'],json.loads((Path(case)/'runtime.json').read_text())['actual_loaded_model_files'])
        contract['purpose']['hypothesis']='Independently integrate complete saved branch tokens and hand commands from original far initial state; compare every state/target/torque'
        contract['control']['stage_interfaces']=[dict(stage='whole_saved_branch',mode='saved SONIC token through original decoder',lower_body_interface='saved body token, measured live proprioception',upper_body_interface='same saved token; synchronized saved14 hand commands at200Hz',target_source=str(parent/'execution/sonic_actions.npz'),encoder_calls=0,planner_calls=0,decoder_overrides=False)]
        contract['pipeline']['entrypoint']='frozen independent token replay validation script named in api_provenance evidence'
        contract['pipeline']['ordered_steps']=['initialize original far state and source decoder history','decode saved token at50Hz using live measured history','apply original PD and saved hands at200Hz','compare complete state/action arrays and native result']
        contract['control']['navigation_checkpoints']['capture_enabled']=False
    resume=option.get('resume')
    if resume:
        parent=Path(resume['case']);manifest_path=parent/'execution/navigation_checkpoint_manifest.json'
        manifest=json.loads(manifest_path.read_text());validate_change(json.loads((parent/'config.json').read_text()),cfg,resume['name'])
        if resume['name'] not in manifest['checkpoints']:raise ValueError('Checkpoint not saved')
        for path,digest in manifest['sha256'].items():
            if file_hash(path)!=digest:raise ValueError('Parent checkpoint evidence changed: '+path)
            evidence[path]=digest
        evidence[str(manifest_path)]=file_hash(manifest_path)
        validate_models(json.loads((parent/'runtime.json').read_text())['actual_loaded_model_files'],json.loads((Path(case)/'runtime.json').read_text())['actual_loaded_model_files'])
        validate_inputs(parent,case)
        parent_contract=json.loads((parent/'experiment.json').read_text())
        parent_cfg=json.loads((parent/'config.json').read_text())
        validate_code(parent_contract['preflight']['evidence_sha256'],evidence,parent_cfg['repository'],cfg['repository'])
        parent_runtime=parent/'runtime.json'
        if file_hash(parent_runtime)!=parent_contract['preflight']['evidence_sha256'][str(parent_runtime)]:raise ValueError('Parent runtime evidence changed')
        contract['purpose']['known_confounders'].append('Only suffix newly integrated; ancestor prefix inherited, not a fresh continuous rollout')
        contract['physics']['action_permissions']='One complete checkpoint restoration before branch integration; SONIC torques thereafter'
        contract['cases'][0]['initialization']='complete_navigation_checkpoint_once: '+str(resume)
        contract['control']['checkpoint_initialization']=resume
        contract['pipeline']['ordered_steps'].insert(0,'verify complete ancestor and restore checkpoint once; inherit synchronized prefix')
        contract['purpose']['invariants']=['Scene, physics, PD, native predicate unchanged','SONIC mode1 and source hands; one explicit checkpoint initialization, no midrun resets']


def restore(case,a,cfg,parking,integral):
    resume=cfg['navigation']['checkpoint']['resume'];parent=Path(resume['case'])
    with gzip.open(parent/'execution/checkpoints'/ (resume['name']+'.json.gz'),'rt') as f:value=unpack(json.load(f))
    if value['version']!=1:raise ValueError('Checkpoint schema unsupported')
    validate_change(json.loads((parent/'config.json').read_text()),cfg,resume['name'])
    restore_adapter(a,value['adapter'])
    restore_phase_state(parking,integral,value)
    count=value['step'];execution=parent/'execution'
    with np.load(execution/'trajectory.npz') as tr:
        prefix={k:list(tr[n][:count].copy()) for k,n in [('states','state'),('nexts','next_state'),('ctrls','commands')]}
    for name in ('commands','observations'):prefix[name]=json.loads((execution/(name+'.json')).read_text())[:count]
    prefix['navigation']=json.loads((execution/'navigation.json').read_text())[:value['prefix_counts']['navigation']]
    a.sonic.records=json.loads((execution/'controller.json').read_text())[:value['prefix_counts']['tokens']]
    a.sonic.planner_records=json.loads((execution/'planner.json').read_text())[:value['prefix_counts']['planner']]
    verify_boundary(prefix['states'],prefix['nexts'],np.r_[a.d.time,a.d.qpos,a.d.qvel])
    value['prefix']=prefix
    value['lineage']=dict(case=str(parent),checkpoint=resume['name'],inherited_physics_steps=count,inherited_token_count=value['prefix_counts']['tokens'],manifest_sha256=file_hash(execution/'navigation_checkpoint_manifest.json'))
    return value


def finalize(case):
    case=Path(case);folder=case/'execution/checkpoints'
    if not folder.exists():return
    paths=list(folder.glob('*.json.gz'))
    paths += [case/n for n in ['scene.mjb','physics.json','task.json','reference.npz','initial.npz','controller_initial.npz','config.json','runtime.json','offline_plan.json','experiment.json','experiment.sha256','result.json']]
    paths += [case/'execution'/n for n in ['trajectory.npz','commands.json','observations.json','navigation.json','controller.json','planner.json','sonic_actions.npz']]
    value=dict(version=1,checkpoints=[p.name.removesuffix('.json.gz') for p in folder.glob('*.json.gz')],sha256={str(p):file_hash(p) for p in paths},snapshot_bytes=sum(p.stat().st_size for p in folder.glob('*.json.gz')))
    with (case/'execution/navigation_checkpoint_manifest.json').open('x') as f:json.dump(value,f,indent=2)
