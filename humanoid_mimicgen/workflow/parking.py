"""Measured stopping-response identification and bounded parking decisions.

No model/API calls, simulator state writes or task-specific positional offsets.
A fitted first-order response is a local estimate, not a dynamics guarantee.
"""
from collections import deque
import json
from pathlib import Path
import numpy as np
from .contracts import WorkflowError, canonical, digest, file_hash, now, positive


class RecentWindow:
    """Same inclusion rule as old full-history scans, bounded memory and work."""
    def __init__(self, seconds, physics_dt):
        positive(seconds,'window');positive(physics_dt,'physics dt')
        self.seconds=seconds;self.dt=physics_dt;self.rows=deque()
    def append(self,row):
        self.rows.append(row);cutoff=row['time_s']-self.seconds-self.dt/2
        while self.rows and self.rows[0]['time_s']<cutoff:self.rows.popleft()


def fit_stop_response(velocities, displacements):
    v=np.asarray(velocities,float);d=np.asarray(displacements,float)
    if v.ndim!=2 or v.shape[1]!=2 or v.shape!=d.shape or len(v)==0 or not np.isfinite(v).all() or not np.isfinite(d).all():
        raise WorkflowError('invalid_input','Stopping calibration arrays')
    energy=float(np.sum(v*v))
    if energy<=np.finfo(float).eps:raise WorkflowError('needs_calibration','No measured motion')
    tau=float(np.sum(v*d)/energy)
    if tau<=0:raise WorkflowError('needs_calibration','Stopping displacement opposes velocity')
    errors=np.linalg.norm(d-tau*v,axis=1)
    return {'tau_s':tau,'sample_count':len(v),'unique_sample_count':len({tuple(np.r_[a,b]) for a,b in zip(v,d)}),'residual_rmse_m':float(np.sqrt(np.mean(errors**2))),
            'residual_max_m':float(max(errors)),'observed_speed_min_m_s':float(min(np.linalg.norm(v,axis=1))),
            'observed_speed_max_m_s':float(max(np.linalg.norm(v,axis=1))),
            'generalization_validated':False}


def predicted_stop(xy,velocity,model):
    positive(model['tau_s'],'stopping tau')
    xy=np.asarray(xy,float);velocity=np.asarray(velocity,float)
    if xy.shape!=(2,) or velocity.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(velocity).all():
        raise WorkflowError('invalid_input','Stopping input')
    return xy+velocity*model['tau_s']


def correction_decision(error,previous_error,count,options):
    positive(error,'parking error',zero=True)
    if error<=options['tolerance_m']:return 'within_tolerance'
    if count>=options['max_corrections']:return 'budget_exhausted'
    if previous_error is not None and previous_error-error<options['minimum_progress_m']:return 'no_progress'
    return 'correct'


def response_scope(strategy, model_sha256, source_sha256, physics_sha256, sonic_models):
    nav=strategy['policy']['navigation']
    return {'adapter':strategy['adapter'],'model_sha256':model_sha256,'source_sha256':source_sha256,
            'physics_sha256':physics_sha256,'sonic_models':{Path(k).name:v for k,v in sonic_models.items()},'bindings':strategy['bindings'],
            'source_entry':strategy['phases'][0]['start']+strategy['command_pose_offset'],
            'navigation':{k:v for k,v in nav.items() if k not in ('parking','correction')},
            'settle':strategy['policy']['settle'],'control_dt_s':strategy['policy']['control_dt_s']}


def calibrate_stopping(experiments,output):
    """Read actual archived runs. Never edit old contracts or advance physics."""
    if not experiments:raise WorkflowError('invalid_input','No calibration experiments')
    rows=[];scope=None;seen=set();evidence={}
    for value in experiments:
        root=Path(value).resolve()
        if root in seen:raise WorkflowError('invalid_input','Duplicate calibration run')
        seen.add(root);contract=root/'experiment.json'
        if file_hash(contract)!=(root/'experiment.sha256').read_text().strip():raise WorkflowError('integrity_error','Historical contract')
        c=json.loads(contract.read_text());s=c['workflow']['strategy'];p=root/'execution'
        providers=json.loads((p/'actual_providers.json').read_text());audit=json.loads((p/'execution_audit.json').read_text())
        if len(providers)!=3 or any(v!=['CPUExecutionProvider'] for v in providers.values()) or audit['actual_encoder_modes']!=[1] or audit['midrun_state_resets']:
            raise WorkflowError('needs_calibration','Unsupported measured interface')
        model_sha=file_hash(c['inputs']['model']);source_sha=file_hash(c['inputs']['reference_states']);physics_sha=file_hash(c['physics']['physics_parameters_manifest'])
        candidate=response_scope(s,model_sha,source_sha,physics_sha,json.loads((p/'actual_model_files.json').read_text()))
        if scope is None:scope=candidate
        if scope!=candidate:raise WorkflowError('needs_calibration','Calibration scopes differ')
        events=json.loads((p/'events.json').read_text());obs=json.loads((p/'observations.json').read_text())
        # Old runs contain one navigation stop and a measured_stance_plan. New
        # runs use navigation_settled so each correction creates a distinct pair.
        stops=[e for e in events if e['event']=='navigation_stop_requested']
        for i,stop in enumerate(stops):
            end=stops[i+1]['time_s'] if i+1<len(stops) else float('inf')
            settled=next((e for e in events if e['event'] in ('navigation_settled','measured_stance_plan') and stop['time_s']<e['time_s']<end),None)
            if settled is None:continue
            at=min(obs,key=lambda o:abs(o['time_s']-stop['time_s']))
            v=stop.get('measured_velocity_xy_m_s',at['root_velocity'][:2])
            delta=(np.asarray(settled['actual_root'][:2])-stop['actual_root'][:2]).tolist()
            rows.append({'experiment_id':root.name,'stop_time_s':stop['time_s'],'settled_time_s':settled['time_s'],
                         'velocity_xy_m_s':v,'displacement_xy_m':delta,'foot_ground_N':at['foot_ground_N']})
        for f in (contract,root/'experiment.sha256',p/'events.json',p/'observations.json',p/'actual_providers.json',p/'actual_model_files.json',p/'execution_audit.json'):
            evidence[str(f)]=file_hash(f)
    if not rows:raise WorkflowError('needs_calibration','No measured stop/settle pairs')
    result={'schema_version':2,'created_at':now(),'kind':'zero_intercept_velocity_response','scope':scope,
            'model':fit_stop_response([r['velocity_xy_m_s'] for r in rows],[r['displacement_xy_m'] for r in rows]),
            'samples':rows,'evidence_sha256':evidence,'api_calls':0,'physics_steps':0}
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(canonical(result))
    return {'status':'completed','calibration_path':str(target.resolve()),'sha256':file_hash(target),'model':result['model'],'physics_steps':0}


def validate_calibration(artifact,scope,quality):
    if artifact.get('schema_version')!=2 or artifact.get('kind')!='zero_intercept_velocity_response' or artifact.get('scope')!=scope:
        raise WorkflowError('needs_calibration','Stopping model scope mismatch')
    for path,sha in artifact['evidence_sha256'].items():
        if not Path(path).is_file() or file_hash(path)!=sha:raise WorkflowError('integrity_error','Stopping evidence changed')
    rows=artifact['samples'];fit=fit_stop_response([r['velocity_xy_m_s'] for r in rows],[r['displacement_xy_m'] for r in rows])
    if digest(fit)!=digest(artifact['model']):raise WorkflowError('integrity_error','Stopping fit mismatch')
    if fit['unique_sample_count']<quality['minimum_samples'] or fit['residual_max_m']>quality['max_residual_m']:
        raise WorkflowError('needs_calibration','Stopping model has insufficient distinct samples or excessive residual')
    return fit


def support_contains(points,xy):
    """Convex support test from measured foot-ground contact points."""
    from scipy.spatial import ConvexHull,QhullError
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():return False
    try:hull=ConvexHull(points)
    except QhullError:return False
    return bool(np.all(hull.equations[:,:2]@np.asarray(xy,float)+hull.equations[:,2]<=1e-10))


def planted_stance_reference(model,current,goal,bindings,policy,contact_events):
    """Plan new leg targets with both feet fixed in world. No live state writes.

    This is newly planned leg IK reference through SONIC encoder1, not mode0 or
    direct joint execution. Geometry and static COM screening are not dynamics.
    """
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from ..stationary_planning import StationaryPlanner
    from .motion import upright_root_reference
    d=mujoco.MjData(model);d.qpos[:]=current;mujoco.mj_forward(model,d)
    feet=[model.body(bindings['feet'][s]).id for s in ('left','right')]
    targets=[(d.xpos[b].copy(),d.xquat[b].copy()) for b in feet]
    joints=[j for j in range(model.njnt) if any(part in model.joint(j).name for part in ('hip_','knee_','ankle_'))]
    if len(joints)!=12:raise WorkflowError('unsupported_capability','Expected G1 twelve leg joints')
    adr=model.jnt_qposadr[joints];bounds=model.jnt_range[joints];q=np.array(current,copy=True)
    q[:7]=upright_root_reference(current[:7]) if policy['navigation'].get('standing_orientation')=='upright' else q[:7]
    q[:2]=goal[:2]
    def rot(quat):return Rotation.from_quat(np.roll(quat,-1))
    def residual(x):
        q[adr]=x;d.qpos[:]=q;mujoco.mj_forward(model,d);errors=[]
        for b,(p,r) in zip(feet,targets):
            errors.extend(10*(d.xpos[b]-p));errors.extend((rot(r).inv()*rot(d.xquat[b])).as_rotvec())
        return np.r_[errors,.001*(x-current[adr])]
    opts=policy['planning']['ik']
    fit=least_squares(residual,np.clip(current[adr],bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=(bounds[:,0],bounds[:,1]),max_nfev=opts['max_nfev'])
    residual(fit.x)
    errors=[{'position_m':float(np.linalg.norm(d.xpos[b]-p)),'angle_rad':float((rot(r).inv()*rot(d.xquat[b])).magnitude())} for b,(p,r) in zip(feet,targets)]
    points=[e['position'][:2] for e in contact_events if e['force_N']>0 and any(g in bindings['ground_geoms'] for g in e['geoms']) and any('ankle' in b or 'foot' in b for b in e['bodies'])]
    com=d.subtree_com[model.body(bindings['root_body']).id,:2].copy();inside=support_contains(points,com)
    planner=StationaryPlanner(model,penetration_tolerance_m=policy['planning']['penetration_tolerance_m'])
    duration=max(policy['navigation']['response_s'],float(np.max(abs(fit.x-current[adr])))/policy['approach']['joint_speed_rad_s'])
    # Quintic endpoint smoothness increases peak derivative by 1.875 over mean.
    duration=max(duration,1.875*float(np.max(abs(fit.x-current[adr])))/policy['approach']['joint_speed_rad_s'])
    checked=[];path_position_error=0.;path_angle_error=0.
    for w in np.linspace(0,1,max(2,int(np.ceil(duration/policy['control_dt_s']))+1)):
        probe=current.copy();probe[:3]=(1-w)*current[:3]+w*q[:3]
        r=rot(current[3:7]);probe[3:7]=np.roll((r*Rotation.from_rotvec((r.inv()*rot(q[3:7])).as_rotvec()*w)).as_quat(),1)
        probe[adr]=(1-w)*current[adr]+w*q[adr]
        d.qpos[:]=probe;mujoco.mj_forward(model,d)
        path_position_error=max(path_position_error,max(float(np.linalg.norm(d.xpos[b]-p)) for b,(p,r) in zip(feet,targets)))
        path_angle_error=max(path_angle_error,max(float((rot(r).inv()*rot(d.xquat[b])).magnitude()) for b,(p,r) in zip(feet,targets)))
        collisions=planner.collisions(probe)
        if collisions:checked=collisions;break
    accepted=inside and not checked and path_position_error<=opts['position_tolerance'] and path_angle_error<=opts['angle_tolerance'] and duration<=policy['navigation']['correction']['max_duration_s'] and all(e['position_m']<=opts['position_tolerance'] and e['angle_rad']<=opts['angle_tolerance'] for e in errors) and duration+policy['settle']['continuous_s']<=policy['budgets']['settle_s']
    audit={'accepted':accepted,'feet_errors':errors,'target_root':q[:7].tolist(),'target_leg_joints_rad':q[adr].tolist(),
           'leg_qpos_addresses':adr.tolist(),'support_points_xy':points,'target_com_xy':com.tolist(),'com_in_support':inside,
           'reference_start':'measured_configuration','path_foot_position_max_m':path_position_error,'path_foot_angle_max_rad':path_angle_error,
           'correction_duration_limit_s':policy['navigation']['correction']['max_duration_s'],'path_collisions':checked,'duration_s':duration,'nfev':fit.nfev,'physics_steps':0,'reference_only':True,
           'interface':'New leg IK reference through SONIC encoder1; no simulator state reset'}
    return q if accepted else None,audit


def corrected_reference(previous_reference,goal,measured):
    """Unit-gain measured residual update; target metric remains the true goal.

    Caller must check progress, retry budget, support and geometric feasibility
    before executing the updated reference. No preset spatial bias.
    """
    arrays=[np.asarray(x,float) for x in (previous_reference,goal,measured)]
    if any(x.shape!=(2,) or not np.isfinite(x).all() for x in arrays):raise WorkflowError('invalid_input','Parking residual')
    return arrays[0]+arrays[1]-arrays[2]
