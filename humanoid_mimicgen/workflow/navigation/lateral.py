"""Apple lateral stance search and immutable planning artifacts."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from ..contracts import WorkflowError, positive, canonical, file_hash, now
from ..native import descendants
from ...privileged import no_physics
from ...reachability_navigation import plan_base_path
from .prior import finite, yaw
from .lateral_prior import load_lateral_source
from .lateral_ik import LateralSolver
from .native import NativeEvaluator
from .reference_bundle import save_reference_bundle


def lateral_candidates(prior,current_feet,current_root,delta,search):
    step=search['step_m'];positive(step,'Lateral grid step')
    count=search['max_candidates']
    if type(count) is not int or not 1<=count<=21 or not np.isfinite(delta):raise WorkflowError('invalid_input','Candidate budget/offset')
    tangent=finite(prior['surface']['tangent'],(3,),'Table tangent');root=np.asarray(current_root)
    rows=[dict(id='current',feet=current_feet,root_pose=root[:7].tolist(),xyyaw=[*root[:2],float(yaw(root[:7]))],
               seed_qpos=root.tolist(),requires_navigation=False)]
    # Exact source and target shifts plus a bounded grid, including end margins.
    lo=min(0.,delta)-step;hi=max(0.,delta)+step
    if (hi-lo)/step>10000:raise WorkflowError('planning_budget_exhausted','Unbounded candidate request')
    grid=sorted(set(np.arange(lo,hi+step*.01,step).round(9).tolist()),key=lambda x:(abs(x),x))
    offsets=list(dict.fromkeys([0.,float(delta),*grid]))
    for windex,w in enumerate(prior['support_windows']):
        for offset in offsets:
            shift=tangent*offset;q=np.array(w['qpos'],float);q[:3]+=shift
            feet={k:np.r_[np.asarray(v[:3])+shift,v[3:]].tolist() for k,v in w['feet'].items()}
            if any(all(np.allclose(feet[k],r['feet'][k],atol=1e-8,rtol=0) for k in feet) for r in rows):continue
            rows.append(dict(id=f'support_{windex}_{offset:+.4f}',feet=feet,root_pose=q[:7].tolist(),
                xyyaw=[*q[:2],float(yaw(q[:7]))],seed_qpos=q.tolist(),requires_navigation=True,lateral_shift_m=float(offset)))
    return rows[:count],len(rows)>count


def replay_candidates(prior,current_feet,current_root,delta,search):
    """Lazy replay-aligned stance, then bounded local alternatives on demand."""
    step=search['step_m'];positive(step,'Lateral fallback step')
    count=search['max_candidates']
    if type(count) is not int or not 1<=count<=3 or not np.isfinite(delta):
        raise WorkflowError('invalid_input','Replay-first candidate budget1..3 and finite offset required')
    tangent=finite(prior['surface']['tangent'],(3,),'Table tangent')
    window=prior['support_windows'][0]
    for rank,offset in enumerate([float(delta),float(delta)-step,float(delta)+step][:count]):
        shift=tangent*offset;q=np.array(window['qpos'],float);q[:3]+=shift
        feet={name:np.r_[np.asarray(value[:3])+shift,value[3:]].tolist() for name,value in window['feet'].items()}
        same=all(np.allclose(feet[name],current_feet[name],atol=1e-8,rtol=0) for name in feet)
        yield dict(id=f'replay_{rank}_{offset:+.4f}',feet=feet,root_pose=q[:7].tolist(),
                   xyyaw=[*q[:2],float(yaw(q[:7]))],seed_qpos=q.tolist(),requires_navigation=not same,
                   lateral_shift_m=offset,source_support_window=0,proposal='object_aligned_replay' if rank==0 else 'local_fallback')


def surface_frame(model,qpos,bindings):
    d=mujoco.MjData(model);d.qpos[:]=qpos;mujoco.mj_forward(model,d)
    name=bindings['lateral_surface']['geom'];g=model.geom(name).id
    if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:raise WorkflowError('unsupported_capability','Box table geometry required')
    r=d.geom_xmat[g].reshape(3,3);n=r@finite(bindings['lateral_surface']['normal_local'],(3,),'Normal')
    if abs(n[2])>1e-6 or not np.isclose(np.linalg.norm(n),1) or not np.allclose(r[:,2],[0,0,1],atol=1e-6):
        raise WorkflowError('invalid_lateral_scene','Horizontal table normal required')
    t=np.cross([0,0,1],n)
    return dict(geom=name,normal=n.tolist(),tangent=t.tolist(),center=d.geom_xpos[g].tolist(),
                rotation=r.tolist(),half_extents=model.geom_size[g].tolist())


def shifted_initial(model,initial,prior,bindings,offset):
    q=finite(initial,(model.nq,),'Initial qpos').copy();delta=np.asarray(prior['surface']['tangent'])*float(offset)
    if not np.isfinite(delta).all():raise WorkflowError('invalid_input','Finite lateral offset required')
    obj=model.body(bindings['object']).id;j=int(model.body_jntadr[obj]);a=int(model.jnt_qposadr[j])
    if j<0 or model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:raise WorkflowError('unsupported_capability','Free object required')
    if not np.allclose(q[a:a+7],prior['source_initial_object_pose'],atol=.003,rtol=0):
        raise WorkflowError('context_changed','Initial apple differs from successful source')
    q[a:a+3]+=delta
    surface=prior['surface'];local=np.asarray(surface['rotation']).T@(q[a:a+3]-surface['center'])
    ids=descendants(model,obj);geoms=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in ids]
    # Project actual mesh vertices; a bounding sphere can falsely exclude a supported apple.
    d=mujoco.MjData(model);d.qpos[:]=q;mujoco.mj_forward(model,d)
    if not geoms:raise WorkflowError('invalid_lateral_scene','Missing object geometry')
    table_rotation=np.asarray(surface['rotation']);center=np.asarray(surface['center'])
    for g in geoms:
        kind=model.geom_type[g];r=d.geom_xmat[g].reshape(3,3)
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=model.geom_dataid[g];v=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
            projected=(v@r.T+d.geom_xpos[g]-center)@table_rotation
            extent=np.max(abs(projected[:,:2]),axis=0)
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:
            extent=abs(table_rotation.T@(d.geom_xpos[g]-center))[:2]+model.geom_size[g,0]
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:
            extent=(abs(table_rotation.T@(d.geom_xpos[g]-center))+abs(table_rotation.T@r)@model.geom_size[g])[:2]
        else:raise WorkflowError('unsupported_capability','Object support projection requires mesh, sphere or box')
        if np.any(extent>np.asarray(surface['half_extents'][:2])+1e-9):
            raise WorkflowError('invalid_lateral_scene','Apple footprint outside tabletop')
    return q,delta


def plan_lateral(manifest,initial,options,output_dir,offset_m=0.):
    if not np.isfinite(offset_m):raise WorkflowError('invalid_input','Finite offset required')
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False);started=now();clock=time.monotonic()
    result=dict(schema_version=1,kind='replay_navigation_plan',reference_mode='apple_lateral_v1',status='failed',
                selected=None,candidates=[],physics_steps=0,api_calls=0,dynamic_execution_verified=False,
                planner_access_to_original_actions=True,original_sequences_accessible=True,operation_kind='trajectory_retargeting',
                started_at=started,offset_m=float(offset_m))
    try:
        with no_physics():
            positive(options['search']['planning_timeout_s'],'Total planning timeout')
            deadline=clock+options['search']['planning_timeout_s'];work_deadline=deadline-min(1.,options['search']['planning_timeout_s']*.1)
            (out/'options.json').write_bytes(canonical(options))
            result['evidence_sha256']={str(out/'options.json'):file_hash(out/'options.json')}
            for p in (Path(manifest).resolve(),Path(initial).resolve()):
                if p.is_file():result['evidence_sha256'][str(p)]=file_hash(p)
            trace,prior,fingers,evidence=load_lateral_source(manifest,options,deadline=work_deadline)
            evidence.update(result['evidence_sha256']);result['evidence_sha256']=evidence
            if prior['status']!='completed':raise WorkflowError(prior['status'],'Successful open supported entry required')
            m=mujoco.MjModel.from_binary_path(trace['files']['model']);bindings=prior['bindings']
            with np.load(initial,allow_pickle=False) as z:initial_data={k:z[k].copy() for k in z.files}
            # Lateral variations share the complete source robot and physical initial state.
            source_initial=trace['files'].get('initial')
            if source_initial:
                with np.load(source_initial,allow_pickle=False) as z:
                    if set(z.files)!=set(initial_data) or any(not np.array_equal(z[k],initial_data[k]) for k in z.files):
                        raise WorkflowError('context_changed','Use the matched successful-source initial state')
            prior['surface']=surface_frame(m,initial_data['qpos'],bindings)
            current,delta=shifted_initial(m,initial_data['qpos'],prior,bindings,offset_m)
            initial_data['qpos']=current;np.savez_compressed(out/'initial.npz',**initial_data)
            (out/'prior.json').write_bytes(canonical(prior))
            evidence.update({str(p):file_hash(p) for p in [Path(initial).resolve(),out/'initial.npz',out/'options.json',out/'prior.json']})
            for p in Path(__file__).parent.glob('*.py'):evidence[str(p.resolve())]=file_hash(p)
            result.update(prior=prior,source_manifest=str(Path(manifest).resolve()),source_trajectory_sha256=prior['source_trajectory_sha256'],
                scene_files=dict(model=trace['files']['model'],initial=str(out/'initial.npz')),
                scene_input_sha256=dict(model=file_hash(trace['files']['model']),initial=file_hash(out/'initial.npz')),
                scene=dict(task_frame_xyzyaw=(np.asarray(prior['source_task_frame_xyzyaw'])+np.r_[delta,0.]).tolist()),
                native_options=options['native'],lateral_options=options,evidence_sha256=evidence,
                scene_change=dict(delta_world_m=delta.tolist(),source_initial_apple=prior['source_initial_object_pose'],
                                  original_dataset_delta_world_m=(np.asarray(prior['source_dataset_translation_m'])+delta).tolist() if prior.get('source_dataset_translation_m') is not None else None,
                                  robot_initial_unchanged=True,depth_height_unchanged=True))
            d=mujoco.MjData(m);d.qpos[:]=current;mujoco.mj_forward(m,d)
            feet={n:np.r_[d.xpos[m.body(n).id],d.xquat[m.body(n).id]].tolist() for n in bindings['feet']}
            mode=options['search'].get('mode','grid')
            if mode not in ('grid','replay_first'):raise WorkflowError('invalid_input','Explicit lateral search mode')
            first_feasible=mode=='replay_first'
            if first_feasible:candidates=replay_candidates(prior,feet,current,offset_m,options['search']);truncated=False
            else:candidates,truncated=lateral_candidates(prior,feet,current,offset_m,options['search'])
            result.update(candidate_limit_reached=truncated,selection_policy='first_feasible_replay_prior' if first_feasible else 'ranked_grid',optimality_claimed=False)
            valid=[];budget_exhausted=truncated
            solver=LateralSolver(m,trace['qpos'],trace['time_s'],prior,bindings,options,deadline=work_deadline)
            for candidate in candidates:
                if time.monotonic()>=work_deadline:budget_exhausted=True;break
                # Source template is a planned landing, not an assertion that the gait will achieve it.
                landing=np.asarray(candidate['seed_qpos']).copy()
                robot=descendants(m,m.body(bindings['root_body']).id)
                for j in range(m.njnt):
                    if int(m.jnt_bodyid[j]) not in robot:
                        a=m.jnt_qposadr[j];width=7 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1;landing[a:a+width]=current[a:a+width]
                solver.o=dict(options['ik'],timeout_s=min(options['ik']['timeout_s'],max(.001,work_deadline-time.monotonic())))
                screen=solver.screen(candidate,delta,landing) if first_feasible else None
                check=solver.solve(candidate,delta,landing) if screen is None or screen['feasible'] else screen
                if screen is not None:check['screening']=dict(screen)
                row={**candidate,**{k:v for k,v in check.items() if not isinstance(v,np.ndarray)}}
                if check['feasible']:
                    ev=NativeEvaluator(m,current,trace['qpos'],prior,result['scene']['task_frame_xyzyaw'],bindings,options['native'])
                    start=np.r_[current[:2],yaw(current[:7])];goal=np.asarray(candidate['xyyaw'])
                    def valid_base(base):
                        if time.monotonic()>=work_deadline:raise WorkflowError('planning_budget_exhausted','Global route deadline')
                        q=current.copy();q[:2]=base[:2]
                        from scipy.spatial.transform import Rotation
                        q[3:7]=np.roll((Rotation.from_euler('z',base[2]-start[2])*Rotation.from_quat(np.roll(current[3:7],-1))).as_quat(),1)
                        radius=options['native']['corridor_margin_m']
                        for angle in [None,*np.arange(8)*np.pi/4]:
                            probe=q.copy()
                            if angle is not None:probe[:2]+=radius*np.array([np.cos(angle),np.sin(angle)])
                            if ev.collisions(probe):return False
                        return True
                    try:
                        route,audit=plan_base_path(start,goal,valid_base,**options['native']['path'])
                    except WorkflowError as exc:
                        if exc.code!='planning_budget_exhausted':raise
                        route=None;audit=dict(method='budget_exhausted',detail=exc.detail,checks=ev.checks)
                    row['navigation_path_audit']=audit
                    if route is None:
                        exhausted=audit['method'] in ('astar_exhausted','budget_exhausted')
                        row.update(feasible=False,reason='planning_budget_exhausted' if exhausted else 'navigation_path_failed')
                    else:
                        row.update(navigation_path=route.tolist(),travel_m=float(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1).sum()),
                                   navigation_path_audit=audit,landing_kind='planned_support_not_measured',
                                   operation_margin_m=check['minimum_position_margin_m'])
                        valid.append((row,check))
                if row.get('reason')=='planning_budget_exhausted':budget_exhausted=True
                result['candidates'].append(row)
                if row['feasible'] and (first_feasible or not candidate['requires_navigation']):
                    result['search_stop_reason']='first_feasible' if first_feasible else 'current_stance_feasible'
                    break
            if first_feasible and not valid and len(result['candidates'])>=options['search']['max_candidates']:
                budget_exhausted=True;result['candidate_limit_reached']=True
            if valid:
                selected,check=valid[0] if first_feasible else min(valid,key=lambda pair:(pair[0]['requires_navigation'],pair[0]['travel_m'],-pair[0]['operation_margin_m'],abs(pair[0].get('lateral_shift_m',0)),pair[0]['id']))
                arrays={k:check[k] for k in ('qpos','time_s','source_rows','bridge_qpos','bridge_time_s')}
                arrays['fingers']=fingers[arrays['source_rows']]
                ref=save_reference_bundle(out/'generated',arrays,dict(operation_interval=prior['operation_interval'],source_evidence_sha256=evidence,
                    candidate=selected,object_delta_world_m=delta.tolist(),source_control=prior['source_control'],original_token_steps=0,
                    verification=check['verification'],solver_audit=check['solver_audit'],
                    reference_reused_without_ik=check['reference_reused_without_ik'],reference_modification=check['reference_modification']))
                result.update(status='completed',selected=selected,generated_reference=ref);result['evidence_sha256'].update(ref['evidence_sha256'])
            else:result['status']='planning_budget_exhausted' if budget_exhausted else 'no_feasible_stance'
            result['search_incomplete']=budget_exhausted
            if time.monotonic()>deadline:
                result.update(status='planning_budget_exhausted',reason='planning_budget_exhausted',search_incomplete=True,
                              detail='Total preparation/planning/artifact deadline exceeded; candidate evidence retained')
    except (WorkflowError,ValueError,KeyError,OSError) as exc:
        code=exc.code if isinstance(exc,WorkflowError) else 'invalid_input'
        result.update(status='planning_budget_exhausted' if code=='planning_budget_exhausted' else 'failed',reason=code,detail=str(exc))
        if code=='planning_budget_exhausted':result['search_incomplete']=True
    result.update(ended_at=now(),planning_wall_s=time.monotonic()-clock)
    (out/'plan.json').write_bytes(canonical(result))
    return result
