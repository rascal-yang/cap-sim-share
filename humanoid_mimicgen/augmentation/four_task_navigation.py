"""Task-aware precision navigation of translated successful whole-body replays."""
import copy
import json
from pathlib import Path
import numpy as np
from .agent import save
from ..workflow.contracts import file_hash


def task_strategy(base, bindings, task):
    strategy=copy.deepcopy(base)
    strategy['bindings'].update(object=bindings['object'], support_geoms=bindings['support_geoms'] or bindings['ground_geoms'],
        ground_geoms=bindings['ground_geoms'], hands=['left','right'] if task=='plate' else [] if task=='chair' else ['right'])
    return strategy


def retained(task, observations):
    if task in ('chair','button'):return None
    return bool(observations and observations[-1]['contacts']['hand_object_N']>.1
        and observations[-1]['contacts']['support_N']<.1 and observations[-1]['clearance_m']>.03)


def translate_reference(qpos, delta, object_address=None):
    result=np.asarray(qpos).copy();result[:,:3]+=delta
    if object_address is not None:result[:,object_address:object_address+3]+=delta
    return result


def feedback_metadata(sides, position_gain, orientation_gain):
    return dict(interface='world wrist pose feedback before SONIC encoder on active_sides only',active_sides=list(sides),position_gain=position_gain if sides else 0.,orientation_gain=orientation_gain if sides else 0.,correction_clipping=False,decoder_overrides=False)


def position_label(offset):
    return 'center' if offset==0 else 'minus10' if offset<0 else 'plus10'


def prepare_plan(root, task, offset):
    import mujoco
    from ..privileged import no_physics
    from ..workflow.navigation.reference_bundle import save_reference_bundle
    from .generalization_scene import validate_scene
    repo=Path(__file__).resolve().parents[2];root=Path(root);folder=root/'plans'/(task+'_'+position_label(offset));folder.mkdir(parents=True)
    manifest=repo/'humanoid_mimicgen/workflow/tests/fixtures/navigation'/('apple_lateral.json' if task=='apple' else task+'.json')
    source=json.loads(manifest.read_text());baseline=Path(source['files']['model']).parent;result=json.loads(Path(source['files']['result']).read_text())
    if not result.get(source['success_field']):
        if task!='button' or not json.loads(Path(source['files']['acceptance']).read_text())['revised_success']:raise ValueError('Source success not verified')
    delta=np.array([0.,offset,0.]);scene=folder/'scene';scene.mkdir()
    with no_physics():
        m=mujoco.MjModel.from_binary_path(str(baseline/'scene.mjb'));obj=m.body(source['bindings']['object']).id;jid=int(m.body_jntadr[obj]);free=jid>=0 and m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE;address=int(m.jnt_qposadr[jid]) if free else None
        with np.load(baseline/'initial.npz') as z:initial={k:z[k].copy() for k in z.files}
        if free:initial['qpos'][address:address+3]+=delta
        else:m.body_pos[obj]+=delta
        mujoco.mj_saveModel(m,str(scene/'scene.mjb'),None);np.savez_compressed(scene/'initial.npz',**initial)
        import shutil
        shutil.copy2(baseline/'physics.json',scene/'physics.json');native_task=json.loads((baseline/'task.json').read_text());native_task['scene']['translation']=(np.asarray(native_task['scene']['translation'])+delta).tolist();save(scene/'task.json',native_task)
        comparison=validate_scene(scene,baseline,obj);save(folder/'scene_comparison.json',comparison)
        first,last=source['entry_index'],source['end_index'];rows=np.arange(first,last+1)
        with np.load(source['files']['trajectory']) as z:q=translate_reference(z['state'][first:last+1,1:1+m.nq],delta,address)
        with np.load(baseline/'execution/hand_commands.npz') as h:fingers=np.concatenate([h['left'][rows//4],h['right'][rows//4]],axis=1)
        arrays=dict(qpos=q,time_s=np.arange(len(q))*.005,source_rows=rows,fingers=fingers,bridge_qpos=np.stack([initial['qpos'],q[0]]),bridge_time_s=np.array([0.,4.]))
        evidence={str(p):file_hash(p) for p in [manifest,Path(source['files']['trajectory']),Path(source['files']['result']),baseline/'execution/hand_commands.npz',baseline/'initial.npz',baseline/'scene.mjb',baseline/'task.json',baseline/'experiment.json',Path(source['files']['contact_contract']),Path(__file__)]}
        if task=='button':evidence[source['files']['acceptance']]=file_hash(source['files']['acceptance'])
        reference=save_reference_bundle(folder/'reference',arrays,dict(operation_interval=[first,last],source_evidence_sha256=evidence,translation_world_m=delta.tolist(),geometric_collision_certification=False,method='rigid world translation of full source operation, source fingers unchanged'))
        from scipy.spatial.transform import Rotation
        yaw=lambda pose:float(Rotation.from_quat(np.roll(pose[3:7],-1)).as_euler('xyz')[2])
        route=[[*initial['qpos'][:2],yaw(initial['qpos'])],[*q[0,:2],yaw(q[0])]]
        # Reuse source-proven G1 policy configuration, with explicit task/object/support bindings.
        base=json.loads((repo.parent/'experiments/2026-10/20/20261020_1710_workflow_parking_final/baseline/request.json').read_text())['strategy']
        task_info=dict(name=task,native_name=native_task['name'],strategy=task_strategy(base,source['bindings'],task),scene_comparison=str(folder/'scene_comparison.json'),acceptance_profile=source['success_criterion'])
        evidence.update({str(p):file_hash(p) for p in scene.iterdir()});evidence[str(folder/'scene_comparison.json')]=file_hash(folder/'scene_comparison.json');evidence.update(reference['evidence_sha256'])
        plan=dict(status='completed',offset_m=offset,source_manifest=str(manifest),generated_reference=reference,prior={'bindings':source['bindings']},scene_files={k:str(scene/v) for k,v in [('model','scene.mjb'),('initial','initial.npz'),('task','task.json'),('physics','physics.json')]},execution_task=task_info,selected=dict(id=folder.name,navigation_path=route,reference_structurally_valid=True,geometric_collision_certification=False),evidence_sha256=evidence)
        save(folder/'plan.json',plan)
    return folder/'plan.json'


def button_full_execution(result, observations, selected_bodies, target_bodies, limits, robot_bodies):
    selected_peak=0.;height=float('inf');tilt=0.;obstacle=0.
    for ob in observations:
        q=ob['root_pose'];height=min(height,q[2]);tilt=max(tilt,float(np.arccos(np.clip(1-2*(q[4]**2+q[5]**2),-1,1))))
        obstacle=max(obstacle,sum(e['force_N'] for e in ob['contacts']['events'] if 'table' in e['bodies'] and any(b in robot_bodies for b in e['bodies'])))
        selected_peak=max(selected_peak,sum(e['force_N'] for e in ob['contacts']['events'] if any(b in selected_bodies for b in e['bodies']) and any(b in target_bodies for b in e['bodies'])))
    passed=bool(result['completed'] and not result['failure'] and result['audit']['simulator_warning_count']==0 and height>=limits['root_height_min_m'] and tilt<=limits['root_tilt_max_rad'] and obstacle<=limits['robot_table_force_N'] and selected_peak<=limits['selected_contact_force_N'])
    return dict(passed=passed,selected_contact_peak_N=selected_peak,min_root_height_m=height,max_root_tilt_rad=tilt,robot_table_peak_N=obstacle,limits=limits,application='posthoc acceptance envelope only; does not abort physics')


def button_approval(case):
    """Reviewed palm mesh rule, unchanged 20N/1mm/50-tick thresholds; saved states only."""
    import mujoco
    from .button_acceptance import accepts_contact,hold_result,PROFILE
    from ..privileged import no_physics
    case=Path(case);plan=json.loads((case/'offline_plan.json').read_text());src=json.loads(Path(plan['source_manifest']).read_text());baseline=json.loads(Path(src['files']['result']).read_text())
    expected=dict(min_downward_force_n=20.,touch_distance_m=.001,min_contact_points=1,hold_steps=50)
    if any(baseline['native_acceptance']['parameters'].get(k)!=v for k,v in expected.items()):raise ValueError('Button native thresholds differ')
    result=json.loads((case/'result.json').read_text());rows=[[],[]]
    baseline_dir=Path(src['files']['model']).parent;limits=json.loads((baseline_dir/'experiment.json').read_text())['physics']['safety_gates'];contact=json.loads(Path(src['files']['contact_contract']).read_text())['contact_control']
    with no_physics():
        m=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));d=mujoco.MjData(m);trace=np.load(case/'execution/trajectory.npz')
        buttons={g for g in range(m.ngeom) if any(k in (m.geom(g).name or '') for k in ('E_button_01_13','E_button_02_6'))}
        meshes={g:(m.mesh(m.geom_dataid[g]).name or '') if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH else '' for g in range(m.ngeom)}
        extra=[g for g in range(m.ngeom) if accepts_contact(m.body(int(m.geom_bodyid[g])).name or '',meshes[g],True) and not accepts_contact(m.body(int(m.geom_bodyid[g])).name or '',meshes[g],False) and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if not buttons or len(extra)!=1:raise ValueError('Exact reviewed palm mesh missing')
        for i in range(3,len(trace['next_state']),4):
            state=trace['next_state'][i];d.time=state[0];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];d.ctrl[:]=trace['commands'][i,-m.nu:];mujoco.mj_forward(m,d)
            values=[dict(force_N=0.,distance_m=1e9,contacts=0) for _ in range(2)]
            for ci,c in enumerate(d.contact):
                a,b=int(c.geom1),int(c.geom2);other=b if a in buttons else a if b in buttons else None
                if other is None:continue
                body=m.body(int(m.geom_bodyid[other])).name or '';f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);world=np.asarray(c.frame).reshape(3,3).T@f[:3];down=max(0.,float(world[2] if a in buttons else -world[2]))
                for j,revised in enumerate((False,True)):
                    if accepts_contact(body,meshes[other],revised):values[j]['force_N']+=down;values[j]['contacts']+=1;values[j]['distance_m']=min(values[j]['distance_m'],float(c.dist))
            for j in (0,1):rows[j].append(values[j])
    from ..workflow.native import descendants
    target_names={m.body(b).name for b in descendants(m,m.body(src['bindings']['object']).id)}
    envelope=button_full_execution(result,json.loads((case/'execution/observations.json').read_text()),contact['contact_bodies'],target_names,limits,{m.body(b).name for b in descendants(m,m.body(src['bindings']['root_body']).id)})
    native,approved=(hold_result(x) for x in rows)
    if native['success']!=result['native_success']:raise ValueError('Independent button native reconstruction differs')
    # The approved historical profile requires complete nonfailed execution. Collision quality remains separate.
    return dict(profile=PROFILE,native_success=native['success'],approved_success=bool(envelope['passed'] and approved['success']),palm_contact_hold_pass=approved['success'],source_execution_envelope=envelope,native_hold_s=native['longest_s'],approved_hold_s=approved['longest_s'],thresholds=expected,physics_steps=0,exact_reviewed_palm_geom=extra[0],contact_quality_reported_separately=True)


def run_request(path):
    from . import direct_grasp,replay_navigation,lateral_delivery
    from .navigation_api import navigation_profile,summarize_contacts
    import time
    path=Path(path);request=json.loads(path.read_text());plan_path=Path(request['plan']);plan=json.loads(plan_path.read_text());task=plan['execution_task']['name'];gain=request['decision']['feedback_gain'];nav=navigation_profile(gain);nav.pop('retry')
    proof=request['provenance'];proof['evidence_sha256'][str(path)]=file_hash(path)
    start=time.monotonic();case,adapter,cfg=direct_grasp.prepare(Path(request['root']),plan_path,task+'_precision_'+position_label(plan['offset_m']),gain,0.,proof,keep_adapter=True,navigation=nav)
    result=replay_navigation.run(case,(adapter,cfg));cycle=time.monotonic()-start;tick=time.monotonic();audit=lateral_delivery.audit(case,scene_only=True)
    if not audit['recorded_native_matches']:raise ValueError('Independent native result mismatch')
    approval=button_approval(case) if task=='button' else None
    if approval:save(case/'button_approval.json',approval)
    obs=json.loads((case/'execution/observations.json').read_text());contacts=summarize_contacts(obs);save(case/'contact_diagnosis.json',contacts)
    summary=dict(task=task,offset_m=plan['offset_m'],case=str(case),native_success=result['native_success'],approved_success=approval['approved_success'] if approval else None,success=approval['approved_success'] if approval else result['success'],retained_grasp=result['retained_grasp'],completed=result['completed'],failure=result['failure'],navigation_arrived=result['navigation_arrived'],max_obstacle_N=result['max_obstacle_N'],contacts=contacts,physics_steps=result['physics_steps'],actual_modes=result['audit']['actual_encoder_modes'],original_token_steps=result['audit']['original_token_steps'],cycle_wall_s=cycle,audit_wall_s=time.monotonic()-tick,simulation_compute_wall_s=result['audit']['simulation_compute_wall_s'],sonic_record=result['audit']['sonic_action_record'],native_audit=audit,approval=approval)
    save(path.with_suffix('.result.json'),summary);return summary


if __name__=='__main__':
    import sys
    if sys.argv[1]=='plan':
        root=Path(sys.argv[2]);plans=[]
        for task in ('chair','apple','button','plate'):
            for offset in (-.1,.1):plans.append(str(prepare_plan(root,task,offset)))
        save(root/'plans.json',plans)
    elif sys.argv[1]=='run':run_request(sys.argv[2])
