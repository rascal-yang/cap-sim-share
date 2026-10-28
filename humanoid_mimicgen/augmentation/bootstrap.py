"""Raw HUMANO contact-task onboarding. No apple workflow dependencies.

Saved-state replay is not physical replay. Source motion tokens remain visible.
The button adapter uses the native downward-force/hold checker, not lift gates.
"""
import copy
import inspect
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .records import create_record
from ..workflow.contracts import file_hash, now
from ..workflow.service import write_new

REPO=Path(__file__).resolve().parents[2]
TASK='G1WholebodyTableButtonPressTeleop'
MODELS=Path('/home/kinetra-4090-01/HUMANO/third_party/gear_sonic_deploy')
SKILL=Path(__file__).parent/'skills/contact_bootstrap/SKILL.md'


def validate_source_clock(times, initial_time, dt, timing):
    t=np.asarray(times,float)
    if timing!='pre_control' or len(t)<2 or not np.isfinite(t).all() or not np.isclose(t[0],initial_time,atol=1e-8,rtol=0) or not np.allclose(np.diff(t),dt,atol=1e-8,rtol=0):
        raise ValueError('Unsupported or misaligned source clock')


def press_segments(times, pressing, hold_steps):
    t=np.asarray(times);mask=np.asarray(pressing,bool)
    if len(t)!=len(mask) or len(t)<2 or np.any(np.diff(t)<=0):raise ValueError('Invalid phase evidence')
    edges=np.diff(np.r_[False,mask,False].astype(int))
    bouts=[(int(a),int(b)-1) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=hold_steps]
    if not bouts:return {'status':'rejected','reason':'No continuous native press hold','segments':[]}
    a,b=bouts[0]
    spans=[('approach',0,a),('press_hold',a,b),('post_contact',b,len(t)-1)]
    return {'status':'ready','segments':[{'name':name,'start_frame':lo,'end_frame':hi,'start_s':float(t[lo]),'end_s':float(t[hi]),'end_inclusive':True} for name,lo,hi in spans], 'boundary_convention':'shared state endpoints; control ticks are not duplicated', 'post_contact_is_not_confirmed_release':True}


def extend_contact_tail(df,hold_steps,reward_dt,policy_dt):
    """Repeat final intent for one native acceptance window, without changing source."""
    import pandas as pd
    n=int(np.ceil(hold_steps*reward_dt/policy_dt))
    out=df.copy(deep=True);out['source_frame']=np.arange(len(out))
    tail=[]
    for i in range(n):
        row=out.iloc[-1].copy();row['sim.time']=float(out['sim.time'].iloc[-1])+(i+1)*policy_dt
        row['sim.qvel']=np.zeros_like(row['sim.qvel']);tail.append(row)
    return pd.concat([out,pd.DataFrame(tail)],ignore_index=True)


def source_paths(dataset,episode):
    root=Path(dataset).resolve();stem=f'episode_{episode:06d}';matches=list(root.glob('data/**/'+stem+'.parquet'))
    if len(matches)!=1:raise ValueError('Expected one source parquet')
    files={'parquet':matches[0],'initial':root/'sonic'/f'{stem}_start.npz','replay':root/'sonic'/f'{stem}_replay.json'}
    for p in files.values():
        if not p.is_file():raise ValueError('Missing source evidence '+str(p))
    return files


def inventory(dataset,episodes):
    import pandas as pd
    rows=[]
    for ep in episodes:
        files=source_paths(dataset,ep);meta=json.loads(files['replay'].read_text());df=pd.read_parquet(files['parquet']);initial=np.load(files['initial'],allow_pickle=False)
        validate_source_clock(df['sim.time'],float(initial['sim_time']),.02,meta['row_timing'])
        if not np.array_equal(initial['qpos'],df['sim.qpos'].iloc[0][:len(initial['qpos'])]):raise ValueError('Initial state differs from pre-control first row')
        rows.append({'episode':ep,'frames':len(df),'duration_s':len(df)*.02,'historical_accepted':meta.get('accepted'),'historical_provider':meta.get('provider'),'hashes':{k:file_hash(v) for k,v in files.items()}})
    return {'task':TASK,'episodes':rows,'physics_steps':0,'historical_success_is_not_new_success':True}


def open_source(dataset,episode,folder,shift=(0,0)):
    import pandas as pd
    from cap_sim.backends.humano import Backend
    from ..privileged import no_physics
    files=source_paths(dataset,episode);meta=json.loads(files['replay'].read_text());df=pd.read_parquet(files['parquet'])
    prepared=Path(folder)/'prepared';prepared.mkdir(exist_ok=True)
    cfg=json.loads(meta['environment_config']);p=prepared/'environment_config.json'
    if not p.exists():write_new(p,cfg)
    task={'name':TASK,'resources':{'prepared':str(prepared),'initial_state':str(files['initial']),'sonic_models':str(MODELS)},'scene':{'translation':[*shift,0]}}
    b=Backend()
    with no_physics():b.describe(task)
    validate_source_clock(df['sim.time'],float(b.initial['sim_time']),.02,meta['row_timing'])
    if not np.all(df['sim.nq']==b.m.nq) or not np.all(df['sim.nv']==b.m.nv):raise ValueError('Source dimension mismatch')
    return b,df,files,task


def saved_replay(dataset,episode,root):
    import mujoco
    from ..privileged import no_physics
    folder=Path(root)/f'saved_replay_{episode:06d}'
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    folder.mkdir(exist_ok=False);start=now();tick=time.monotonic();b,df,files,task=open_source(dataset,episode,folder)
    try:
        rows=[]
        with no_physics():
            for _,row in df.iterrows():
                b.d.qpos[:]=row['sim.qpos'][:b.m.nq];b.d.qvel[:]=row['sim.qvel'][:b.m.nv];b.d.time=float(row['sim.time']);mujoco.mj_forward(b.m,b.d)
                met=b.e.task._button_metrics(mujoco_env=b.e.mujoco)
                rows.append({'time_s':float(b.d.time),'contact':bool(met['has_button_contact']),'pressing':bool(met['has_downward_button_contact'])})
        write_new(folder/'frames.json',rows)
        result={'episode':episode,'frames':len(rows),'physics_steps':0,'phase_candidates':press_segments([r['time_s'] for r in rows],[r['pressing'] for r in rows],b.e.task.hold_steps),'contact_frames':sum(r['contact'] for r in rows),'force_evidence':'recomputed from saved state; not recorded force or new physical success','started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'artifact_path':str(folder)}
        write_new(folder/'result.json',result);return result
    finally:b.close()


def source_poses(b,df):
    import mujoco
    d=mujoco.MjData(b.m);q=np.stack(df['sim.qpos'])[:,:b.m.nq];v=np.stack(df['sim.qvel'])[:,:b.m.nv]
    poses=[]
    for state in q:
        d.qpos[:]=state;mujoco.mj_kinematics(b.m,d)
        poses.append(np.array([np.r_[d.xpos[i],d.xquat[i]] for i in (b.left,b.right,b.torso)]))
    return q,v,np.asarray(poses)


def decode(s,d,token):
    from cap_sim.backends.native_sonic import rot
    s.history.append({'omega':d.qvel[3:6].copy(),'q':(d.qpos[s.qids]-s.default)[s.order],'dq':d.qvel[s.vids][s.order].copy(),'last':s.last.copy(),'gravity':rot(d.qpos[3:7]).inv().apply([0,0,-1])})
    if len(s.history)!=10:raise ValueError('Missing SONIC history')
    obs=np.concatenate([token]+[np.concatenate([h[k] for h in s.history]) for k in ['omega','q','dq','last','gravity']]).astype(np.float32)
    raw=s.decoder.run(None,{s.decoder.get_inputs()[0].name:obs[None]})[0][0];s.last=raw.copy();s.tick+=1
    return s.default+s.scale*raw[s.reverse]


def encode_reference(s,d,q,v,poses,index,shift):
    from cap_sim.backends.native_sonic import rot,quat
    e=np.zeros(1762,np.float32);e[0]=1
    ids=np.minimum(index+np.arange(10)*5,len(q)-1);anchor=rot(q[index,3:7]);actual=rot(d.qpos[3:7])
    e[595:601]=(actual.inv()*anchor).as_matrix()[:,:2].reshape(-1)
    e[661:781]=q[ids][:,s.qids[:12]].reshape(-1);e[781:901]=v[ids][:,s.vids[:12]].reshape(-1)
    ps=[];qs=[]
    for j,offset in enumerate(([.18,-.025,0],[.18,.025,0],[0,0,.35])):
        p=poses[index,j,:3].copy();r=rot(poses[index,j,3:])
        if j<2:p[:2]+=shift
        ps.extend(anchor.inv().apply(p+r.apply(offset)-d.qpos[:3]));qs.extend(quat(anchor.inv()*r))
    e[901:910]=ps;e[910:922]=qs
    token=s.encoder.run(None,{s.encoder.get_inputs()[0].name:e[None]})[0][0]
    return token,e


def check_frozen(folder):
    r=json.loads((folder/'experiment.json').read_text())
    if file_hash(folder/'experiment.json')!=(folder/'experiment.sha256').read_text():raise ValueError('Contract changed')
    if r['record_status']!='frozen_ready' or not r['preflight']['passed']:raise ValueError('Preflight incomplete')
    for p,h in r['preflight']['evidence_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Bound evidence changed '+p)
    return r


def physical_case(dataset,episode,root,mode,shift=(0,0),hold_tail=False):
    """One frozen fresh rollout. Only initialization may write state."""
    import mujoco
    from cap_sim.backends.native_sonic import Sonic
    from humano.mt.base import MotorCommand
    from humano.core.action import ActionCmd
    from .worker import device_lock
    if mode not in ('original_token','cartesian_source_legs'):raise ValueError('Unsupported explicit mode')
    if mode=='original_token' and np.any(shift):raise ValueError('Original replay must retain scene')
    with device_lock() as acquired:
        if not acquired:raise RuntimeError("Physical worker busy")
        return _physical_case(dataset,episode,root,mode,shift,hold_tail,mujoco,Sonic,MotorCommand,ActionCmd)


def _physical_case(dataset,episode,root,mode,shift,hold_tail,mujoco,Sonic,MotorCommand,ActionCmd):
    from ..cli import create_experiment
    folder=create_experiment(Path(root)/'experiments','button_'+mode+('_hold' if hold_tail else ''));start=now();tick=time.monotonic()
    b,df,files,task=open_source(dataset,episode,folder,shift)
    try:
        source_frames=len(df)
        if hold_tail:df=extend_contact_tail(df,b.e.task.hold_steps,b.reward_stride*.005,.02)
        s=Sonic(b.m,b.robot,MODELS,b.initial);q,v,poses=source_poses(b,df)
        b.d.time=float(b.initial['sim_time']);b.d.qacc_warmstart[:]=b.initial['qacc_warmstart']
        # Runtime check loads the exact frozen models; also record sessions actually used below.
        subprocess.run([str(REPO/'scripts/humano_python.sh'),str(REPO/'scripts/check_humano_runtime.py'),'--load-models','--out',str(folder/'runtime.json')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        native_source=Path(inspect.getfile(type(b.e.task)))
        params={k:getattr(b.m,k).tolist() for k in ('body_mass','body_inertia','geom_friction','geom_size','geom_pos','geom_quat','jnt_range','actuator_gainprm','actuator_biasprm')}
        params.update(kp=s.kp.tolist(),kd=s.kd.tolist(),physics_dt_s=float(b.m.opt.timestep),reward_stride=b.reward_stride)
        write_new(folder/'physics.json',params);write_new(folder/'task.json',task);mujoco.mj_saveModel(b.m,str(folder/'scene.mjb'),None)
        np.savez_compressed(folder/'initial.npz',qpos=b.d.qpos.copy(),qvel=b.d.qvel.copy(),ctrl=b.d.ctrl.copy(),qacc_warmstart=b.d.qacc_warmstart.copy())
        write_new(folder/'comparison_diff.json',{'object_delta_xy_m':list(shift),'robot_initial_changed':False,'mode':mode,'hold_tail':hold_tail,'tail_policy_ticks':len(df)-source_frames,'source_provider':'historical gpu; current CPU','cartesian_reference':'saved measured wrist poses and saved leg references, not original encoder commands'})
        sources=list((REPO/'humanoid_mimicgen/augmentation').glob('*.py'))+[REPO/'src/cap_sim/backends/humano.py',REPO/'src/cap_sim/backends/native_sonic.py',REPO/'humanoid_mimicgen/privileged.py',native_source,SKILL]
        bound=list(files.values())+sources+[folder/n for n in ('runtime.json','physics.json','task.json','scene.mjb','initial.npz','comparison_diff.json')]+list(MODELS.glob('policy/release/*.onnx'))+[MODELS/'planner/target_vel/V2/planner_sonic.onnx']
        snapshot=folder/'source_snapshot';snapshot.mkdir()
        import shutil
        for i,p in enumerate(sources):shutil.copy2(p,snapshot/f'{i:03d}_{p.name}')
        acceptance={k:getattr(b.e.task,k) for k in ('touch_distance_m','min_downward_force_n','min_contact_points','hold_steps')}
        stage={'stage':'whole_episode','mode':mode,'lower_body_interface':'original token decoder' if mode=='original_token' else 'recorded leg q/dq in SONIC mode1 encoder','upper_body_interface':'original token decoder' if mode=='original_token' else 'object-translated world Cartesian wrists and source torso via mode1','action_fields_units_frames':'wrist xyz metres, quaternion wxyz; hand joint radians; q_target radians; torque Nm','target_source':{'path':str(files['parquet']),'tail':'repeat final token or hold final Cartesian pose for one native hold window' if hold_tail else 'none'},'interpolation':'50Hz policy, 200Hz PD zero-order hold','ik_used':False,'native_prefix_source_and_hash':{'source':str(files['parquet']),'sha256':file_hash(files['parquet']),'duration_s':len(df)*.02 if mode=='original_token' else 0},'handoff_rule':'none; source initial history restored once'}
        r=json.loads((REPO/'docs/experiment-record.template.json').read_text());runtime=json.loads((folder/'runtime.json').read_text())
        r.update(record_status='frozen_ready',experiment_id=folder.name,created_at=start,purpose={'hypothesis':'API tool can replay button demonstration and test small object-relative Cartesian retargeting','baseline_paths':[str(files['parquet'])],'planned_changes':[mode,list(shift),{'hold_tail':hold_tail,'tail_policy_ticks':len(df)-source_frames}],'invariants':['robot initial state','physics','native acceptance','source hand commands','duration'],'known_confounders':['source GPU vs current CPU','mode1 uses measured pose references, not source encoder inputs']},cases=[{'episode':episode,'object_delta_xy_m':list(shift)}])
        r['runtime'].update(actual_loaded_session_evidence=str(folder/'runtime.json'),model_hashes=runtime['model_sha256'],python_version='3.10',gear_sonic_package_version=runtime['sonic_package_version'],render_device='not rendered')
        r['physics'].update(simulator_version=mujoco.__version__,robot_and_hand='G1 29DoF Dex3',asset_and_initial_state_manifest={str(p):file_hash(p) for p in (folder/'scene.mjb',folder/'initial.npz')},physics_parameters_manifest=str(folder/'physics.json'),physics_dt_s=.005,control_dt_s=.02,pd_parameters=str(folder/'physics.json'),action_permissions='actuator torque only after single initialization; object shift only at initialization',safety_gates={'root_height_min_m':.3,'root_tilt_max_rad':float(np.deg2rad(50)),'robot_table_force_N':50,'nonfinite':'stop','mujoco_warning':'stop'},acceptance_source_and_hash={str(native_source):file_hash(native_source)},acceptance_thresholds=acceptance)
        r['control'].update(stage_interfaces=[stage],actual_mode_log='execution/controller.npz')
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip()
        r['pipeline'].update(entrypoint='humanoid_mimicgen.augmentation.bootstrap.physical_case',ordered_steps=['source binding','forward-only setup','loaded runtime','freeze','SONIC rollout','native acceptance','independent saved-state verification','export'],source_snapshot_and_hashes={str(p):file_hash(p) for p in sources},git_commit_and_dirty_patch={'commit':git,'source_snapshot':str(snapshot)},skill_versions_and_hashes=[{str(SKILL):file_hash(SKILL)}],planner_identity_and_prompt_hash='gpt-6-astra low via bootstrap_agent; API logs in parent',external_model_calls='API orchestrates tools; zero calls inside physics',original_actions_visible=True,feedback_and_reflection_rules='native acceptance and safety only; no online model',allowed_retry_changes=['new source episode under new frozen case'],optimization_enabled=False)
        r['budget'].update(attempts_per_case=1,recovery_attempts=0,simulation_seconds_per_attempt=len(df)*.02,feedback_calls=0)
        r['metrics'].update(target_definition='original source wrist pose + object XY shift for Cartesian cases',sampling_and_alignment='source pre-control row; action then four 5ms steps; paired transitions exported',tracking_error_definition='Euclidean actual wrist minus desired world wrist, sampled before policy tick',endpoint_error_definition='last executed reference tracking error; coverage reported separately',orientation_error_definition='not measured',base_drift_definition='final root XY minus source initial XY',failure_truncation_rule='retain failed prefix, never zero-fill')
        r['preflight'].update(passed=True,comparison_diff_path=str(folder/'comparison_diff.json'),evidence_paths=[str(p) for p in bound],evidence_sha256={str(p):file_hash(p) for p in bound})
        write_new(folder/'experiment.json',r);(folder/'experiment.sha256').write_text(file_hash(folder/'experiment.json'));check_frozen(folder)
        out=folder/'execution';out.mkdir();prep=time.monotonic()-tick;simtick=time.monotonic();states=[];commands=[];nextstates=[];metrows=[];controllers=[];failure=None;ever=False;initial_time=float(b.d.time);errors=[]
        for i,(_,row) in enumerate(df.iterrows()):
            if mode=='original_token':token=np.asarray(row['action.motion_token'],np.float32);enc=np.zeros(1762,np.float32)
            else:token,enc=encode_reference(s,b.d,q,v,poses,i,np.asarray(shift))
            target=decode(s,b.d,token);controllers.append(np.r_[b.d.time,(-1 if mode=='original_token' else 1),token,enc,target]);errors.append([float(np.linalg.norm(b.d.xpos[b.left]-poses[i,0,:3]-np.r_[shift,0])),float(np.linalg.norm(b.d.xpos[b.right]-poses[i,1,:3]-np.r_[shift,0]))])
            for sub in range(4):
                states.append(np.r_[b.d.time,b.d.qpos,b.d.qvel])
                b.robot.apply_action(ActionCmd('wbc_torque',low_cmd=MotorCommand(q_target=target,kp=s.kp,kd=s.kd),use_sensor=False,left_hand_cmd=MotorCommand.hand_command(np.asarray(row['teleop.left_hand_joints'])),right_hand_cmd=MotorCommand.hand_command(np.asarray(row['teleop.right_hand_joints']))))
                commands.append(np.r_[min(i,source_frames-1),target,b.d.ctrl.copy()]);mujoco.mj_step(b.m,b.d);mujoco.mj_forward(b.m,b.d);nextstates.append(np.r_[b.d.time,b.d.qpos,b.d.qvel])
                if sub==3:
                    info=b.e._get_info();b.e.task.compute_reward(info,mujoco_env=b.e.mujoco);passed=bool(b.e.task.check_success(info,mujoco_env=b.e.mujoco));ever|=passed
                    met=b.e.task._button_metrics(mujoco_env=b.e.mujoco);metrows.append({'time_s':float(b.d.time),'pressing':bool(met['has_downward_button_contact']),'native_success':passed,'downward_force_N':float(met['button_downward_force_n']),'contact_count':int(met['button_contact_count'])})
                tilt=np.arccos(np.clip(1-2*(b.d.qpos[4]**2+b.d.qpos[5]**2),-1,1))
                if not np.isfinite(b.d.qpos).all() or not np.isfinite(b.d.qvel).all():failure='nonfinite'
                elif b.d.qpos[2]<.3 or tilt>np.deg2rad(50):failure='fall'
                elif any(w.number for w in b.d.warning):failure='simulation_warning'
                elif b.contacts()['robot_table_N']>50:failure='table_collision'
                if failure:break
            if failure:break
        simwall=time.monotonic()-simtick
        np.savez_compressed(out/'trajectory.npz',state=np.asarray(states),next_state=np.asarray(nextstates),commands=np.asarray(commands),wrist_error_m=np.asarray(errors))
        np.savez_compressed(out/'controller.npz',records=np.asarray(controllers));write_new(out/'metrics.json',metrows)
        # Independently reconstruct contact force and native hold from saved new states.
        b.e.task._button_hold_count=0;verified=False;verifytick=time.monotonic()
        for index in range(3,len(nextstates),4):
            row=nextstates[index];b.d.qpos[:]=row[1:1+b.m.nq];b.d.qvel[:]=row[1+b.m.nq:];b.d.ctrl[:]=commands[index][-b.m.nu:];b.d.time=row[0];mujoco.mj_forward(b.m,b.d)
            b.e.task.compute_reward(b.e._get_info(),mujoco_env=b.e.mujoco);verified|=bool(b.e.task.check_success({},mujoco_env=b.e.mujoco))
        independent_wall=time.monotonic()-verifytick
        if verified!=ever:raise ValueError('Independent native checker disagrees')
        providers={k:getattr(s,k).get_providers() for k in ('encoder','decoder','planner')}
        if any(p!=['CPUExecutionProvider'] for p in providers.values()):raise ValueError('Actual provider mismatch')
        success=bool(ever and not failure)
        result={'experiment_id':folder.name,'experiment_path':str(folder),'episode':episode,'mode':mode,'hold_tail':hold_tail,'source_frames':source_frames,'tail_policy_ticks':len(df)-source_frames,'object_delta_xy_m':list(shift),'success':success,'native_success':ever,'independent_success':verified,'failure':failure,'physics_steps':len(states),'expected_steps':len(df)*4,'coverage':len(states)/(len(df)*4),'started_at':start,'ended_at':now(),'preparation_wall_s':prep,'simulation_wall_s':simwall,'simulation_s':len(states)*.005,'verification_wall_s':independent_wall,'rendering_wall_s':None,'total_wall_s':time.monotonic()-tick,'actual_session_providers':providers,'original_token_steps':len(controllers) if mode=='original_token' else 0,'encoder_modes':[] if mode=='original_token' else [1],'state_initializations':1,'midrun_state_resets':0,'runtime_api_calls':0,'training_ready':False,'retry_count':0,'root_drift_m':float(np.linalg.norm(nextstates[-1][1:3]-states[0][1:3])),'max_downward_force_N':max([r['downward_force_N'] for r in metrows],default=0),'wrist_error_rmse_m':np.sqrt(np.mean(np.asarray(errors)**2,axis=0)).tolist(),'phase_candidates':press_segments([r['time_s'] for r in metrows],[r['pressing'] for r in metrows],b.e.task.hold_steps) if len(metrows)>1 else {'status':'rejected','segments':[]}}
        write_new(out/'execution_audit.json',result);write_new(folder/'result.json',result);check_frozen(folder);return result
    finally:b.close()
