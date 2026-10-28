"""Matched XY contact-retargeting range evaluation; preserves frozen executor.

The same original 5s tokens + 3s mode0 prefix is used in every case. Only apple
initial XY and right wrist target XY change. No navigation or online IK added.
"""
import argparse,json,shutil,subprocess,time,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from .action_replay import sha,write
from .privileged import no_physics

BASE=Path('/home/kinetra-4090-01/code/zijian')
REPO=BASE/'cap-sim'
BASELINE=BASE/'experiments/20261017_1725_apple_contact_range_y_plus8'
CASES={'y_plus8':(0,.08),'y_plus10':(0,.10),'y_plus12':(0,.12)}

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def shifted_targets(source,delta):
    delta=np.asarray(delta,float)
    if delta.shape!=(3,) or not np.isfinite(delta).all() or delta[2]!=0 or np.linalg.norm(delta)>.12000001:raise ValueError('XY scan only, <=12cm radius')
    result={k:np.array(v,copy=True) for k,v in source.items()};result['right_pose'][:,:3]+=delta
    return result

def physical_classification(contact_started,completed,hold_fraction,root_drift,table_duration,peak_table):
    if not contact_started:return 'no_contact_execution'
    if not completed or hold_fraction<.95 or root_drift>.03:return 'contact_execution_failed'
    if table_duration>0 or peak_table>.1:return 'grasp_lift_hold_with_table_contact'
    return 'clean_grasp_lift_hold'

def prepare(label):
    import mujoco
    from .transfer_feedback_direct import check_contract
    from .transfer_feedback import check_contract as check_baseline
    from .action_replay import initialize_source,measurement
    check_baseline(BASELINE/'experiment.json');c=json.loads((BASE/'experiments/20261016_1853_apple_transfer_feedback_origin/experiment.json').read_text())
    if label not in CASES:raise ValueError('Unknown condition')
    delta=np.r_[CASES[label],0.];created=now();stem=datetime.fromisoformat(created).strftime('%Y%m%d_%H%M')+'_apple_contact_direct_'+label;r=BASE/'experiments'/stem;seq=1
    while r.exists():seq+=1;r=BASE/'experiments'/(stem+'_'+str(seq))
    r.mkdir();write(r/'planning_started.json',{'started_at':created});tick=time.perf_counter()
    initial={k:v.copy() for k,v in np.load(c['inputs']['initial']).items()};original=initial['qpos'].copy();initial['qpos'][50:53]+=delta;np.savez_compressed(r/'initial.npz',**initial)
    raw={k:v.copy() for k,v in np.load(c['inputs']['targets']).items()};targets=shifted_targets(raw,delta);np.savez_compressed(r/'targets.npz',**targets)
    assert np.array_equal(initial['qpos'][:50],original[:50]) and np.array_equal(initial['qpos'][53:],original[53:])
    for k in raw:
        if k!='right_pose':assert np.array_equal(raw[k],targets[k]),k
    assert np.array_equal(raw['right_pose'][:,3:],targets['right_pose'][:,3:])
    task=json.loads(Path(c['inputs']['task']).read_text());task['resources']['initial_state']=str(r/'initial.npz');write(r/'task.json',task)
    manifest=json.loads(Path(c['inputs']['transfer_manifest']).read_text());manifest.update(experiment_id=r.name,translation_m=delta.tolist());anchor=np.asarray(manifest['source_object_anchor_pose']);anchor[:3]+=delta;manifest['target_object_anchor_pose']=anchor.tolist();write(r/'transfer_manifest.json',manifest)
    m=mujoco.MjModel.from_binary_path(c['inputs']['model']);d=mujoco.MjData(m);source=np.load(c['inputs']['reference_states'])['samples'];hits=[]
    with no_physics():
        for key,val in json.loads(Path(c['physics']['physics_parameters_manifest']).read_text())['arrays'].items():assert np.array_equal(getattr(m,key),np.asarray(val)),key
        initialize_source(m,d,initial);init_ob=measurement(m,d)
        for i in range(1600):
            d.qpos[:]=source[i,1:58];d.qpos[50:53]+=delta;d.qvel[:]=source[i,58:113];d.ctrl[:]=source[i,-43:];mujoco.mj_forward(m,d)
            for contact in d.contact:
                names=[m.body(int(m.geom_bodyid[g])).name for g in contact.geom]
                if 'apple_001' in names and any('hand' in n for n in names) and contact.dist<0:hits.append({'frame':i,'bodies':names,'penetration_m':float(-contact.dist)})
        handoff=measurement(m,d)
    pre={'physics_steps':0,'initial_only_apple_XY_changed':True,'all_other_target_channels_unchanged':True,'initial':init_ob,'prefix_frames_geometrically_checked':1600,'prefix_hand_object_penetrations':hits,'predicted_handoff':handoff,'geometry_wall_s':time.perf_counter()-tick,'contact_stage_feasibility':'unknown until physical execution','scope':'including same fixed original approach prefix; blocked prefix is not intrinsic contact infeasibility'}
    write(r/'preflight_evidence.json',pre)
    if hits:
        write(r/'preflight_failure.json',{'experiment_id':r.name,'time':now(),'reason':'translated apple intersects fixed prefix hand path','physics_steps':0,'translation_m':delta.tolist()});return r,False
    subprocess.run([str(REPO/'scripts/humano_python.sh'),str(REPO/'scripts/check_humano_runtime.py'),'--load-models','--out',str(r/'runtime.json')],check=True,stdout=open(r/'runtime.log','w'),stderr=subprocess.STDOUT)
    c.update(experiment_id=r.name,created_at=created)
    c['purpose']={'hypothesis':'Fixed-stance SONIC contact retargeting has a bounded direction-dependent XY physical and visually plausible domain','baseline_paths':[str(BASELINE),str(BASE/'experiments/20261016_1853_apple_transfer_feedback_y_plus_2cm')],'planned_changes':['Only apple initial XY and all right wrist XYZ targets translated by '+str(delta.tolist()),'Extend standing timeout from2s to5s; all pose, stability, collision and contact thresholds unchanged'],'invariants':['Exact originalrobotinitial and physics','5s250originaltokens +3smode0 sourceprefix','same mode1 fixedlegreference and1sbridge/1..5sstandinggate','same orientations/fingers/leftwrist/torso/root/reference timing','no online IK or planner; no resets afterinitial'],'known_confounders':['n=1 per offset, sampled domain not continuous guaranteed radius','1s bridge and fixedprefix constrain operational range','native9cm and strict1shold diagnostics do not define our relaxed task-completion label','not formal tencondition controller comparison','source trajectory retargeting; source actions visible','white diagnostic rendering; visual sampling not full temporal training acceptance']}
    c['pipeline']['stage_gates']=[x.replace('1..2s','1..5s') if isinstance(x,str) else x for x in c['pipeline'].get('stage_gates',[])]
    c['cases']=[{'id':label,'apple_translation_m':delta.tolist(),'scene_model_sha256':sha(c['inputs']['model']),'initial_state_sha256':sha(r/'initial.npz')}]
    for key,name in [('initial','initial.npz'),('targets','targets.npz'),('task','task.json'),('transfer_manifest','transfer_manifest.json')]:c['inputs'][key]=str(r/name)
    c['runtime']['actual_loaded_session_evidence']=str(r/'runtime.json');c['control']['object_translation_m']=delta.tolist()
    c['physics']['acceptance_thresholds']['standing_gate'].update(enabled=False,reason='User requested direct grasp; no pose or continuous standing admission gate',minimum_dwell_s=0,timeout_s=0)
    c['control']['stage_interfaces']=[x for x in c['control']['stage_interfaces'] if x['stage']!='standing']
    for interface in c['control']['stage_interfaces']:
        if interface['stage']!='prefix_0_to_8s':interface['handoff_rule']='after1sapproach at9s directly enter grasp without standing/pose gate; collision/fall/contact-loss/object-anchor stops unchanged'
    c['purpose']['planned_changes']=['At Y+8cm same initial and targets; direct grasp at9s after1sapproach, bypass standing/pose admission only; larger offsets only if previous completes']
    c['purpose']['invariants']=['same robotinitial physics model sourceprefix wrists/fingers and mode1 legs','unchanged collision,fall,contact-loss and pregrasp-object-motion stops','no onlineIK or planner or reset']
    c['pipeline']['stage_gates']=['prefix arrival','1sapproach then direct grasp without standing gate','unchanged collision/fall/contact-loss/object-anchor stops','native and independent acceptance']
    c['physics']['acceptance_source_and_hash']['independent_validator']=str(REPO/'humanoid_mimicgen/transfer_feedback_direct.py')
    c['budget'].update(simulation_seconds_per_attempt=19.4,feedback_calls='per5ms at most3880steps')
    c['physics']['range_evaluation']={'task_candidate':'completed source sequence; >=95percent hold frames withhandcontact>.1N,support<.1N,clearance>=.01m;postprefixrootdrift<=.03m','clean_contact':'robot-table force never exceeds.1N afterprefix; warnings kept separate','strict_native':'existing verify retained unchanged','visual':'ego/external samples around approach/grasp/lift/hold/final; contact/image consistency; unresolved or unreviewed temporal quality means training_ready false','physical_claim':'a sampled feasible offset; never extrapolate an entire disk'}
    c['pipeline'].update(entrypoint='python -m humanoid_mimicgen.transfer_feedback_direct run --contract '+str(r/'experiment.json'),planner_identity_and_prompt_hash={'identity':'deterministic XY sweep; no online planner or external model','external_prompt':'not applicable'},feedback_and_reflection_rules='At most3 runs: Y+8,+10,+12cm. Advance only after previous relaxed completion; stop on first failure. At+8cm bypass standing admission and enter grasp at9s; safety stops unchanged. Record reflection before next.')
    diff={'experiment_id':r.name,'baseline':str(BASELINE),'physics_input_difference_vs_Yplus8':'none at+8cm; appleXY only forlarger offsets','command_reference_difference_vs_Yplus8':'none at+8cm; rightwristXYZ translation only forlarger offsets','delta_m':delta.tolist(),'same_prefix':True,'standing_pose_gate_bypassed':True,'contact_start_s':9,'safety_stops_unchanged':True,'retained_original_prefix':'250 source tokens 0..5s,source mode0 5..8s','evaluation_added':'task_candidate and table_contact warning, strict/native unchanged'};write(r/'comparison_diff.json',diff)
    if label=='y_plus8':
        old=json.loads((BASELINE/'experiment.json').read_text())
        equality={}
        for key in ['initial','targets']:
            with np.load(c['inputs'][key]) as new,np.load(old['inputs'][key]) as prev:
                equality[key]=set(new.files)==set(prev.files) and all(np.array_equal(new[k],prev[k]) for k in new.files)
        assert all(equality.values())
        write(r/'paired_input_equality.json',equality)
    hashes=dict(c['preflight']['evidence_sha256']);snap=r/'source_snapshot';snap.mkdir()
    for p in [Path(__file__).resolve(),REPO/'humanoid_mimicgen/transfer_feedback_direct.py',REPO/'humanoid_mimicgen/render_standing.py',REPO/'humanoid_mimicgen/standing_handoff.py']:
        hashes[str(p)]=sha(p);shutil.copy2(p,snap/p.name)
    shutil.copy2('/tmp/direct_tests.log',r/'tests.log')
    for name in ['initial.npz','targets.npz','task.json','transfer_manifest.json','runtime.json','preflight_evidence.json','comparison_diff.json','tests.log']:
        p=r/name;hashes[str(p)]=sha(p)
    write(r/'assets_manifest.json',{'model':c['inputs']['model'],'sha256':sha(c['inputs']['model']),'initial_sha256':sha(r/'initial.npz'),'physical_model_unchanged':True});hashes[str(r/'assets_manifest.json')]=sha(r/'assets_manifest.json');c['physics']['asset_and_initial_state_manifest']=str(r/'assets_manifest.json')
    c['preflight'].update(evidence_paths=list(hashes),evidence_sha256=hashes,comparison_diff_path=str(r/'comparison_diff.json'),passed=True);c['pipeline']['source_snapshot_and_hashes']=hashes;c['pipeline']['git_commit_and_dirty_patch']['dirty_sources']=str(snap)
    for p,h in hashes.items():assert sha(p)==h,p
    write(r/'experiment.json',c);(r/'experiment.sha256').write_text(sha(r/'experiment.json')+'\n');write(r/'freeze.json',{'frozen_at':now(),'sha256':sha(r/'experiment.json')});check_contract(r/'experiment.json')
    return r,True

def evaluate(r):
    r=Path(r);a=json.loads((r/'execution/execution_audit.json').read_text());v=json.loads((r/'execution/validation.json').read_text());obs=json.loads((r/'execution/observations.json').read_text());hold=[o for o in obs if o['stage']=='hold'];post=[o for o in obs if o['time_s']>8];fraction=sum(o['clearance_m']>=.01 and o['contacts']['hand_object_N']>.1 and o['contacts']['support_N']<.1 for o in hold)/len(hold) if hold else 0.
    category=physical_classification(a['contact_started_at_s'] is not None,a['completed'],fraction,v.get('max_root_xy_drift_m',1e9),v.get('table_contact_gt_0p1_s',0),v['max_table_N'])
    tr=json.loads((r/'execution/tracking.json').read_text());last=tr[-1] if tr else None
    row={'experiment_id':r.name,'translation_m':a['object_translation_m'],'category':category,'completed':a['completed'],'failure':a['failure'],'exception':a['exception'],'contact_started_s':a['contact_started_at_s'],'hold_valid_fraction':fraction,'hold_min_clearance_m':v['hold_min_clearance_m'],'max_clearance_m':v['max_clearance_m'],'stable_hold_s_strict_diagnostic':v['stable_hold_s'],'native_success':v['native']['success'],'max_table_N':v['max_table_N'],'table_contact_s':v.get('table_contact_gt_0p1_s',0),'max_root_xy_drift_m':v.get('max_root_xy_drift_m'),'feet':v.get('feet'),'right_wrist_tracking':v.get('wrist_tracking',{}).get('right'),'final_wrist_error_m':None if last is None else last['right']['position_error_m'],'phase_coverage':v['phase_coverage'],'sim_seconds':a['sim_seconds'],'actual_contact_modes':a['actual_contact_modes'],'training_ready':False}
    write(r/'range_result.json',row)
    reflection={'id':r.name,'time':now(),'prompt_version':'human_prior_v7; no external model calls','stage':'physics','configuration':'G1Dex3/apple, unchanged source scene','source_reference':str(BASELINE),'code_snapshot':str(r/'source_snapshot'),'human_strategy':'same contact trajectory, shifted apple/rightwrist only','hypothesis':'direct contact execution can succeed despite entrance pose residual above old gate','previous_attempt':str(BASELINE),'changed_parameters':{'translation_m':a['object_translation_m'],'standing_gate_bypassed':True},'predicted_observations':'small offsets may transfer; boundary depends on direction and approach','observed_metrics':row,'stop_reason':a['failure'],'evidence_paths':[str(r/'execution/validation.json'),str(r/'range_result.json')],'result':category,'reflection':'Evaluate sampled offset only. Separate no-contact gate failures, actual contact failures, table-contact warnings and visual review. Do not infer continuous radius.','experience_status':'observed','applicability':'fixed sourceprefix, 1sbridge, fixedmode1legs, unchangedorientation/handtiming; not arbitrarynavigation endpoint'}
    with (r/'reflection.json').open('x') as f:json.dump(reflection,f,indent=2)
    with (BASE/'policy_memory/grasp_learning/attempts.jsonl').open('a') as f:f.write(json.dumps(reflection)+'\n')
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','evaluate']);p.add_argument('value');args=p.parse_args()
    if args.command=='prepare':
        r,ok=prepare(args.value)
        with Path('/tmp/direct_paths').open('a') as f:f.write(str(r)+'\n')
        print(json.dumps({'path':str(r),'ready':ok}),flush=True)
    else:print(json.dumps(evaluate(args.value),indent=2),flush=True)
