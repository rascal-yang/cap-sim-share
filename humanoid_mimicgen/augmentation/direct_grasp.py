"""Bounded direct-stance SONIC mode1 physics probe; no navigation or mid-run reset."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,time,copy,shutil,subprocess,platform,inspect
from pathlib import Path
from collections import deque
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from humanoid_mimicgen.workflow.contracts import now,file_hash,WorkflowError
from humanoid_mimicgen.workflow.native import NativeAdapter,validate_model_identity,validate_physics
from humanoid_mimicgen.workflow.navigation.archive import verify_evidence
from humanoid_mimicgen.workflow.navigation.reference_bundle import load_reference_bundle
from humanoid_mimicgen.workflow.navigation.whole_body import BodyReference,set_body_reference
from humanoid_mimicgen.privileged import no_physics
REPO=Path(os.environ.get('CAP_SIM_REPOSITORY',Path(__file__).resolve().parents[2]))
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def rot(q):return Rotation.from_quat(np.roll(q,-1))
def initialize(case):
 cfg=json.loads((case/'config.json').read_text());task=json.loads((case/'task.json').read_text())
 with np.load(case/'initial.npz') as z:ini={k:z[k].copy() for k in z.files}
 with no_physics():
  a=NativeAdapter(task,cfg['strategy'],ini)
  validate_model_identity(mujoco.MjModel.from_binary_path(str(case/'scene.mjb')),a.m)
  ph=json.loads((case/'physics.json').read_text())
  for key,value in ph.items():
   if key in ('kp','kd'):assert np.array_equal(getattr(a.sonic,key),value),key
   elif key=='physics_dt_s':assert a.dt==value
   elif key!='reward_stride':assert np.array_equal(getattr(a.m,key),value),key
  if not cfg.get('navigation'):
   # Explicit synthetic stationary history at this new initial state, not a source checkpoint.
   s=a.sonic;last=np.empty(29);last[s.reverse]=(a.d.qpos[s.qids]-s.default)/s.scale;s.last=last.copy();s.history.clear()
   for _ in range(10):s.history.append(dict(omega=a.d.qvel[3:6].copy(),q=(a.d.qpos[s.qids]-s.default)[s.order],dq=a.d.qvel[s.vids][s.order].copy(),last=last.copy(),gravity=rot(a.d.qpos[3:7]).inv().apply([0,0,-1])))
  assert all(p==['CPUExecutionProvider'] for p in a.providers.values())
  # Native acceptance anchor uses this explicitly changed initial scene.
  for k in vars(a.b.e.task):
   if k.startswith('_init_'):setattr(a.b.e.task,k,None)
   if k.endswith('_hold_count'):setattr(a.b.e.task,k,0)
  a.native_success()
 return a,cfg

def prepare(root,plan_path,label,feedback=0.,zoffset=0.,provenance=None,keep_adapter=False,navigation=None):
 privileged=bool(provenance and provenance.get('perception_mode')=='privileged_only')
 begin=time.monotonic();stamp=now()[:16].replace('-','').replace(':','').replace('T','_');case=root/(stamp+'_'+label);i=1
 while case.exists():case=root/(stamp+'_'+label+'_'+str(i));i+=1
 case.mkdir();(case/'execution').mkdir();p=json.loads(plan_path.read_text());verify_evidence(p['evidence_sha256']);b=load_reference_bundle(p['generated_reference']);src=json.loads(Path(p['source_manifest']).read_text());baseline=Path(src['files']['model']).parent
 for name,key in [('scene.mjb','model'),('physics.json','physics'),('task.json','task')]:shutil.copy2(Path(p['scene_files'][key]) if p.get('execution_task') else baseline/name,case/name)
 shutil.copy2(plan_path,case/'offline_plan.json');shutil.copy2(p['generated_reference']['arrays_path'],case/'reference.npz')
 with np.load(p['scene_files']['initial']) as z:ini={k:z[k].copy() for k in z.files}
 model=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));obj=model.body(p['prior']['bindings']['object']).id;addr=model.jnt_qposadr[model.body_jntadr[obj]]
 q=ini['qpos'].copy() if navigation else b['qpos'][0].copy()
 if not navigation:
  q[addr:addr+7]=ini['qpos'][addr:addr+7];ini['qpos']=q;ini['qvel'][:]=0;ini['ctrl'][:]=0;ini['qacc_warmstart'][:]=0
 np.savez_compressed(case/'initial.npz',**ini)
 strategy=json.loads(Path('/home/kinetra-4090-01/code/zijian/experiments/2026-10/20/20261020_1710_workflow_parking_final/baseline/request.json').read_text())['strategy']
 if p.get('execution_task'):strategy=copy.deepcopy(p['execution_task']['strategy'])
 strategy['policy']['control_dt_s']=.02
 cfg=dict(strategy=strategy,offset_m=p['offset_m'],feedback_gain=float(feedback),wrist_z_offset_m=float(zoffset),max_feedback_m=.06,entry_min_s=.3,entry_timeout_s=2.,closure_wait_max_s=2.,hold_s=1.,wrist_gate_m=.025,wrist_gate_rad=.25,force_stop_N=200.,obstacle_stop_N=50.,root_drift_stop_m=.15,simulation_budget_s=12.)
 if navigation:
  cfg={k:cfg[k] for k in ('strategy','offset_m','feedback_gain','wrist_z_offset_m','hold_s')}
  cfg.update(navigation=navigation,simulation_budget_s=float(navigation.get('simulation_budget_s',75.)))
 if p.get('execution_task'):cfg['task_family']=p['execution_task']['name']
 cfg['repository']=str(REPO);save(case/'config.json',cfg);shutil.copy2(Path(__file__),case/'run.py')
 if provenance is not None:save(case/'api_provenance.json',provenance)
 with no_physics():
  a,cfg=initialize(case);m,d=a.m,a.d
  np.savez_compressed(case/'controller_initial.npz',decoder_last=a.sonic.last,**{k:np.asarray([h[k] for h in a.sonic.history]) for k in a.sonic.history[0]})
  runtime_cmd=[str(REPO/'scripts/humano_python.sh'),str(REPO/'scripts/check_humano_runtime.py'),* ([] if privileged else ['--load-models']),'--models',a.b.e and json.loads((case/'task.json').read_text())['resources']['sonic_models'],'--out',str(case/'runtime.json')]
  with (case/'runtime.log').open('w') as log:subprocess.run(runtime_cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
  runtime=json.loads((case/'runtime.json').read_text())
  if privileged:
   runtime.update(actual_session_providers=a.providers,sessions_loaded=True,loaded_session_source='executing NativeAdapter; reused for physics',actual_loaded_model_files=a.model_files)
   (case/'runtime.json').write_text(json.dumps(runtime,indent=2))
  save(case/'actual_preflight_providers.json',a.providers)
  ob=a.observe();save(case/'initial_observation.json',ob)
  if not privileged:
   m.vis.global_.offwidth=960;m.vis.global_.offheight=600;r=mujoco.Renderer(m,height=600,width=960);cam=mujoco.MjvCamera();cam.lookat[:]=[q[0]+.2,q[1],.65];cam.distance=2.;cam.azimuth=-60;cam.elevation=-20;r.update_scene(d,camera=cam)
   from PIL import Image
   Image.fromarray(r.render()).save(case/'initial.png');r.close()
  if not navigation:
   assert ob['contacts']['robot_obstacle_N']<cfg['obstacle_stop_N']
   assert ob['contacts']['robot_object_N']<cfg['force_stop_N'] and ob['root_pose'][2]>=.5
  assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
  assert (p['selected'].get('reference_structurally_valid') if p.get('execution_task') else p['selected']['feasible']) and len(b['qpos'])>1
  checker=Path(inspect.getfile(type(a.b.e.task)))
  compare=dict(baseline=str(baseline),model_identical=True,physics_identical=True,declared_changes=['robot starts at generated operation entry, not original far pose','zero initial qvel/ctrl/warmstart; synthetic stationary SONIC history','apple world Y offset relative to source '+str(p['offset_m']),'mode1 for complete operation; no original token prefix','no walking/stopping calibration because navigation is not executed','measured closure gate and 1s final hold','preencoder right-wrist proportional correction gain '+str(feedback),'right wrist world Z target offset '+str(zoffset)+'m ramped quintic in0.3s'],not_a_matched_controller_comparison=True)
  if navigation:
   compare['declared_changes']=['Original far scene initial state and real source SONIC history','Apple lateral offset '+str(p['offset_m']),'Live SONIC navigation then measured quintic transition and replay; no readiness/contact/corridor veto','Right-wrist preencoder positional feedback gain '+str(feedback),'Native task success only; tracking/contact recorded as diagnostics','No original token prefix, direct decoder override or physical reset']
   if navigation.get('parking'):compare['declared_changes'].append('Legacy deceleration/upright handoff and finite planted-foot correction; retain standing legs through operation, soft settle expiry continues')
   if navigation.get('parking',{}).get('support_frame')=='measured_root':compare['declared_changes'].append('Inactive left wrist and torso follow measured root frame; active right wrist stays in task world frame')
   if navigation.get('parking',{}).get('leg_mode')=='source_replay':compare['declared_changes'].append('After measured parking, smoothly resume full replay body/leg movement; do not lock demonstrated weight shifts')
   if navigation.get('wrist_integral'):compare['declared_changes'].append('Existing bounded right-wrist integral feedback before encoder; reset bias at a reanchored retry: '+str(navigation['wrist_integral']))
  save(case/'comparison_diff.json',compare)
  base=json.loads((baseline/'experiment.json').read_text());c=copy.deepcopy(base);c.update(experiment_id=case.name,created_at=now(),record_status='frozen_ready')
  c['purpose']=dict(hypothesis='Generated lateral operation is physically trackable and can lift apple from the proposed support stance',baseline_paths=[str(baseline),str(plan_path)],planned_changes=compare['declared_changes'],invariants=['compiled scene, physics, PD, native checker unchanged','source hand command values retained; closure gate can delay timeline','no live state writes after initialization','all actual control uses SONIC mode1; no decoder override'],known_confounders=['new initialized stance isolates grasp; does not verify navigation','synthetic stationary history differs from source controller checkpoint','trajectory retargeting with source action access; not independent skill generation'])
  c['cases']=[dict(task='apple',offset_m=p['offset_m'],initialization='operation_entry_reset_only_before_physics',feedback_gain=feedback)]
  c['runtime'].update(actual_loaded_session_evidence=str(case/'runtime.json'),model_hashes=a.model_files,python_version=platform.python_version(),gear_sonic_package_version=runtime['versions']['gear_sonic'],render_device='disabled; privileged numerical evidence only' if privileged else 'MuJoCo EGL saved-state render; no physics in renderer')
  c['physics'].update(simulator_version=mujoco.__version__,robot_and_hand='G1 native Dex3 bilateral; unchanged source model, not Revo2',asset_and_initial_state_manifest={str(case/n):file_hash(case/n) for n in ['scene.mjb','initial.npz','controller_initial.npz']},physics_parameters_manifest=str(case/'physics.json'),physics_dt_s=.005,control_dt_s=.02,pd_parameters=dict(kp=a.sonic.kp.tolist(),kd=a.sonic.kd.tolist(),hand='unchanged robot native MotorCommand.hand_command'),action_permissions='SONIC mode1 references and source finger motor commands only; no state writes after reset',safety_gates=dict(root_height_min_m=.5,root_tilt_max_rad=float(np.pi/6),root_drift_stop_m=.15,robot_obstacle_N=50.,robot_object_N=200.),acceptance_source_and_hash={str(checker):file_hash(checker)},acceptance_thresholds=dict(native_lift_m=.09,full_reference_required=True,clean_quality_obstacle_peak_N=5.,stable_hold_diagnostic_clearance_m=.03,stable_hold_required_s=.5,hand_force_N=.1,support_force_max_N=.1))
  stages=[]
  for name in ['entry_stabilize','approach_close_lift','final_hold']:
   stages.append(dict(stage=name,mode='SONIC encoder mode1',lower_body_interface='generated leg q via fixed current 100-frame reference; no live locomotion planner calls',upper_body_interface='world wrists and torso FK targets to mode1; right xyz correction gain '+str(feedback),action_fields_units_frames='world xyz m, wxyz quaternion, joints rad; motor torque Nm',target_source=str(case/'reference.npz'),interpolation='200Hz saved reference samples; 50Hz SONIC and fingers zero order hold',ik_used='offline reference only',native_prefix_source_and_hash=dict(source='none',duration_s=0),handoff_rule='entry bilateral support; closure measured wrist alignment; bounded wait; terminal hold1s'))
  c['control']=dict(stage_interfaces=stages,stage_entry_required_fields=base['control']['stage_entry_required_fields'],fallback_allowed=False,actual_mode_log=str(case/'execution/controller.json'),original_token_steps=0,post_decoder_identity=True)
  snap=case/'source_snapshot';snap.mkdir();source_files=[Path(__file__).resolve(),REPO/'humanoid_mimicgen/augmentation/lateral_tools.py',REPO/'humanoid_mimicgen/augmentation/navigation_api.py',REPO/'humanoid_mimicgen/augmentation/four_task_navigation.py',REPO/'humanoid_mimicgen/augmentation/button_acceptance.py',REPO/'humanoid_mimicgen/augmentation/privileged_lateral.py',REPO/'humanoid_mimicgen/augmentation/replay_navigation.py',REPO/'humanoid_mimicgen/augmentation/lateral_agent.py',REPO/'humanoid_mimicgen/augmentation/lateral_delivery.py']+list((REPO/'humanoid_mimicgen/workflow/navigation').glob('*.py'))+[REPO/'humanoid_mimicgen/workflow/native.py',REPO/'humanoid_mimicgen/workflow/sonic_record.py',REPO/'humanoid_mimicgen/standing_handoff.py',REPO/'humanoid_mimicgen/action_replay.py',REPO/'src/cap_sim/backends/native_sonic.py',REPO/'src/cap_sim/backends/humano.py',checker]
  hashes={}
  source_files += [REPO/'humanoid_mimicgen/augmentation/replay_parking.py',REPO/'humanoid_mimicgen/workflow/motion.py',REPO/'humanoid_mimicgen/workflow/parking.py',REPO/'humanoid_mimicgen/workflow/planning.py',REPO/'humanoid_mimicgen/workflow/feedback.py',REPO/'humanoid_mimicgen/stationary_planning.py',REPO/'humanoid_mimicgen/planned_approach.py']
  for j,f in enumerate(source_files):dest=snap/(f'{j:03d}_'+f.name);shutil.copy2(f,dest);hashes[str(dest)]=file_hash(dest);hashes[str(f)]=file_hash(f)
  skills=[Path('/home/kinetra-4090-01/.codex/skills')/x/'SKILL.md' for x in ['grasp-prior-refinement','g1-grasp-runtime-review','sim-visual-realism-review']]
  c['pipeline'].update(entrypoint='scripts/humano_python.sh -m humanoid_mimicgen.augmentation.direct_grasp run '+str(case),ordered_steps=['inspect saved initial','verify frozen hashes and loaded providers','initialize stance and declared history once','entry','approach/closure/lift','hold','independent metrics and saved-state render'],source_snapshot_and_hashes=hashes,git_commit_and_dirty_patch=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),experiment_script_sha256=file_hash(case/'run.py')),skill_versions_and_hashes=[dict(path=str(x),sha256=file_hash(x)) for x in skills],planner_identity_and_prompt_hash=dict(identity='session authored deterministic direct-stance probe; original reference accessible',prompt_path='/home/kinetra-4090-01/code/zijian/policy_memory/grasp_learning/prompts/human_prior_v7.md'),external_model_calls={'before_launch':provenance.get('api_calls_before_launch',0),'during_physics':0} if provenance else 0,original_actions_visible=True,feedback_and_reflection_rules='record failed and successful attempts; at most3 cases; next run changes one hypothesis; no live retries',allowed_retry_changes=['new frozen case after reflection'],optimization_enabled=False)
  c['budget']=dict(attempts_per_case=1,recovery_attempts=0,simulation_seconds_per_attempt=12.,feedback_calls=0,optimization_attempts=0)
  c['metrics']=dict(target_definition='saved generated q FK at applied policy tick; source row and intended pose retained',sampling_and_alignment='pre-state/control and post-state every .005s; exact held target logged',tracking_error_definition='measured post-step wrist vs applied original desired pose; encoder-corrected pose logged separately',endpoint_error_definition='final applied target at truncation; never label unexecuted source end reached',orientation_error_definition='relative SO3 angle radians',base_drift_definition='measured horizontal root displacement from initialized stance; feet separately',failure_truncation_rule='only executed substeps included; last next_state retained; source coverage separately')
  evidence=([case/'api_provenance.json'] if provenance is not None else [])+[case/n for n in ['run.py','config.json','scene.mjb','initial.npz','controller_initial.npz','task.json','physics.json','reference.npz','offline_plan.json','initial_observation.json','runtime.json','actual_preflight_providers.json','comparison_diff.json']]+skills
  if navigation:
   seed=Path(json.loads((case/'task.json').read_text())['resources']['initial_state']);evidence.append(seed)
   c['runtime']['controller_history_seed']={'path':str(seed),'sha256':file_hash(seed)}
  hashes.update({str(x):file_hash(x) for x in evidence});hashes.update(a.model_files)
  if provenance is not None:
   verify_evidence(provenance['evidence_sha256']);hashes.update(provenance['evidence_sha256']);c['pipeline']['planner_identity_and_prompt_hash']=dict(identity='Responses API tool decisions with deterministic local IK/physics' if provenance.get('transport')=='responses' else 'Declared local replay-first reference reuse; no new API decision',provenance=str(case/'api_provenance.json'),sha256=file_hash(case/'api_provenance.json'))
  c['preflight']=dict(passed=True,comparison_diff_path=str(case/'comparison_diff.json'),evidence_paths=[str(x) for x in evidence],evidence_sha256=hashes,initial_visual_review='not required; user requested privileged_only; no image generated' if privileged else 'initial.png must be inspected before run; see launch log')
  if privileged:
   c['pipeline']['ordered_steps']=['verify cached plan and declared numeric decision','load controller once and numeric initial checks','freeze experiment before physics','entry support gate','approach/closure/lift','hold','independent numeric audit; no rendering']
   c['pipeline']['perception_mode']='privileged_only'
   c['pipeline']['rendering_required']=False
   c['pipeline']['adapter_reused_after_preflight']=keep_adapter
  c['control']['action_recording']=dict(body='SONIC token each actual policy tick; no mode0/prefix fallback',hands='14 separate commands each physical step; not encoded by body token',mapping='execution/sonic_actions.npz and sonic_actions.json; every executed step must map to a token')
  if navigation:
   c['purpose'].update(hypothesis='Navigate from original far initial state and complete apple task through continuous replay reuse',planned_changes=compare['declared_changes'],invariants=['Scene, physics, PD and native predicate unchanged','Source finger commands; SONIC mode1 only; no live state writes'],known_confounders=['Trajectory retargeting with original action access','No extra contact/pose readiness gates; measured task outcome decides'])
   c['cases'][0]['initialization']='original_far_scene_state_and_real_controller_history'
   c['physics']['safety_gates']={'extra_contact_pose_corridor_stops':False,'simulator_nonfinite_error':'execution error'}
   c['physics']['acceptance_thresholds']={'native_task_predicate':True,'completed_replay_and_hold':True,'contact_and_tracking':'diagnostics only'}
   c['control']['stage_interfaces']=[dict(stage=name,mode='SONIC encoder mode1',lower_body_interface='live locomotion planner' if name=='navigation' else 'measured-to-replay body reference',upper_body_interface='world wrist/torso FK and original finger commands',action_fields_units_frames='world xyz m, wxyz quaternion, joint rad; actuator Nm',target_source=str(case/'reference.npz'),interpolation='200Hz physics;50Hz SONIC; quintic measured transition',ik_used='offline reference only',native_prefix_source_and_hash={'duration_s':0},handoff_rule='target arrival then timed continuous reference') for name in ['navigation','transition','operation','hold']]
   c['control']['pose_feedback']=dict(interface='right wrist before SONIC encoder; source desired and corrected encoder_pose recorded separately',position_gain=feedback,orientation_gain=navigation.get('orientation_feedback_gain',0.),correction_clipping=False,decoder_overrides=False)
   if navigation.get('wrist_integral'):c['control']['pose_feedback']['integral']=dict(navigation['wrist_integral'],applied_stage='transition/operation/hold',reset_on='new reanchored retry; no physics/controller history reset')
   c['pipeline'].update(entrypoint='scripts/humano_python.sh -m humanoid_mimicgen.augmentation.replay_navigation run '+str(case),ordered_steps=['freeze original initial/runtime/source','navigate to target','measured transition','replay source reference','hold','record native outcome and all tokens'])
   c['budget']['simulation_seconds_per_attempt']=cfg['simulation_budget_s']
   if navigation.get('parking'):
    c['control']['stage_interfaces'].insert(1,dict(stage='parking',mode='SONIC encoder mode1',lower_body_interface='upright blended standing reference; up to2 new planted-foot IK references; never physical state writes',upper_body_interface='measured world wrist/torso hold',action_fields_units_frames='world m/wxyz, joints rad; actuator Nm',target_source='actual measured root/feet and source entry leg posture',interpolation='quintic standing blend; finite settle/correction windows',ik_used='reference-only leg IK',native_prefix_source_and_hash={'duration_s':0},handoff_rule='settled or dwell expires; failed/no-progress correction proceeds to operation'))
    for stage in c['control']['stage_interfaces']:
     if stage['stage'] in ('transition','operation','hold'):stage['lower_body_interface']='retain corrected upright standing legs; no switch back to replay leg reference'
    c['control']['support_reference_frame']=navigation['parking'].get('support_frame','source_world')
    if navigation['parking'].get('leg_mode')=='source_replay':
     compare_source='Measured posture smoothly transitions into complete source body reference; preserve demonstrated leg/weight-shift motion after parking'
     c['purpose']['planned_changes'].append(compare_source)
     for stage in c['control']['stage_interfaces']:
      if stage['stage'] in ('transition','operation','hold'):stage['lower_body_interface']='measured-to-source quintic body transition, then full source leg reference through SONIC mode1'
    c['pipeline']['ordered_steps']=['freeze original initial/runtime/source','decelerating waypoint navigation','upright blended standing','finite measured parking correction','measured arm transition with retained standing legs','source replay and hold','record native outcome and all tokens']
    if navigation['parking'].get('leg_mode')=='source_replay':c['pipeline']['ordered_steps'][4]='measured whole-body transition into complete source replay'
   if navigation.get('retry'):
    c['budget']['recovery_attempts']=navigation['retry']['max_retries']
    c['pipeline']['feedback_and_reflection_rules']='After a completed source operation, retry failed acquisition or lost grasp up to declared in-run retry count; no physical resets or implicit re-execution'
    c['pipeline']['allowed_retry_changes']=['original open hand command','current-root arm/waist posture restoration','measured apple translation applied to active wrist; preserve reference orientation','restart original replay hand sequence']
    c['control']['retry_target_planning']='Measured-root arm/waist IK screens translated approach entry; infeasible target records retry_target_unreachable, never commands a floor target through a table-grasp reference'
    for name in ['recovery_open','recovery_restore']:
     c['control']['stage_interfaces'].append(dict(stage=name,mode='SONIC encoder mode1',lower_body_interface='retain corrected standing leg reference',upper_body_interface='measured-root FK of restored arm and waist joints; no physical state writes',action_fields_units_frames='world m/wxyz, joints rad; actuator Nm',target_source='normal measured pre-operation arm/waist posture',interpolation='quintic joint-space reference with finite measured restoration window',ik_used=False,native_prefix_source_and_hash={'duration_s':0},handoff_rule='restored for0.3s or finite window expires; reobserve object and restart source approach'))
  if p.get('execution_task'):
   info=p['execution_task'];comparison=json.loads(Path(info['scene_comparison']).read_text())
   from .four_task_navigation import feedback_metadata
   sides=cfg['strategy']['bindings']['hands'];c['control']['pose_feedback']=feedback_metadata(sides,feedback,navigation.get('orientation_feedback_gain',0.))
   for stage in c['control']['stage_interfaces']:
    if stage['stage'] in ('transition','operation','hold'):stage['upper_body_interface']='World wrists/torso FK and original fingers; preencoder feedback on '+str(sides)
   c['cases'][0]['task']=info['name'];c['purpose']['hypothesis']='Precision navigation and translated successful whole-body replay at a new lateral target position'
   c['purpose']['planned_changes']=[comparison,'Full source operation rigidly translated with object; source fingers unchanged; no original token prefix','Fixed precision parking, measured-to-replay transition and complete SONIC mode1 operation']
   c['purpose']['invariants']=['Original robot initial state, physics, native criterion and source controller history unchanged','Only declared target world translation; no geometric collision certificate inferred','No decoder override or live physical state reset']
   c['physics']['acceptance_thresholds']={'native_task_predicate':True,'completed_replay_and_hold':True,'approved_profile':info['acceptance_profile'],'grasp_retention_applicable':info['name'] in ('apple','plate'),'contact_and_tracking':'diagnostics separately reported; button reviewed palm criterion audited independently'}
   decision_kind='API-authorized numerical decision' if provenance and provenance.get('transport')=='responses' else 'explicit deterministic verified-source baseline; no API authorization'
   c['pipeline']['decision_origin']=dict(kind=decision_kind,transport=provenance.get('transport') if provenance else 'local')
   c['pipeline']['ordered_steps']=['freeze task-specific scene/translated source/'+decision_kind,'precision navigation from original far robot state','measured parking and full-body transition','source operation and hold; all body tokens and fingers','task-specific native/contact/approved-palm audit']
   c['pipeline']['feedback_and_reflection_rules']='One '+decision_kind+' per task/position; no physical recovery retries; checkpoint captured only for this condition'
   c['pipeline']['allowed_retry_changes']=[]
   c['control'].pop('retry_target_planning',None)
   c['physics']['scene_comparison']=comparison
  if navigation and navigation.get('checkpoint'):
   from .navigation_checkpoint import bind_contract
   bind_contract(case,c,cfg)
  c['execution']=dict(started_at='pending; execution_audit.json',ended_at='pending; execution_audit.json',contract_sha256='experiment.sha256',audit_path='execution/execution_audit.json',timing_path='execution/execution_audit.json',results_path='result.json')
  save(case/'experiment.json',c);(case/'experiment.sha256').write_text(file_hash(case/'experiment.json'));save(case/'prepare_timing.json',dict(ended_at=now(),wall_s=time.monotonic()-begin,physics_steps=0))
  if not keep_adapter:a.close()
 (root/'latest_case.txt').write_text(str(case));print(str(case),flush=True);return (case,a,cfg) if keep_adapter else case

def run(case,prepared_adapter=None):
 c=json.loads((case/'experiment.json').read_text());assert file_hash(case/'experiment.json')==(case/'experiment.sha256').read_text();verify_evidence(c['preflight']['evidence_sha256'])
 a,cfg=prepared_adapter if prepared_adapter is not None else initialize(case);m,d=a.m,a.d;out=case/'execution';assert not (out/'trajectory.npz').exists()
 z=np.load(case/'reference.npz');qs=z['qpos'];fingers=z['fingers'];source_rows=z['source_rows'];fk=BodyReference(m,{**cfg['strategy']['bindings']['wrists'],'torso':cfg['strategy']['bindings']['torso']},[])
 dt=a.dt;assert dt==.005 and a.control_stride==4
 obs=[];states=[];nexts=[];cmds=[];commands=[];events=[];stage='entry';i=0;failure=None;completed=False;native=False;start=now();clock=time.monotonic();entryquiet=0.;closurewait=0.;terminal_at=None;initialq=d.qpos.copy();lastf=fingers[0].copy();firstclosure=int(np.where(np.max(abs(fingers-fingers[0]),axis=1)>1e-6)[0][0]);entered=False
 def target_poses(q):
  p,_=fk.poses(q);u=min(1.,float(d.time)/.3);p['right'][2]+=cfg.get('wrist_z_offset_m',0.)*(10*u**3-15*u**4+6*u**5);return p
 def pose_error(p):
  actual=a.poses();return {s:dict(position_m=float(np.linalg.norm(actual[s][:3]-p[s][:3])),angle_rad=float((rot(p[s][3:]).inv()*rot(actual[s][3:])).magnitude())) for s in ['left','right','torso']}
 try:
  for k in range(int(cfg['simulation_budget_s']/dt)):
   tick=k%4==0;ob=a.observe();stable=min(ob['foot_ground_N'])>10 and ob['root_speed_m_s']<.08 and ob['yaw_speed_rad_s']<.3
   if stage=='entry':
    entryquiet=entryquiet+dt if stable else 0
    if tick and k*dt>=cfg['entry_min_s'] and entryquiet>=.1:stage='operation';entered=True;events.append(dict(event='entry_supported',time_s=float(d.time)))
    elif k*dt>=cfg['entry_timeout_s']:raise WorkflowError('initial_stance_unstable')
   if stage=='operation' and i==firstclosure:
    p=target_poses(qs[i]);e=pose_error(p);ready=stable and e['right']['position_m']<=cfg['wrist_gate_m'] and e['right']['angle_rad']<=cfg['wrist_gate_rad']
    if not ready:
     closurewait+=dt
     if closurewait>=cfg['closure_wait_max_s']:raise WorkflowError('pregrasp_tracking_timeout')
    elif tick:events.append(dict(event='closure_started',time_s=float(d.time),source_row=int(source_rows[i]),error=e['right']))
   else:ready=True
   q=qs[i];desired=target_poses(q)
   if tick:
    latched_desired={s:v.copy() for s,v in desired.items()};poses={s:v.copy() for s,v in desired.items()};error=desired['right'][:3]-a.poses()['right'][:3];corr=cfg['feedback_gain']*error
    corr*=min(1.,cfg['max_feedback_m']/max(np.linalg.norm(corr),1e-12));poses['right'][:3]+=corr
    set_body_reference(a.sonic,q);lastf=fingers[i].copy() if ready else fingers[max(0,i-1)].copy();applied_i=i
   a.command(poses,np.zeros(2),float(rot(q[3:7]).as_euler('xyz')[2]),float(q[2]),lastf,policy_tick=tick)
   states.append(np.r_[d.time,d.qpos.copy(),d.qvel.copy()]);cmds.append(np.r_[d.time,a.last_target,d.ctrl.copy()]);commands.append(dict(time_s=float(d.time),stage=stage,source_row=int(source_rows[applied_i]),body_target_rad=a.last_target.tolist(),ctrl_Nm=d.ctrl.tolist(),policy_tick=tick,desired={s:v.tolist() for s,v in latched_desired.items()},encoder_pose={s:v.tolist() for s,v in poses.items()},fingers=lastf.tolist()))
   a.step();nexts.append(np.r_[d.time,d.qpos.copy(),d.qvel.copy()]);o=a.observe();o.update(stage=stage,source_row=int(source_rows[applied_i]),tracking=pose_error(latched_desired),feet={s:np.r_[d.xpos[m.body(n).id],d.xquat[m.body(n).id]].tolist() for s,n in cfg['strategy']['bindings']['feet'].items()});obs.append(o)
   if (k+1)%4==0:native=a.native_success() or native
   tilt=float(np.arccos(np.clip(rot(d.qpos[3:7]).as_matrix()[2,2],-1,1)))
   if d.qpos[2]<.5 or tilt>np.pi/6:raise WorkflowError('fall')
   if np.linalg.norm(d.qpos[:2]-initialq[:2])>cfg['root_drift_stop_m']:raise WorkflowError('root_drift_limit')
   if o['contacts']['robot_obstacle_N']>cfg['obstacle_stop_N']:raise WorkflowError('obstacle_force_limit')
   if o['contacts']['robot_object_N']>cfg['force_stop_N']:raise WorkflowError('object_force_limit')
   if any(w.number for w in d.warning):raise WorkflowError('simulation_warning')
   if stage=='operation' and ready:
    # A changed finger target must wait until a policy tick is able to transmit it.
    nxt=min(i+1,len(qs)-1)
    if nxt==firstclosure and (k+1)%4!=0:pass
    elif i<len(qs)-1:i=nxt
    elif (k+1)%4==0:stage='hold';terminal_at=float(d.time)
   if stage=='hold' and float(d.time)-terminal_at>=cfg['hold_s']:completed=True;break
  if not completed:failure='simulation_budget_exhausted'
 except Exception as ex:
  failure=ex.code if isinstance(ex,WorkflowError) else type(ex).__name__;save(out/'error.json',dict(failure=failure,detail=str(ex)))
 finally:
  wall=time.monotonic()-clock;ended=now();np.savez_compressed(out/'trajectory.npz',state=np.asarray(states),next_state=np.asarray(nexts),commands=np.asarray(cmds));np.savez_compressed(out/'terminal.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy());save(out/'observations.json',obs);save(out/'commands.json',commands);save(out/'events.json',events);save(out/'controller.json',a.sonic.records);save(out/'actual_providers.json',a.providers);save(out/'actual_model_files.json',a.model_files)
  from humanoid_mimicgen.workflow.sonic_record import save_action_bundle
  try:sonic_record=save_action_bundle(out,a.sonic.records,[dict(c,fingers_rad=c['fingers']) for c in commands[:len(nexts)]],dt,.02,expected_physics_steps=len(nexts))
  except WorkflowError as exc:
   sonic_record=dict(coverage_complete=False,reason=exc.code,detail=exc.detail);completed=False;failure=failure or exc.code
  modes=sorted(set(x['mode'] for x in a.sonic.records));assert modes==[1] if len(states) else not modes
  audit=dict(experiment_id=case.name,sonic_action_record=sonic_record,started_at=start,ended_at=ended,simulation_compute_wall_s=wall,physics_steps=len(states),sim_seconds=len(states)*dt,completed=completed,failure=failure,stage=stage,actual_encoder_modes=modes,original_token_steps=0,mode0_calls=0,midrun_state_resets=0,post_decoder_overrides=0,api_calls_during_physics=0,api_orchestration_provenance=str(case/'api_provenance.json') if (case/'api_provenance.json').exists() else None,contract_sha256=file_hash(case/'experiment.json'),native_success=native,initialization='generated entry stance with zero velocity and synthetic stationary policy history',navigation_executed=False,source_row_last=int(source_rows[i]),entry_supported=entered,closure_wait_s=closurewait,training_ready=False)
  save(out/'execution_audit.json',audit)
  good=longest=0
  for o in obs:
   okay=o['clearance_m']>=.03 and o['contacts']['hand_object_N']>.1 and o['contacts']['support_N']<.1
   good=good+1 if okay else 0;longest=max(longest,good)
  maxobs=max((o['contacts']['robot_obstacle_N'] for o in obs),default=0);hold=longest*dt
  result=dict(experiment_id=case.name,task='G1WholebodyTableApplePickTeleop',mode='直接站位SONIC_mode1',success=bool(completed and native and hold>=.5 and maxobs<=5),native_success=native,completed=completed,failure=failure,stage=stage,physics_steps=len(states),expected_steps=2400,navigation_executed=False,offset_m=cfg['offset_m'],max_clearance_m=max((o['clearance_m'] for o in obs),default=0),unsupported_hand_contact_clearance_hold_s=hold,max_obstacle_N=maxobs,obstacle_contact_gt_0p1_s=sum(o['contacts']['robot_obstacle_N']>.1 for o in obs)*dt,max_hand_object_N=max((o['contacts']['hand_object_N'] for o in obs),default=0),max_root_xy_drift_m=max((float(np.linalg.norm(np.asarray(o['root_pose'][:2])-initialq[:2])) for o in obs),default=0),wrist_tracking={s:{key:dict(rmse=float(np.sqrt(np.mean(np.square([o['tracking'][s][key] for o in obs])))),max=max(o['tracking'][s][key] for o in obs),p95=float(np.percentile([o['tracking'][s][key] for o in obs],95))) for key in ['position_m','angle_rad']} for s in ['left','right'] } if obs else {},audit=audit,training_ready=False)
  save(case/'result.json',result);save(case/'reflection.json',dict(observed=result,hypothesis='conditional stance physical grasp feasibility',scope='G1 Dex3, this source/stance/offset/controller initialization only',status='observed',next='inspect tracking, contacts and feet before choosing next bounded case'));a.close();print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],float(sys.argv[5]),float(sys.argv[6]) if len(sys.argv)>6 else 0.,json.loads(Path(sys.argv[7]).read_text()) if len(sys.argv)>7 else None)
 elif sys.argv[1]=='run':run(Path(sys.argv[2]))
 elif sys.argv[1]=='cycle':
  case,a,cfg=prepare(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],float(sys.argv[5]),float(sys.argv[6]),json.loads(Path(sys.argv[7]).read_text()),keep_adapter=True)
  run(case,(a,cfg))
