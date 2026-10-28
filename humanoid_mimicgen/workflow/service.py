"""Independent public workflow orchestration with content-bound preflight."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from .contracts import WorkflowError, canonical, digest, file_hash, now, validate_strategy
from .store import Store
from .profile import validate_policy
from .planning import candidate_plan, rank_candidates, validate_source, nominal_stance
from .decisions import ResponsesDecision, replay_decision

REPO=Path(__file__).resolve().parents[2]


def input_context(files):
    result={}
    for name,path in files.items():
        p=Path(path)
        if not p.is_file():raise WorkflowError('invalid_input','Missing evidence '+name)
        result[name]=file_hash(p)
    return result


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(canonical(value))


def implementation_files():
    files={}
    for folder in (REPO/'humanoid_mimicgen',REPO/'src/cap_sim'):
        for path in folder.rglob('*.py'):
            if 'tests' not in path.parts:files[str(path.relative_to(REPO))]=path
    return files


def _strategy(request):
    s=copy.deepcopy(request['strategy']);s['context']=input_context(request['files'])
    s['provenance']['implementation_sha256']={name:file_hash(path) for name,path in implementation_files().items()}
    if s['policy']['compensation']['mode']=='bounded_integral':
        if s['policy']['compensation']['evidence_file'] not in request['files']:raise WorkflowError('invalid_input','Missing feedback evidence')
    if s['policy']['compensation']['mode']=='identified_equilibrium':
        from .calibration import identify_response
        comp=s['policy']['compensation'];scope={'adapter':s['adapter'],'model':s['context']['model'],'initial':s['context']['initial'],'bindings':s['bindings']}
        calibrated={}
        for hand in s['bindings']['hands']:
            name='calibration_'+hand
            if name not in request['files']:raise WorkflowError('needs_calibration','Missing measured response for '+hand)
            trace=np.load(request['files'][name],allow_pickle=False)
            if 'dt_s' not in trace or not np.isclose(float(trace['dt_s']),s['policy']['control_dt_s']):raise WorkflowError('needs_calibration','Calibration/control sample period mismatch')
            if json.loads(str(trace['scope_json'].item()))!=scope:raise WorkflowError('needs_calibration','Measured response scope changed')
            calibrated[hand]=identify_response(trace['command_xyz'],trace['measured_xyz'],scope=scope,max_prediction_error_m=comp['max_prediction_error_m'])
        comp.update(scope=scope,calibrations=calibrated)
    if 'parking' in s['policy']['navigation']:
        from .parking import response_scope,validate_calibration
        from .native import sonic_files
        park=s['policy']['navigation']['parking'];name=park['evidence_file']
        if name not in request['files']:raise WorkflowError('needs_calibration','Missing parking calibration')
        record=json.loads(Path(request['files']['record']).read_text())
        scope=response_scope(s,s['context']['model'],s['context']['reference_states'],file_hash(record['physics']['physics_parameters_manifest']),sonic_files(json.loads(Path(request['files']['task']).read_text())))
        park['model']=validate_calibration(json.loads(Path(request['files'][name]).read_text()),scope,park)
    if 'grasp_trigger' in s['policy']:
        trigger=s['policy']['grasp_trigger'];name=trigger['preflight_evidence_file']
        if name not in request['files'] or request['files'].get('grasp_judge_executable')!=trigger['executable']:raise WorkflowError('preflight_failed','Grasp judge inputs not bound')
        proof=json.loads(Path(request['files'][name]).read_text())
        if proof.get('status')!='completed' or proof.get('model_requested')!=trigger['model'] or proof.get('executable_sha256')!=file_hash(trigger['executable']):raise WorkflowError('preflight_failed','Grasp judge identity mismatch')
        for path,expected in proof['evidence_sha256'].items():
            if file_hash(path)!=expected:raise WorkflowError('preflight_failed','Grasp judge connectivity evidence changed')
    if s['policy']['navigation'].get('profile')=='replay_navigation':
        from .navigation.integration import validate_bound_plan
        path=request['files'].get('navigation_plan')
        if not path or s['policy']['navigation'].get('plan_file')!=str(Path(path).resolve()):
            raise WorkflowError('preflight_failed','Navigation plan_file must match bound input')
        validate_bound_plan(json.loads(Path(path).read_text()),request['files'],s)
        from .launch_evidence import navigation_admission
        from .parking import response_scope
        from .native import sonic_files
        nav=s['policy']['navigation'];proof_name=nav.get('control_evidence_file')
        if proof_name not in request['files']:raise WorkflowError('needs_calibration','Bound navigation calibration missing')
        record=json.loads(Path(request['files']['record']).read_text())
        scope=response_scope(s,s['context']['model'],s['context']['reference_states'],file_hash(record['physics']['physics_parameters_manifest']),sonic_files(json.loads(Path(request['files']['task']).read_text())))
        s['provenance']['navigation_admission']=navigation_admission(request,json.loads(Path(request['files'][proof_name]).read_text()),scope,nav['calibration_quality'],nav['route_limits'])

    validate_strategy(s);validate_policy(s['policy'])
    return s


def _geometry(request,s,proposal=None):
    import mujoco
    from ..privileged import no_physics
    from ..stance_selection import yaw_of
    if s['policy']['navigation'].get('profile')=='replay_navigation':
        from .navigation.integration import validate_bound_plan
        plan=json.loads(Path(request['files']['navigation_plan']).read_text())
        selected=validate_bound_plan(plan,request['files'],s)
        if s['policy']['navigation']['navigation_only'] is False:
            from .navigation.integration import validate_operation_inputs
            alignment=json.loads(Path(request['files']['alignment']).read_text())
            validate_operation_inputs(plan,request['files'],s,alignment['command_dt_s'])
            from .navigation.whole_body import validate_operation_model
            model=mujoco.MjModel.from_binary_path(request['files']['model'])
            validate_operation_model(model,s)
        return {'selected':selected,'candidates':plan['candidates'],'physics_steps':0,
                'dynamic_execution_verified':False,'evidence':{'navigation_plan':plan['evidence_sha256']}}
    files=request['files'];targets=dict(np.load(files['targets']));initial=np.load(files['initial'])['qpos']
    source=np.load(files['reference_states'])['samples'];alignment=json.loads(Path(files['alignment']).read_text())
    validate_source(targets,s,alignment);entry=s['phases'][0]['start']+s['command_pose_offset']
    with no_physics():
        model=mujoco.MjModel.from_binary_path(files['model']);data=mujoco.MjData(model)
        if len(initial)!=model.nq or source.ndim!=2 or len(source)<s['source_frame_count']:
            raise WorkflowError('invalid_input','State dimensions')
        data.qpos[:]=initial;mujoco.mj_forward(model,data)
        objid=model.body(s['bindings']['object']).id
        obj=data.xpos[objid].copy();target_object=np.r_[obj,data.xquat[objid]]
        data.qpos[:]=source[entry,1:1+model.nq];mujoco.mj_forward(model,data)
        source_object=np.r_[data.xpos[objid],data.xquat[objid]]
        nominal=nominal_stance(source[entry,1:8],source_object,target_object)
        candidates=[{'id':'current','xyyaw':[*initial[:2],yaw_of(initial[3:7])]},
                    {'id':'object_relative_demonstration','xyyaw':nominal.tolist()}]
        if 'stance_hint' in files:
            hint=json.loads(Path(files['stance_hint']).read_text())
            candidates=[{'id':'preset_stance','xyyaw':hint['xyyaw']}]
        if proposal:candidates+=proposal['stances']
        rows=[]
        for candidate in candidates:
            rows.append(candidate_plan(model,initial,targets,entry,s['bindings'],s['policy'],candidate))
        selected=rank_candidates(rows)
        return {'selected':selected,'candidates':rows,'physics_steps':0,'dynamic_execution_verified':False,
                'evidence':{'scene':{'object_xyz':obj.tolist(),'initial_root':initial[:7].tolist()},
                            'failed_geometry':rows,'source':{'entry_frame':entry,'hands':s['bindings']['hands'],'nominal_stance_xyyaw':nominal.tolist()}}}


def plan(request):
    """Save a new context-bound strategy; this function never advances physics."""
    from ..cli import create_experiment
    s=_strategy(request)
    work=create_experiment(request['experiment_root'],'workflow_plan')
    write_new(work/'request.json',request);start=time.monotonic();status='failed'
    try:
        geometry=_geometry(request,s)
        decision=request['decision'];proposal=None;mode=decision['mode']
        if mode=='saved':
            proposal=replay_decision(decision['path'],geometry['evidence'],provider_identity=decision['identity'])
            write_new(work/'saved_decision.json',{'identity':decision['identity'],'proposal':proposal,'source_sha256':file_hash(decision['path'])})
        elif mode=='responses' and (geometry['selected'] is None or decision.get('request_alternatives',False)):
            provider=ResponsesDecision(decision['model'],decision['base_url'],decision['key_env'],decision['timeout_s'])
            proposal=provider.propose(geometry['evidence'],work/'decision')
        elif mode not in ('deterministic','responses'):raise WorkflowError('invalid_input','Decision mode')
        if proposal:
            if proposal['action']!='propose_stances':raise WorkflowError('needs_observation' if proposal['action']=='needs_observation' else 'unsupported_capability',proposal['reason'])
            geometry=_geometry(request,s,proposal)
        write_new(work/'geometry.json',geometry)
        if geometry['selected'] is None:raise WorkflowError('unreachable','No checked stance; request a model proposal or new evidence')
        s['proposal']={'stance':geometry['selected'],'decision':proposal,'method':'geometry_checked_candidates'}
        artifacts={f'inputs/{name}{Path(path).suffix}':path for name,path in request['files'].items()}
        artifacts.update({f'implementation/{name}':path for name,path in implementation_files().items()})
        artifacts.update({f'planning/{p.relative_to(work).as_posix()}':p for p in work.rglob('*') if p.is_file() and p.name!='request.json'})
        # The request contains deployment paths; the immutable strategy context contains hashes.
        ident=Store(request['strategy_root']).save(s,artifacts);status='completed'
        result={'status':status,'strategy_id':ident,'planning_directory':str(work),'physics_steps':0}
        write_new(work/'result.json',result);return result
    except Exception as exc:
        write_new(work/'failure.json',{'status':'failed','reason_code':exc.code if isinstance(exc,WorkflowError) else 'planning_error','error_type':type(exc).__name__});raise
    finally:write_new(work/'timing.json',{'ended_at':now(),'planning_wall_s':time.monotonic()-start,'physics_steps':0,'status':status})


def check_contract(path):
    path=Path(path)
    try:
        if file_hash(path)!=(path.parent/'experiment.sha256').read_text().strip():raise WorkflowError('preflight_failed','Contract digest')
        c=json.loads(path.read_text())
        if c['record_status']!='frozen_ready' or not c['preflight']['passed']:raise WorkflowError('preflight_failed','Not frozen ready')
        validate_strategy(c['workflow']['strategy']);validate_policy(c['workflow']['strategy']['policy'])
        hashes=c['preflight']['evidence_sha256']
        if str(Path(__file__).resolve()) not in hashes or str(Path(__file__).with_name('runner.py')) not in hashes:
            raise WorkflowError('preflight_failed','Workflow source not bound')
        for name in ('model','initial','targets','task','reference_states'):
            if c['inputs'][name] not in hashes:raise WorkflowError('preflight_failed','Input not bound '+name)
        if c['runtime']['actual_loaded_session_evidence'] not in hashes:raise WorkflowError('preflight_failed','Runtime not bound')
        for name,sha in hashes.items():
            if not Path(name).is_file() or file_hash(name)!=sha:raise WorkflowError('preflight_failed','Evidence changed '+name)
        from .native import sonic_files
        for path,sha in sonic_files(json.loads(Path(c['inputs']['task']).read_text())).items():
            if hashes.get(path)!=sha:raise WorkflowError('preflight_failed','Executed network not bound')
        runtime=json.loads(Path(c['runtime']['actual_loaded_session_evidence']).read_text())
        if not runtime['passed'] or not runtime['sessions_loaded'] or len(runtime['actual_session_providers'])!=3 or any(v!=['CPUExecutionProvider'] for v in runtime['actual_session_providers'].values()):raise WorkflowError('preflight_failed','Actual runtime providers')
        if runtime['versions']['tensorrt']!='10.13.0.35' or runtime['versions']['onnxruntime']!='1.23.2' or runtime['sonic_version']!='1.0':raise WorkflowError('preflight_failed','Runtime versions')
        if c['control']['fallback_allowed'] or c['budget']['attempts_per_case']!=1:raise WorkflowError('preflight_failed','Execution scope')
        return c
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise WorkflowError('preflight_failed','Incomplete frozen evidence: '+type(exc).__name__) from None


def prepare(strategy_id,request):
    """Revalidate stored strategy and freeze a fresh, fully evidenced experiment."""
    from ..cli import create_experiment
    bundle=Store(request['strategy_root']).load(strategy_id);s=bundle['strategy'];current=_strategy(request)
    if s['provenance'].get('implementation_sha256')!=current['provenance']['implementation_sha256']:
        raise WorkflowError('implementation_changed','Restore recorded code or create a new strategy revision')
    if s['context']!=current['context'] or any(s[k]!=current[k] for k in ('bindings','phases','policy','command_pose_offset','source_frame_count','adapter')):
        raise WorkflowError('context_changed','Create a new strategy revision and revalidate')
    geometry=_geometry(request,s,s['proposal'].get('decision'))
    selected=next((r for r in geometry['candidates'] if r['id']==s['proposal']['stance']['id'] and r['feasible']),None)
    if selected is None:raise WorkflowError('preflight_failed','Saved stance no longer feasible')
    files=request['files'];template=json.loads(Path(files['record']).read_text())
    template.setdefault('inputs',{})
    # Historical physical evidence may be reused only after verifying every bound file.
    from .launch_evidence import recover_template_evidence
    archived_hashes,source_recovery=recover_template_evidence(template,REPO)
    template['preflight']['evidence_sha256']=archived_hashes
    root=create_experiment(request['experiment_root'],'workflow_native');started=now()
    write_new(root/'planning_started.json',{'started_at':started});write_new(root/'request.json',request)
    try:
        write_new(root/'baseline_source_recovery.json',{'recovered':source_recovery,'historical_contract_unchanged':True})
        for name in ('model','initial','targets','reference_states'):
            dest=root/(name+Path(files[name]).suffix);shutil.copyfile(files[name],dest);template['inputs'][name]=str(dest)
        task=json.loads(Path(files['task']).read_text())
        # Physical state is passed separately; preserve the genuine SONIC history resource.
        write_new(root/'task.json',task);template['inputs']['task']=str(root/'task.json')
        from .native import NativeAdapter,validate_physics,validate_model_identity,sonic_files
        import mujoco
        from ..privileged import no_physics
        with no_physics():
            native=NativeAdapter(task,s,np.load(template['inputs']['initial']))
            try:
                validate_model_identity(mujoco.MjModel.from_binary_path(template['inputs']['model']),native.m)
                validate_physics(native.m,json.loads(Path(template['physics']['physics_parameters_manifest']).read_text()),sonic=native.sonic,reward_stride=native.b.reward_stride)
                if not np.isclose(native.dt,json.loads(Path(files['alignment']).read_text())['command_dt_s']):raise WorkflowError('unsupported_capability','Source resampling required')
                write_new(root/'native_preflight.json',{'physics_steps':0,'actual_session_providers':native.providers,'executed_model_sha256':native.model_files,'planning_execution_model_identical':True,'initial_observation':native.observe(),
                    'model_nq':native.m.nq,'model_nv':native.m.nv,'model_nu':native.m.nu,'physics_dt_s':native.dt,'control_stride':native.control_stride,'bindings_verified':True,'physics_matches_manifest':True,
                    'physical_initial_state':template['inputs']['initial'],'controller_history_seed':task['resources']['initial_state'],
                    'controller_history_seed_sha256':file_hash(task['resources']['initial_state'])})
            finally:native.close()
        alignment=json.loads(Path(files['alignment']).read_text());write_new(root/'strategy.json',s);write_new(root/'stance_selection.json',{'selected':selected,'candidates':geometry['candidates']})
        with (root/'runtime.log').open('w') as log:
            subprocess.run([str(REPO/'scripts/humano_python.sh'),str(REPO/'scripts/check_humano_runtime.py'),'--load-models','--models',str(Path(task['resources']['sonic_models']).resolve()),'--out',str(root/'runtime.json')],check=True,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
        policy=s['policy'];template.update(experiment_id=root.name,created_at=started,record_status='frozen_ready')
        template['purpose']={'hypothesis':'Restore evidenced closed-loop wrist correction and preset stance; stable base initiates grasp without wrist-error veto; bounded physical retreat/retry if contact fails',
            'baseline_paths':[str(Path(files['record']).parent)],'planned_changes':['Independent workflow; named phases; checked preset stance if supplied; compensation mode '+policy['compensation']['mode'],'No pregrasp wrist-error gate; base stability only','Bounded physical recovery from measured state, without simulator resets'],
            'invariants':['No changes to other pipelines','No original-token prefix or implicit fallback','No physical state writes after reset'],
            'known_confounders':['Not a matched controller comparison; new strategy/stance/approach timing','Single-condition execution does not establish generality']}
        if policy['navigation'].get('profile')=='replay_navigation':
            template['purpose'].update(hypothesis='Replay-relative navigation reaches a measured stable stance with a checked manipulation reference; no manipulation is executed',
                planned_changes=['Deterministic task-frame stance and bounded route control','Full-reference kinematic handoff screening; navigation-only acceptance'],
                known_confounders=['Navigation completion is not task completion','Geometric support is not a dynamic guarantee','Scoped stopping calibration required'])
        if 'study_case' in files:
            study=json.loads(Path(files['study_case']).read_text())
            if set(study['purpose'])!=set(template['purpose']):raise WorkflowError('invalid_input','Study purpose fields')
            template['purpose']=copy.deepcopy(study['purpose'])
        template['workflow']={'schema_version':1,'strategy_id':strategy_id,'strategy':s,'selection':selected,'alignment':alignment}
        template['inputs']={k:template['inputs'][k] for k in ('model','initial','targets','reference_states','task')}
        template['runtime']['actual_loaded_session_evidence']=str(root/'runtime.json')
        template['cases']=[dict(task=task['name'],initial_state=template['inputs']['initial'],initial_sha256=file_hash(template['inputs']['initial']),navigation_goal_xyyaw=selected['xyyaw'])]
        template['physics']['asset_and_initial_state_manifest']={template['inputs'][key]:file_hash(template['inputs'][key]) for key in ('model','initial')}
        template['control']={'fallback_allowed':False,'actual_mode':'SONIC encoder1','prefix':'none','stage_interfaces':[
            {'stage':'navigation','mode':'SONIC encoder1 + live locomotion planner','fields':'world velocity m/s; facing yaw rad; body-relative initial wrist/torso hold'},
            {'stage':'settle','mode':'SONIC encoder1','fields':'source entry legs and roll/pitch reference, measured root XYZ/yaw; physically measured support/motion gate'},
            {'stage':'approach','mode':'SONIC encoder1','fields':'measured-state IK/RRT world wrist/torso reference only; explicit timing limits'},
            {'stage':'contact_phases','mode':'SONIC encoder1','fields':'source world poses with declared command alignment; original finger commands; declared measured feedback'}],
            'settings_and_origins':policy}
        template['control']['postapproach_pose_gate_enabled']=False
        template['control']['recovery_interfaces']={'open':'SONIC mode1 measured wrist hold + source open finger command','retreat':'SONIC mode1 IK/RRT reference from actual state','retry':'measured object retargeting, base settle, source command restart; controller and physical state continuous'}
        if policy['navigation'].get('profile')=='distance_velocity':
            template['control']['stage_interfaces'][0]['fields']+='; distance-driven acceleration-limited velocity; C2 blend of walking leg/root orientation references to standing at arrival; measured stability required before approach'
        if policy['navigation'].get('standing_orientation')=='upright':
            template['control']['stage_interfaces'][1]['fields']='source entry legs, world-up roll/pitch, measured root XYZ/yaw; physically measured support/motion gate'
        if policy.get('recovery',{}).get('posture_restore',False):
            template['control']['recovery_interfaces']['retreat']='SONIC mode1 FK of collision-checked arms+waist joint path to first stable normal posture in current root frame; measured joint/wrist/torso convergence before retry'
        if 'parking' in policy['navigation'] or 'correction' in policy['navigation']:
            template['control']['parking']='Declared measured stopping predictor and bounded post-settle correction; walk uses checked route; planted_feet uses newly planned leg IK with measured world foot holds and COM support screening, executed only through SONIC encoder1; no GPT or state reset'
        if 'grasp_trigger' in policy:
            template['control']['grasp_trigger']={'policy':policy['grasp_trigger'],'observations':'privileged object/wrist/digit geometry, contacts, base stability; source reference geometry','actions':['continue','close_now','abort'],'execution':'Open fingers during approach/probe. On GPT close_now replace grasp wrist motion by measured hold, replay finger closure, then adapted relative lift. Retain integral feedback and physical gates. Model wait pauses physics; logged separately. No physical reset.'}
        template['budget']={'attempts_per_case':1,'recovery_attempts':policy.get('recovery',{}).get('max_retries',0),'simulation_seconds_per_attempt':policy['budgets']['simulation_s'],'stage_budgets':policy['budgets']}
        template['physics']['safety_gates']=policy['gates'];template['physics']['acceptance_thresholds']=policy['acceptance']
        if policy['navigation'].get('profile')=='replay_navigation':
            template['physics']['acceptance_thresholds']={'scope':'navigation_handoff_only','arrival':policy['navigation']['route_limits'],
                'settle':policy['settle'],'handoff':'full reference kinematic check from measured stable pose','task_success':None}
            template['control']['stage_interfaces']=template['control']['stage_interfaces'][:2]
            template['control']['stage_interfaces'][0]['fields']='SONIC live mode1; bounded world XY velocity and world yaw; calibrated envelope; no token prefix'
            template['control']['stage_interfaces'][1]['fields']='Blend to initial standing articulation anchored at measured root; measured support and stability; geometry check; stop before manipulation'
        if policy['navigation'].get('profile')=='replay_navigation' and policy['navigation']['navigation_only'] is False:
            template['purpose']['hypothesis']='Measured whole-body alignment and contact-gated reference progression enable operation after navigation; physical efficacy unverified'
            template['purpose']['planned_changes']=['Calibrated navigation followed by continuous whole-body reference retargeting and measured phase gates']
            template['control']['postapproach_pose_gate_enabled']=True
            template['control']['stage_interfaces'][1]['fields']='Blend to initial standing articulation anchored at measured root; measured support and stability; full reference geometry recheck'
            template['control']['stage_interfaces'].append({'stage':'whole_body_operation','mode':'SONIC encoder1',
                'fields':'Whole-body reference FK wrists/torso and matching root/leg reference; checked quintic measured-state entry; source fingers follow gated reference clock; no live state/history reset'})
            template['control']['recovery_interfaces']={}
            template['physics']['acceptance_thresholds']={'scope':'whole_body_operation_native','native_task_validator':True,
                'trajectory_completion_required':True,'entry_and_contact_gates':policy['whole_body_operation']}
        admission=s['provenance'].get('navigation_admission')
        if admission:
            template['control']['navigation_admission']=admission
            template['pipeline']['navigation_execution']=admission
            template['pipeline']['original_actions_visible']=True
            if request['decision']['mode']=='deterministic':template['pipeline']['external_model_calls']={'before_launch':0,'during_physics':0}
            template['pipeline']['planner_identity_and_prompt_hash']={'identity':'Deterministic replay-relative geometry and explicit diagnostic admission','strategy_id':strategy_id}
            template['control']['action_recording']={'body':'Actual SONIC tokens at every policy tick','hands':'Separate14-joint commands at every physical step','bundle':'execution/sonic_actions.npz'}
            template['purpose']['known_confounders'].append('Explicit diagnostic execution; stopping calibration unvalidated' if not admission['calibrated'] else 'Stopping calibration verified for bound scope')
            if admission['diagnostic_only']:
                for stage in template['control']['stage_interfaces']:stage['fields']=stage['fields'].replace('calibrated envelope','configured envelope; stopping fit unvalidated')
                template['purpose']['known_confounders']=[x.replace('Scoped stopping calibration required','Explicit diagnostic; stopping calibration remains unvalidated') for x in template['purpose']['known_confounders']]
                template['purpose']['planned_changes']=[x.replace('Calibrated navigation','Uncalibrated diagnostic navigation') for x in template['purpose']['planned_changes']]
        template['physics']['control_dt_s']=policy['control_dt_s']
        if policy['navigation'].get('profile')=='replay_navigation':
            bound_navigation=json.loads(Path(files['navigation_plan']).read_text())
            if bound_navigation.get('reference_mode')=='apple_lateral_v1':
                template['control']['generated_operation_reference']={
                    'method':'trajectory_retargeting','planner_access_to_original_actions':True,'original_token_steps':0,
                    'source_control':bound_navigation['prior']['source_control'],
                    'handoff':'One bounded measured-foot IK regeneration at a recorded simulation pause; verify fresh state before resuming',
                    'derived_output':'execution/measured_operation_reference','options':bound_navigation['lateral_options'],
                    'failure':'Fail closed on pose/contact/object/timeout mismatch; no fallback or live state writes'}
        template['physics']['acceptance_source_and_hash']['independent_validator']=str(Path(__file__).with_name('verification.py'))
        template['pipeline']['entrypoint']='python -m humanoid_mimicgen.workflow execute --experiment '+str(root)
        template['pipeline']['feedback_and_reflection_rules']='Only declared bounded physical retries; save checkpoint and reflection before retry; no state reset or API override of frozen contract'
        write_new(root/'preflight_evidence.json',{'physics_steps':0,'fresh_geometry':geometry,'strategy_id':strategy_id,'actual_sessions':'runtime.json','template_dependencies_verified':True})
        write_new(root/'comparison_diff.json',{'baseline':str(Path(files['record']).parent),'changes':template['purpose']['planned_changes'],'not_a_matched_method_comparison':True})
        hashes=dict(template['preflight']['evidence_sha256']);hashes.update(sonic_files(task));
        seed_path=str(Path(task['resources']['initial_state']).resolve());hashes[seed_path]=file_hash(seed_path)
        if policy['navigation'].get('profile')=='replay_navigation':
            nav_path=Path(files['navigation_plan']).resolve();hashes[str(nav_path)]=file_hash(nav_path)
            hashes.update(json.loads(nav_path.read_text())['evidence_sha256'])
            proof_path=Path(files[policy['navigation']['control_evidence_file']]).resolve()
            hashes[str(proof_path)]=file_hash(proof_path)
            if s['provenance'].get('navigation_admission',{}).get('calibrated',True):
                hashes.update(json.loads(proof_path.read_text())['evidence_sha256'])
        snap=root/'source_snapshot';snap.mkdir()
        for source in sorted((REPO/'humanoid_mimicgen').rglob('*.py')):
            hashes[str(source)]=file_hash(source);dest=snap/source.relative_to(REPO/'humanoid_mimicgen');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        for source in root.iterdir():
            if source.is_file():hashes[str(source)]=file_hash(source)
        hashes[str(Path(files['record']).resolve())]=file_hash(files['record'])
        hashes[str(bundle['path']/'manifest.json')]=file_hash(bundle['path']/'manifest.json')
        template['preflight'].update(passed=True,evidence_paths=list(hashes),evidence_sha256=hashes,comparison_diff_path=str(root/'comparison_diff.json'))
        template['pipeline']['source_snapshot_and_hashes']=hashes
        template['pipeline']['git_commit_and_dirty_patch']={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source_snapshot':str(snap),'source_sha256':hashes}
        write_new(root/'experiment.json',template);(root/'experiment.sha256').write_text(file_hash(root/'experiment.json')+'\n')
        write_new(root/'freeze.json',{'frozen_at':now(),'sha256':file_hash(root/'experiment.json')});check_contract(root/'experiment.json')
        return {'status':'prepared','experiment_id':root.name,'experiment_path':str(root),'strategy_id':strategy_id}
    except Exception as exc:
        write_new(root/'preflight_failure.json',{'ended_at':now(),'reason_code':exc.code if isinstance(exc,WorkflowError) else 'preflight_failed','error_type':type(exc).__name__});raise


def inspect(experiment):
    root=Path(experiment)
    if (root/'result.json').exists():return json.loads((root/'result.json').read_text())
    if (root/'execution/execution_audit.json').exists():return {'status':'awaiting_verification','audit':json.loads((root/'execution/execution_audit.json').read_text())}
    if (root/'execution').exists():return {'status':'execution_artifacts_present','liveness':'unknown','experiment_id':root.name}
    if (root/'experiment.sha256').exists():return {'status':'prepared','experiment_id':root.name}
    return {'status':'unprepared','experiment_id':root.name}


def execute(experiment):
    from .runner import execute_native
    from .verification import verify
    root=Path(experiment);contract=check_contract(root/'experiment.json')
    if contract['workflow']['strategy']['policy']['navigation'].get('profile')=='replay_navigation':
        from .navigation.integration import execute_plan
        return execute_plan(contract['workflow']['strategy']['policy']['navigation']['plan_file'],root)
    execute_native(root/'experiment.json');return verify(root)
