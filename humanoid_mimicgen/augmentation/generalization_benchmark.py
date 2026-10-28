"""Forty frozen scene conditions, paired first-success search and replay proofs."""
import copy,json,time,traceback
from pathlib import Path
import numpy as np
from . import rollback_benchmark as rb
from . import generalization_runtime
from .agent import save
from .rollback_plan import patch_from_plan
from .contact_review import tools
from .quality_suite import api_decide
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_evidence import install
from .leg_fast_retry import exact_rollout
from .encoder_wrist_study import replay_audit
from ..workflow.contracts import now,file_hash

SKILL=Path(__file__).parent/'skills/scene_generalization/SKILL.md'

def offsets():
    return [[0.,0.],[.02,0.],[-.02,0.],[0.,.02],[0.,-.02],[.02,.02],[.02,-.02],[-.02,.02],[-.02,-.02],[.04,0.]]

def verify(config,spec,sp,bp,index,folder,winners):
    full=rb.execute(config,spec,sp,bp,index,folder/'full',rb.restart())
    matches=[{'case':w['result']['experiment_path'],'proof':exact_rollout(w['result']['experiment_path'],full['result']['experiment_path'])} for w in winners]
    rep=rb.execute(config,spec,sp,bp,index,folder/'token',rb.restart(),Path(full['result']['experiment_path']))
    proof=replay_audit(full['result'],rep['result'])
    with np.load(Path(full['result']['experiment_path'])/'execution/hand_commands.npz') as a,np.load(Path(rep['result']['experiment_path'])/'execution/hand_commands.npz') as b:
        proof['hands_exact']=all(np.array_equal(a[k],b[k]) for k in ('left','right'))
    return {'full':full,'token':rep,'branch_full':matches,'token_proof':proof,'passed':bool(full['success'] and rep['success'] and all(m['proof']['exact'] for m in matches) and proof['exact'] and proof['hands_exact'])}

def run(config,provider_config):
    install();rb.runtime=generalization_runtime
    root=Path(config['output_root']);start=now();tick=time.monotonic();results=[]
    save(root/'run_started.json',{'started_at':start,'config_sha256':file_hash(config['config_path'])})
    for ci,sp in enumerate(config['spec_paths']):
        spec=json.loads(Path(sp).read_text());folder=root/spec['condition_id'];summary=folder/'summary.json'
        if summary.exists():
            results.append(json.loads(summary.read_text()));continue
        if folder.exists():raise RuntimeError('Interrupted condition requires explicit recovery; refusing overwrite: '+str(folder))
        folder.mkdir();task={'condition_id':spec['condition_id'],'task':spec['task'],'task_label':spec['task_label'],'shift':spec['shift'],'started_at':now(),'status':'running','arms':{},'verification':{}};results.append(task)
        try:
            pr=folder/'planning';pr.mkdir();pt=time.monotonic()
            provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+900)
            context={'task':spec['task'],'controller_profile':spec['profile'],'world_object_shift_m':spec['shift'],'offset_from_successful_scene_m':spec['offset'],'baseline_patch':patch_from_plan(spec['profile'],spec['base_plan']),'known_experience':spec['experience'],'fixed_constraints':spec['constraints'],'candidate_count':spec['candidate_count'],'objective':'Scene-conditioned shared candidate sequence. Candidate0 exact baseline; rank plausible repairs next. Body reference already translates both wrists by full object shift; do not double-count translation.'}
            batch,bp=api_decide(provider,pr,context,[spec['base_plan']['episode']],tools=tools(rb.candidate_schema(spec['profile'],spec['candidate_count'])),validator=lambda d,eps:rb.validate_candidates(d,spec),skill_path=SKILL)
            task['planning_wall_s']=time.monotonic()-pt;task['api_batch_path']=str(bp)
            seed=rb.execute(config,spec,sp,bp,0,folder/'seed',rb.restart());task['seed']=seed
            safe=min(spec['base_plan']['contact_start'],spec['base_plan'].get('finger_control',{}).get('start_frame',10**9));case=Path(seed['result']['experiment_path'])
            task['safe_frame']=safe;shared=task['planning_wall_s']+seed['attempt_wall_s'];task['shared_cost_s']=shared
            # Every scene has its own seed; never reuse physical checkpoints across positions.
            if seed['result']['physics_steps']//4>safe:
                eq=rb.execute(config,spec,sp,bp,0,folder/'equivalence_boundary',rb.checkpoint(case,safe))
                proof=exact_rollout(case,eq['result']['experiment_path']);task['equivalence']={'trial':eq,'proof':proof}
                if not proof['exact']:raise ValueError('Boundary checkpoint failed exact equality')
                if spec['position_index']==0:
                    active=min(spec['base_plan']['contact_start']+40,seed['result']['physics_steps']//4-1)
                    eq2=rb.execute(config,spec,sp,bp,0,folder/'equivalence_active',rb.checkpoint(case,active))
                    proof2=exact_rollout(case,eq2['result']['experiment_path']);task['active_equivalence']={'trial':eq2,'proof':proof2}
                    if not proof2['exact']:raise ValueError('Active controller checkpoint failed exact equality')
                checkpoint_available=True
            else:checkpoint_available=False
            budget=config['benchmark']['wall_budget_s'];winners={}
            for arm in (['restart','rollback'] if ci%2==0 else ['rollback','restart']):
                clock=time.monotonic();rows=[dict(seed,elapsed_s=0.,shared_seed=True)]
                for index in range(1,spec['candidate_count']):
                    if rb.budget_metrics(rows,budget,shared)['success'] or time.monotonic()-clock+shared>=budget:break
                    resume=rb.checkpoint(case,safe) if arm=='rollback' and checkpoint_available else rb.restart()
                    row=rb.execute(config,spec,sp,bp,index,folder/arm/f'candidate_{index:02d}',resume)
                    row.update(elapsed_s=time.monotonic()-clock,shared_seed=False);rows.append(row)
                    save(folder/f'{arm}_progress.json',rows)
                metrics=rb.budget_metrics(rows,budget,shared)
                task['arms'][arm]={'rows':rows,'metrics':metrics,'retry_count':max(0,metrics['attempts']-1),'arm_wall_s':time.monotonic()-clock,'fallback_restart':arm=='rollback' and not checkpoint_available,'queue_exhausted':len(rows)==spec['candidate_count']}
                for row in rows:
                    if row['success'] and not row.get('shared_seed'):winners.setdefault(row['candidate_index'],[]).append(row)
            # Also verify seed successes that missed the operational planning deadline,
            # but do not count these as in-budget successes.
            if seed['success']:winners.setdefault(0,[seed])
            for index,winner in winners.items():
                task['verification'][str(index)]=verify(config,spec,sp,bp,index,folder/f'verify_{index:02d}',winner)
            for arm,record in task['arms'].items():
                eligible=[r for r in record['rows'] if r['success'] and r['elapsed_s']+shared<=budget]
                record['verified_success']=bool(eligible and task['verification'][str(eligible[0]['candidate_index'])]['passed'])
                record['metrics']['provisional']=True
            task['status']='complete'
            if any(not v['passed'] for v in task['verification'].values()):task['status']='verification_failed'
        except Exception as exc:
            task.update(status='error',error=str(exc),traceback=traceback.format_exc())
        task['ended_at']=now();save(summary,task);save(root/'progress.json',results)
        print(json.dumps({'condition':task['condition_id'],'status':task['status'],'arms':{k:v['metrics'] for k,v in task['arms'].items()},'error':task.get('error')},ensure_ascii=False),flush=True)
    out={'experiment_id':root.name,'started_at':start,'ended_at':now(),'total_wall_s':time.monotonic()-tick,'results':results,'training_ready':False};save(root/'summary.json',out);return out
