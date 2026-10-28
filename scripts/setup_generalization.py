import json,copy,argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--apple-root',type=Path,required=True);p.add_argument('--plate-case',type=Path,required=True);a=p.parse_args()
base=a.workspace.resolve();old=a.baseline_root.resolve()
now=datetime.now(ZoneInfo('Asia/Shanghai'));root=base/'experiments'/now.strftime('%Y-%m/%d/%Y%m%d_%H%M_four_task_positions')
stem=root;sequence=1
while root.exists():
 root=stem.with_name(stem.name+f'_{sequence:02d}');sequence+=1
root.mkdir(parents=True)
read=lambda p:json.loads(p.read_text())
write=lambda p,v:p.write_text(json.dumps(v,indent=2,ensure_ascii=False))
specs={k:read(old/f'{k}_spec.json') for k in ('chair','button','apple')}
for k in specs:
    source=(old if k!='apple' else a.apple_root.resolve())
    source=Path(source)/k/'verify_full/trial.json'
    trial=read(source);d=read(Path(trial['decision_path']))['decision'];specs[k]['base_plan']=d
    specs[k]['source_case']=trial['result']['experiment_path']
p=a.plate_case.resolve()
r=read(p/'result.json');d=read(Path(r['decision_path']))['decision']
specs['plate']={'profile':'apple','task':r['task'],'dataset':str(base/'experiments/2026-10/13/20261013_cap_ten_tasks/dataset'/r['task']),'shift':[.005,0.],'base_plan':d,'source_case':str(p),'source_frames':r['source_frames'],'experience':{'native_success':True,'token_replay_success':True,'source_case':str(p),'limitation':'Native height gain >=9cm; does not certify sustained bilateral hold'},'constraints':{}}
from humanoid_mimicgen.augmentation.generalization_benchmark import offsets
paths=[]
# Plate first: missing checkpoint capability must be validated early.
for k in ('plate','chair','button','apple'):
    for i,offset in enumerate(offsets()):
        s=copy.deepcopy(specs[k]);s.update(task_label=k,condition_id=f'{k}_{i:02d}',position_index=i,offset=offset,shift=[round(.005+offset[0],6),offset[1]],candidate_count=12)
        s['base_plan']['resume']={'kind':'restart','case':'','frame':0,'rationale':'Fresh same-scene baseline'}
        s['constraints'].update(fixed_scene='only target translated at initialization; robot/table/physics fixed',queue_size=12,minimum_tail_s=max(0,round((s['base_plan']['contact_end']+1-s['source_frames'])*.02,2)))
        s.pop('preserved_candidate_parameters',None)
        p=root/(s['condition_id']+'_spec.json');write(p,s);paths.append(str(p))
c={'output_root':str(root),'config_path':str(root/'config.json'),'created_at':now.isoformat(),'spec_paths':paths,'benchmark':{'wall_budget_s':120,'order_repeats':1,'report_budgets_s':[30,60,90,120],'candidate_count_including_seed':12,'api_cost':'full actual per-condition planning wall charged to both arms','seed_cost':'same scene full seed cost charged equally','search_policy':'fixed API order; stop first in-budget success; same queue both arms','overrun_policy':'finish atomic attempt but exclude late completion','equivalence_validation':'separate validation overhead, same-scene boundary for all40 plus active controller for each task center','verification':'every observed successful execution including deadline misses: full restart and token+hands; compare all successful arm traces; exclude late completion from budget success','scene_scope':'10 predetermined object offsets per task; robot initial fixed; original-token approach prefix retained','rendering':'separate measured cost','success':'native except button_palm_mesh_v1; native button also reported','checkpoint_unavailable':'explicit restart fallback when seed terminates before boundary','positions_relative_to_successful_5mm_scene':offsets()}}
write(root/'config.json',c);write(root/'provider.json',read(old/'provider.json'));print(root)
