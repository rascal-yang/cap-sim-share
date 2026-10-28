#!/usr/bin/env python3
"""Read saved native/contact metrics; do not reinterpret acceptance or step physics."""
import argparse,json
from pathlib import Path
from generalization_records import roots as study_roots, conditions as study_conditions
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);out=[]
for s in study_conditions(root):
    records=[]
    candidates=[('seed',s['seed'])] if s.get('seed') else []
    for arm,record in s['arms'].items():
        candidates += [(arm,r) for r in record['rows'] if not r.get('shared_seed')]
    for arm,r in candidates:
        case=Path(r['result']['experiment_path']);ms=json.loads((case/'execution/metrics.json').read_text())
        info={'arm':arm,'candidate_index':r['candidate_index'],'case':str(case),'success':r['success'],'native_success':r['audit']['native_success'],'failure':r['result']['failure'],'coverage':r['result']['coverage'],'wrist_error_rmse_m':r['result'].get('wrist_error_rmse_m'),'max_robot_table_N':max(m['robot_table_N'] for m in ms),'max_penetration_m':max(m['max_penetration_m'] for m in ms),'max_hand_object_N':max(m['hand_object_N'] for m in ms),'final_hand_object_N':ms[-1]['hand_object_N']}
        if s['task_label'] in ('apple','plate'):
            z=[m['object_xyz']['target'][2] for m in ms];info['max_target_lift_m']=max(z)-z[0];info['final_target_lift_m']=z[-1]-z[0]
        if s['task_label']=='chair':
            info['max_sit_hold_ticks']=max(m['hold_counts'].get('_sit_hold_count',0) for m in ms)
            info['chair_contact_ticks']=sum(bool(m['_sit_metrics']['has_chair_contact']) for m in ms)
            info['final_sit_metrics']=ms[-1]['_sit_metrics']
        if s['task_label']=='button':info['approved_button']=r['audit'].get('button')
        records.append(info)
    out.append({'condition_id':s['condition_id'],'shift':s['shift'],'task_label':s['task_label'],'status':s['status'],'attempts':records})
(root/'DIAGNOSTICS.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print('conditions',len(out))
