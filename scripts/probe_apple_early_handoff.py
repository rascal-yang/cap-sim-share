"""No-API matched diagnostic: change only original-token/mode1 handoff time."""
import argparse,copy,json,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from humanoid_mimicgen.augmentation import wrist_position_ablation as w
from humanoid_mimicgen.augmentation import generalization_runtime as rt
from humanoid_mimicgen.augmentation import rollback_benchmark as rb
from humanoid_mimicgen.augmentation import suite_physics as ph
from humanoid_mimicgen.augmentation.suite_evidence import install
from humanoid_mimicgen.workflow.contracts import now,file_hash
from humanoid_mimicgen.augmentation.agent import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--handoff',type=int,default=300);p.add_argument('--conditions',nargs='+',default=['apple_03','apple_05','apple_09']);a=p.parse_args()
 cfg=json.loads((a.source/'config.json').read_text());stamp=datetime.now(ZoneInfo('Asia/Shanghai'));parent=a.source.parents[2];root=parent/stamp.strftime('%Y-%m/%d/%Y%m%d_%H%M_apple_early_handoff');i=1;stem=root
 while root.exists():root=stem.with_name(stem.name+f'_{i:02d}');i+=1
 root.mkdir(parents=True);Path('/tmp/early_handoff_root').write_text(str(root));items=[x for x in cfg['conditions'] if x['condition_id'] in a.conditions]
 assert len(items)==len(a.conditions) and 0<=a.handoff<340
 cfg.update(output_root=str(root),config_path=str(root/'config.json'),handoff_frame=a.handoff,conditions=items)
 protocol=root/'protocol.md';protocol.write_text('Fixed source hand commands (closure starts340), physical scene, initial state, wrist fixed Z offset active346..443, source leg/torso reference, historical integral gain/cap and native acceptance remain unchanged. Only handoff from original tokens to mode1 moves346 to '+str(a.handoff)+'. World XY scene adaptation and integral activate at the new handoff. This changes both arms reference encoding earlier; does not implement navigation planning. No API, no postdecoder changes, one fresh attempt per condition; replay successful tokens+hands. Compare prior feedback arm.\n');cfg['protocol_path']=str(protocol);save(root/'config.json',cfg)
 install();oldruntime=rb.runtime;rb.runtime=rt;oldprefix=rt.token_prefix_steps;oldwrite=ph.write_new
 def prefix(mode,contact_start,frames):return a.handoff if mode=='token_prefix_mode1' else oldprefix(mode,contact_start,frames)
 def write(path,value):
  if Path(path).name=='experiment.json':
   value=copy.deepcopy(value);value['purpose'].update(hypothesis='Does object-relative approach adaptation before finger closure rescue apple position transfer?',baseline_paths=[str(a.source)],planned_changes=[{'handoff_frame':a.handoff,'previous_handoff_frame':346}],known_confounders=['Earlier mode1 changes arm/body execution during approach; no navigation replanning','Reuses prior authenticated fixed plan; no new API'])
   value['pipeline'].update(entrypoint='scripts/probe_apple_early_handoff.py',external_model_calls='0; local session selected handoff frame',feedback_and_reflection_rules='One fresh attempt per case, no retuning within frozen case; successful token+hands replay')
   value['control']['right_wrist_position_feedback']['activation']='mode1 starts at '+str(a.handoff)+'; fixed reference Z correction still starts346'
   value['control']['handoff_ablation']={'original_token_prefix_frames':a.handoff,'contact_annotation_start':346,'closure_command_start':340,'body_postdecoder_changes':False,'navigation_planner_calls':0}
   f=str(Path(__file__).resolve());value['preflight']['evidence_paths'].append(f);value['preflight']['evidence_sha256'][f]=file_hash(f)
  return oldwrite(path,value)
 rt.token_prefix_steps=prefix;ph.write_new=write;results=[];start=now();tick=time.monotonic()
 try:
  for item in items:
   folder=root/item['condition_id'];row=w.execute(cfg,item,'feedback',folder/'early')
   prior=json.loads((a.source/item['condition_id']/'summary.json').read_text())['arms']['feedback'];rec={'condition_id':item['condition_id'],'prior_success':prior['success'],'trial':row}
   x=Path(prior['result']['experiment_path']);y=Path(row['result']['experiment_path'])
   with np.load(x/'execution/trajectory.npz') as u,np.load(y/'execution/trajectory.npz') as v:rec['prefix_exact']=all(np.array_equal(u[k][:a.handoff*4],v[k][:a.handoff*4]) for k in ('state','next_state','commands'))
   with np.load(x/'execution/hand_commands.npz') as u,np.load(y/'execution/hand_commands.npz') as v:rec['hands_exact']=all(np.array_equal(u[k][:len(v[k])],v[k]) for k in ('left','right'))
   if not rec['prefix_exact'] or not rec['hands_exact']:raise ValueError('Declared invariant violated')
   if row['success']:
    replay=w.execute(cfg,item,'feedback',folder/'token_replay',y);rec['replay']={'trial':replay,'proof':w.replay_audit(row['result'],replay['result'])}
    if not replay['success'] or not rec['replay']['proof']['exact']:raise ValueError('Token replay mismatch')
   rec['reflection']={'observation':'success' if row['success'] else 'failed','interpretation':'Only earlier handoff tested; do not equate this with measured-stance navigation','next':'Continue declared positions, then inspect failure geometry'}
   save(folder/'summary.json',rec);results.append(rec);save(root/'progress.json',results);print(json.dumps({'condition':item['condition_id'],'success':row['success'],'diagnostics':row['diagnostics']}),flush=True)
 finally:
  rt.token_prefix_steps=oldprefix;ph.write_new=oldwrite;rb.runtime=oldruntime;save(root/'summary.json',{'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'api_calls':0,'results':results})
if __name__=='__main__':main()
