"""Move only apple and right contact reference; optionally translate navigation stance."""
import argparse,copy,json,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import mujoco
from humanoid_mimicgen.workflow import service
from humanoid_mimicgen.workflow.contracts import file_hash,now
from humanoid_mimicgen.augmentation.agent import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--historical-request',type=Path,required=True);p.add_argument('--current-root',type=Path,required=True);p.add_argument('--condition',required=True);p.add_argument('--stance',choices=['follow_object','fixed'],required=True);p.add_argument('--root',type=Path);a=p.parse_args();stamp=datetime.now(ZoneInfo('Asia/Shanghai'))
 root=a.root or a.current_root.parents[2]/stamp.strftime('%Y-%m/%d/%Y%m%d_%H%M_apple_navigation_transfer');root.mkdir(parents=True,exist_ok=True);Path('/tmp/nav_transfer_root').write_text(str(root));folder=root/(a.condition+'_'+a.stance);folder.mkdir(exist_ok=False);tick=time.monotonic();start=now()
 req=json.loads(a.historical_request.read_text());original_req=copy.deepcopy(req);files=req['files'];m=mujoco.MjModel.from_binary_path(files['model']);data=mujoco.MjData(m);initial=dict(np.load(files['initial']));original=initial['qpos'].copy();bid=m.body('apple_001').id;address=int(m.jnt_qposadr[m.body_jntadr[bid]])
 oldrow=json.loads((a.current_root/a.condition/'summary.json').read_text());case=Path(oldrow['arms']['baseline']['result']['experiment_path']);cm=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));cd=mujoco.MjData(cm);cd.qpos[:]=np.load(case/'initial.npz')['qpos'];mujoco.mj_forward(cm,cd);desired=cd.xpos[cm.body('apple_001').id].copy()
 data.qpos[:]=original;mujoco.mj_forward(m,data);oldpos=data.xpos[bid].copy();delta=desired-oldpos
 if abs(delta[2])>1e-8:raise ValueError('Unmatched object height; need separate declared study')
 initial['qpos'][address:address+2]+=delta[:2];np.savez_compressed(folder/'initial.npz',**initial)
 targets=dict(np.load(files['targets']));targets['right_pose'][:,:3]+=delta;np.savez_compressed(folder/'targets.npz',**targets)
 hint=json.loads(Path(files['stance_hint']).read_text());oldgoal=copy.deepcopy(hint['xyyaw'])
 if a.stance=='follow_object':hint['xyyaw'][:2]=(np.array(hint['xyyaw'][:2])+delta[:2]).tolist()
 hint.update(source=str(a.historical_request),reason='Explicit diagnostic: translate historical successful stance with object' if a.stance=='follow_object' else 'Explicit diagnostic: retain historical successful world stance')
 save(folder/'stance_hint.json',hint)
 geometry={'old_apple_xyz':oldpos.tolist(),'new_apple_xyz':desired.tolist(),'shift_m':delta.tolist(),'old_goal':oldgoal,'new_goal':hint['xyyaw'],'changed_qpos_indices':np.flatnonzero(initial['qpos']!=original).tolist(),'robot_initial_exact':np.array_equal(initial['qpos'][:address],original[:address]),'current_dataset_case':str(case),'reference_source':'historical successful recovered contact trajectory, not current raw dataset'}
 assert geometry['changed_qpos_indices']==[address,address+1] or set(geometry['changed_qpos_indices']).issubset({address,address+1})
 save(folder/'scene_change.json',geometry)
 record=json.loads(Path(files['record']).read_text());record['preflight']['evidence_sha256'].update({str(p):file_hash(p) for p in [a.historical_request,Path(__file__).resolve(),folder/'initial.npz',folder/'targets.npz',folder/'stance_hint.json',folder/'scene_change.json']});record['preflight']['evidence_paths']=list(record['preflight']['evidence_sha256']);save(folder/'derived_record.json',record)
 purpose={'hypothesis':'Does moving the stance with the apple preserve historical capture when compared with a fixed world stance?', 'baseline_paths':[str(a.historical_request),str(case)],'planned_changes':[geometry],'invariants':['same historical reference contact motion and fingers, initial robot, physics, native and hold acceptance','no API calls, no postdecoder correction; SONIC mode1 body control'],'known_confounders':['Cross-pipeline comparison is descriptive: historical source, approach timing and initial controller state differ from current fixed-prefix pipeline','Within follow_object vs fixed pairs, only navigation stance target changes; approach is replanned from achieved state','This is not the pending ten-condition historical IK/mode0 vs teleop benchmark']}
 save(folder/'study_case.json',{'purpose':purpose})
 for k in ('initial','targets','stance_hint','record','study_case'):files[k]=str(folder/({'record':'derived_record.json','initial':'initial.npz','targets':'targets.npz'}.get(k,k+'.json')))
 files['transfer_script']=str(Path(__file__).resolve());req['experiment_root']=str(folder/'runs');req['strategy_root']=str(folder/'strategies');req['decision']={'mode':'deterministic'};save(folder/'request.json',req)
 result={'condition_id':a.condition,'stance':a.stance,'started_at':start,'geometry':geometry}
 try:
  plan=service.plan(req);result['plan']=plan;prep=service.prepare(plan['strategy_id'],req);result['prepared']=prep;result['execution']=service.execute(prep['experiment_path'])
 except Exception as exc:
  import traceback
  result['error']=repr(exc);result['traceback']=traceback.format_exc()
 finally:
  result.update(ended_at=now(),wall_s=time.monotonic()-tick,api_calls=0);save(folder/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
