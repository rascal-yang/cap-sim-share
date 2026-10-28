"""Explicit historical request reconstruction; no remote planner or source mutation."""
import argparse,copy,json,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from humanoid_mimicgen.workflow import service
from humanoid_mimicgen.workflow.contracts import file_hash,now
BASE=Path('/home/kinetra-4090-01/code/zijian')
REPO=Path(__file__).resolve().parents[1]

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False))
def resolve_tree(obj,mapping):
 if isinstance(obj,str):
  for a,b in sorted(mapping.items(),key=lambda x:-len(x[0])):
   if obj==a or obj.startswith(a+'/'):return b+obj[len(a):]
  return obj
 if isinstance(obj,dict):return {k:resolve_tree(v,mapping) for k,v in obj.items()}
 if isinstance(obj,list):return [resolve_tree(v,mapping) for v in obj]
 return obj

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--variant',choices=['restore','smooth','no_feedback'],default='restore');a=p.parse_args()
 stamp=datetime.now(ZoneInfo('Asia/Shanghai'))
 root=a.root or BASE/'experiments'/stamp.strftime('%Y-%m/%d/%Y%m%d_%H%M_apple_navigation_forensics')
 root.mkdir(parents=True,exist_ok=True);case=root/a.variant;case.mkdir(exist_ok=False);start=now();tick=time.monotonic()
 idx=json.loads((BASE/'experiments/_index/latest-migration.json').read_text());mapping=json.loads(Path(idx['path_map']).read_text())
 origin=BASE/'experiments/2026-10/19/20261019_2050_workflow_native/request.json'
 req=resolve_tree(json.loads(origin.read_text()),mapping);old_record=Path(req['files']['record']);old=json.loads(old_record.read_text());template=resolve_tree(copy.deepcopy(old),mapping)
 # Preserve historical inputs byte-for-byte. Derive only path-bearing JSON files.
 checked={}
 for key in ('model','initial','reference_states','targets'):
  f=Path(req['files'][key]);expected=old['preflight']['evidence_sha256'].get(old['inputs'][key]);actual=file_hash(f)
  if actual!=expected:raise ValueError('Historical physical input differs: '+key)
  checked[str(f)]=actual
 taskpath=Path(req['files']['task']);save(case/'task.json',resolve_tree(json.loads(taskpath.read_text()),mapping));req['files']['task']=str(case/'task.json')
 # This is a new provenance manifest, not a rewrite or validation claim about all old code.
 checked[str(old_record)]=file_hash(old_record);checked[str(origin)]=file_hash(origin)
 checked[str(taskpath)]=file_hash(taskpath);checked[str(case/'task.json')]=file_hash(case/'task.json')
 physics=Path(template['physics']['physics_parameters_manifest']);checked[str(physics)]=file_hash(physics)
 checked[str(Path(__file__).resolve())]=file_hash(__file__)
 for name in ('grasp-prior-refinement','g1-grasp-runtime-review'):
  f=Path('/home/kinetra-4090-01/.codex/skills')/name/'SKILL.md';checked[str(f)]=file_hash(f)
 template['preflight']['evidence_sha256']=checked;template['preflight']['evidence_paths']=list(checked)
 template['pipeline']['external_model_calls']='none; current session chooses explicit diagnostic variants'
 template['pipeline']['planner_identity']='current Codex session; no external API'
 save(case/'derived_record.json',template);req['files']['record']=str(case/'derived_record.json')
 changes=['Path relocation only for restore; original physical inputs verified against historical hashes','Fresh provenance and current code; historical controller outcomes compared after execution']
 if a.variant=='smooth':
  req['strategy']['policy']['navigation'].update(profile='distance_velocity',acceleration_m_s2=.5,response_s=.6)
  changes+=['Navigation velocity envelope and smooth walking-to-standing blend; same stance and standing orientation']
 if a.variant=='no_feedback':
  req['strategy']['policy']['compensation']={'mode':'none'};changes+=['Disable only preencoder right-wrist integral through approach/contact']
 purpose={'hypothesis':'Can the historical navigation -> measured-state approach -> closed-loop contact success be reproduced, and how does navigation profile affect capture?', 'baseline_paths':[str(origin.parent)],'planned_changes':changes,'invariants':['historical scene, initial state, contact targets, source fingers and success criteria','all executed body control through SONIC encoder1; no postdecoder correction','No API calls'], 'known_confounders':['Historical trajectory is a recovered controller rollout, not current dataset measured reference','Cross-generation success percentages are not directly comparable; native and held-object outcomes reported separately','This diagnostic is not the pending historical ten-condition IK/mode0 vs teleop benchmark']}
 save(case/'study_case.json',{'purpose':purpose});req['files']['study_case']=str(case/'study_case.json');req['files']['probe_script']=str(Path(__file__).resolve())
 req['strategy_root']=str(case/'strategies');req['experiment_root']=str(case/'runs');req['decision']={'mode':'deterministic'}
 assert 'grasp_trigger' not in req['strategy']['policy']
 save(case/'request.json',req);save(root/'location.json',{'root':str(root),'created_at':start})
 result={'variant':a.variant,'started_at':start,'request':str(case/'request.json')}
 try:
  plan=service.plan(req);result['plan']=plan;prep=service.prepare(plan['strategy_id'],req);result['prepared']=prep
  result['execution']=service.execute(prep['experiment_path'])
  actual=np.load(Path(prep['experiment_path'])/'execution/states.npz')['samples'];expected=np.load(origin.parent/'execution/states.npz')['samples']
  result['historical_state_match']={'same_shape':actual.shape==expected.shape,'exact':actual.shape==expected.shape and np.array_equal(actual,expected),'max_abs_diff':float(np.max(abs(actual-expected))) if actual.shape==expected.shape else None}
 except Exception as exc:
  import traceback
  result['error']=repr(exc);result['traceback']=traceback.format_exc()
 finally:
  result.update(ended_at=now(),wall_s=time.monotonic()-tick,api_calls=0);save(case/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
