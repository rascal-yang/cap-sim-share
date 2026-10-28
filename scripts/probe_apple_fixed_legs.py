"""One-variable diagnostic on early handoff: fixed measured stance leg references."""
import copy,runpy,sys
from pathlib import Path
import numpy as np
from humanoid_mimicgen.augmentation import suite_physics as ph
from humanoid_mimicgen.workflow.contracts import file_hash
old_encode=ph.encode_reference;old_write=ph.write_new;anchor=None

def encode(s,d,q,v,poses,index,shift,pose_frame=None):
 global anchor
 if anchor is None:anchor=d.qpos[s.qids[:12]].copy()
 qq=q.copy();vv=v.copy();qq[:,s.qids[:12]]=anchor;vv[:,s.vids[:12]]=0
 return old_encode(s,d,qq,vv,poses,index,shift,pose_frame)

def write(path,value):
 global anchor
 if Path(path).name=='experiment.json':
  anchor=None;value=copy.deepcopy(value)
  value['purpose'].update(hypothesis='Do continuing source leg references interfere with shifted apple grasp after early mode1 handoff?',planned_changes=['Compared with early handoff300: hold the12 measured leg joints at first mode1 frame and zero leg reference velocities; all upper references, original hands, feedback and timing unchanged'],known_confounders=['This is fixed measured leg input to SONIC, not direct joint pinning; actual balance can drift','Source root orientation reference and torso remain unchanged','Not a navigation planner replacement'])
  value['control']['fixed_measured_leg_reference']={'enabled':value['control']['stage_interfaces'][0]['mode']!='saved_token_replay','anchor':'actual12 leg joint positions at first mode1 frame','velocity':'zero','postdecoder_correction':False}
  value['pipeline']['entrypoint']='scripts/probe_apple_fixed_legs.py -> probe_apple_early_handoff.py'
  f=str(Path(__file__).resolve());value['preflight']['evidence_paths'].append(f);value['preflight']['evidence_sha256'][f]=file_hash(f)
 if Path(path).name in ('result.json','execution_audit.json'):value.update(fixed_leg_anchor_rad=anchor.tolist() if anchor is not None else None)
 return old_write(path,value)
ph.encode_reference=encode;ph.write_new=write
try:runpy.run_path(str(Path(__file__).with_name('probe_apple_early_handoff.py')),run_name='__main__')
finally:ph.encode_reference=old_encode;ph.write_new=old_write
