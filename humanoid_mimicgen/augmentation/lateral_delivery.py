"""Independent saved-state native verification and indexed diagnostic videos."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,time,shutil,subprocess,collections
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_mimicgen.privileged import no_physics
from humanoid_mimicgen.workflow.contracts import file_hash,now

def native_scene(case):
 """Load only the independent scene/checker; numerical verification needs no policy."""
 from types import SimpleNamespace
 from cap_sim.backends.humano import Backend
 from humanoid_mimicgen.action_replay import initialize_source
 from humanoid_mimicgen.workflow.native import validate_model_identity
 import mujoco
 case=Path(case);b=Backend()
 try:
  b.describe(json.loads((case/'task.json').read_text()))
  validate_model_identity(mujoco.MjModel.from_binary_path(str(case/'scene.mjb')),b.m)
  with np.load(case/'initial.npz') as z:initialize_source(b.m,b.d,{k:z[k].copy() for k in z.files})
  task=b.e.task
  for key in vars(task):
   if key.startswith('_init_'):setattr(task,key,None)
   if key.endswith('_hold_count'):setattr(task,key,0)
  def native_success():
   info=b.e._get_info();task.compute_reward(info,mujoco_env=b.e.mujoco)
   return bool(task.check_success(info,mujoco_env=b.e.mujoco))
  native_success()
  return SimpleNamespace(m=b.m,d=b.d,native_success=native_success,close=b.close)
 except Exception:
  b.close();raise


def audit(c,scene_only=False,output_path=None):
 c=Path(c)
 z=np.load(c/'execution/trajectory.npz');st=z['state'];ns=z['next_state'];cmd=z['commands'];co=json.loads((c/'execution/controller.json').read_text());obs=json.loads((c/'execution/observations.json').read_text());controls=json.loads((c/'execution/commands.json').read_text());res=json.loads((c/'result.json').read_text());cfg=json.loads((c/'config.json').read_text())
 assert len(st)==len(ns)==len(obs)==len(controls)==len(cmd)==res['physics_steps']
 assert np.array_equal(ns[:-1],st[1:]);assert np.allclose(ns[:,0]-st[:,0],.005,atol=1e-12)
 targets=np.asarray([x['target'] for x in co]);assert np.array_equal(cmd[:,1:30],np.repeat(targets,4,axis=0)[:len(cmd)])
 assert set(x['mode'] for x in co)=={1}
 from . import direct_grasp as mod
 with no_physics():
  if scene_only:a=native_scene(c)
  else:a,_=mod.initialize(c)
  native=False;first_native=None
  for i,s in enumerate(ns):
   if (i+1)%4:continue
   a.d.time=s[0];a.d.qpos[:]=s[1:1+a.m.nq];a.d.qvel[:]=s[1+a.m.nq:];mod.mujoco.mj_forward(a.m,a.d)
   ok=a.native_success();native=native or ok
   if ok and first_native is None:first_native=float(s[0])
  a.close()
 obj=np.array([o['object_pose'] for o in obs]);wr=np.array([o['right_pose'] for o in obs]);rr=Rotation.from_quat(np.roll(wr[:,3:],-1,axis=1));rel=rr.inv().apply(obj[:,:3]-wr[:,:3]);orot=rr.inv()*Rotation.from_quat(np.roll(obj[:,3:],-1,axis=1));vel=np.gradient(obj[:,:3],.005,axis=0);n=len(obs)
 # A diagnostic final .5s window, separate from frozen native lift and collision predicates.
 ids=np.arange(max(0,n-100),n);slip=float(np.max(np.linalg.norm(rel[ids]-rel[ids[0]],axis=1)));angular_slip=float((orot[ids[0]].inv()*orot[ids]).magnitude().max());speed=float(np.max(np.linalg.norm(vel[ids],axis=1)));bottom=min(obs[i]['clearance_m'] for i in ids)
 supported=all(obs[i]['contacts']['hand_object_N']>.1 and obs[i]['contacts']['support_N']<.1 for i in ids)
 feet={side:dict(max_xy_drift_m=float(np.max(np.linalg.norm(np.asarray([o['feet'][side][:2] for o in obs])-obs[0]['feet'][side][:2],axis=1))),max_rise_m=float(max(o['feet'][side][2]-obs[0]['feet'][side][2] for o in obs))) for side in ['left','right']}
 out=dict(audited_at=now(),physics_steps_in_verifier=0,frame_count=n,contiguous_substeps=True,decoder_targets_unmodified=True,all_encoder_mode1=True,no_token_prefix=True,contract_sha_valid=file_hash(c/'experiment.json')==(c/'experiment.sha256').read_text(),independent_native_success=bool(native),first_native_time_s=first_native,final_half_second=dict(min_clearance_m=bottom,hand_contact_without_support=supported,max_object_speed_m_s=speed,relative_position_slip_m=slip,relative_angle_slip_rad=angular_slip,stable_diagnostic_pass=bool(bottom>=.03 and supported and speed<=.04 and slip<=.02 and angular_slip<=.25)),feet=feet,recorded_native_matches=bool(native)==res['native_success'])

 assert out['recorded_native_matches'];Path(output_path or c/'independent_audit.json').write_text(json.dumps(out,indent=2));return out


def render(c):
 from .suite_inspect import inspect_case,FONT
 c=Path(c);started=now();tick=time.monotonic();root=c.parent
 inspect_case(c,c/'inspection')
 result=json.loads((c/'result.json').read_text());observations=json.loads((c/'execution/observations.json').read_text())
 pairs=collections.defaultdict(list)
 for o in observations:
  for event in o['contacts']['events']:
   if 'table' in event['bodies'] and any('hand' in b or 'wrist' in b for b in event['bodies']):
    pairs[' / '.join(event['bodies'])].append(dict(time_s=o['time_s'],source_row=o['source_row'],force_N=event['force_N']))
 diagnosis={pair:dict(samples=len(values),duration_s=len(values)*.005,first=values[0],peak=max(values,key=lambda v:v['force_N'])) for pair,values in pairs.items()}
 (c/'contact_diagnosis.json').write_text(json.dumps(diagnosis,indent=2))
 caption=c/'inspection/api_caption.txt'
 caption.write_text(f'{root.name} / {c.name}\nAPI选择并调用工具 | 原生抬起：{result["native_success"]} | 碰桌质量通过：{result["success"]}\n操作站位初始化；无行走/原token前缀；SONIC mode1；保存状态重渲染；非训练视频')
 original=c/'inspection/review_zh.mp4';corrected=c/'inspection/api_review.mp4'
 subprocess.run(['ffmpeg','-v','error','-i',str(original),'-vf',f'drawbox=x=0:y=0:w=iw:h=85:color=black:t=fill,drawtext=fontfile={FONT}:textfile={caption}:fontcolor=white:fontsize=22:x=12:y=4','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(corrected)],check=True)
 dest=root/(c.name[:13]+'_苹果_API抓取_'+c.name.rsplit('_',1)[-1]+'.mp4')
 if dest.exists():raise FileExistsError(dest)
 shutil.copy2(corrected,dest);assert file_hash(corrected)==file_hash(dest)
 subprocess.run(['ffmpeg','-v','error','-i',str(dest),'-f','null','-'],check=True)
 video=dict(experiment_id=c.name,path=str(dest),source=str(corrected),original=str(original),sha256=file_hash(dest),copy_hash_verified=True,full_decode_passed=True,training_ready=False)
 (c/'video.json').write_text(json.dumps(video,ensure_ascii=False,indent=2))
 (c/'render_timing.json').write_text(json.dumps(dict(started_at=started,ended_at=now(),wall_s=time.monotonic()-tick,physics_steps=0),indent=2))
 (c/'inspection_label_correction.json').write_text(json.dumps(dict(original_preserved=True,reason='Shared inspector labels combined result.success as native result; corrected caption explicitly separates native and collision quality',native_success=result['native_success'],clean_success=result['success'],corrected=str(corrected)),indent=2))
 return video


if __name__=='__main__':
 print(json.dumps(audit(sys.argv[2],scene_only=sys.argv[1]=='audit_numeric') if sys.argv[1] in ('audit','audit_numeric') else render(sys.argv[2]),ensure_ascii=False))
