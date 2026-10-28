import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,time,hashlib,subprocess,sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,mujoco,cv2
r=Path(sys.argv[1]).resolve();out=r/'video';out.mkdir(exist_ok=False);start=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();tick=time.perf_counter()
contract=json.loads((r/'experiment.json').read_text());m=mujoco.MjModel.from_binary_path(contract['inputs']['model']);d=mujoco.MjData(m);s=np.load(r/'execution/states.npz')['samples'];init=np.load(r/'execution/actual_initial.npz');obs=json.loads((r/'execution/observations.json').read_text())
m.vis.global_.offwidth=800;m.vis.global_.offheight=600
renderer=mujoco.Renderer(m,height=600,width=800);cam=mujoco.MjvCamera();cam.lookat[:]=[-.4,0,.7];cam.distance=2.4;cam.azimuth=-65;cam.elevation=-20
# Source poststep frames at 50Hz, including exact final state; initialization shown once.
indices=[-1]+list(range(3,len(s),4))
if indices[-1]!=len(s)-1:indices.append(len(s)-1)
writers={name:cv2.VideoWriter(str(out/(name+'_raw.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),50,(800,600)) for name in ('external','ego')}
assert all(w.isOpened() for w in writers.values());mapping=[];sheets={name:[] for name in writers};selected=set(np.linspace(0,len(indices)-1,8).astype(int))
# Rendering is explicitly incapable of integrating physics.
def no_step(*a,**k):raise RuntimeError('Rendering must not integrate')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,no_step)
for k,i in enumerate(indices):
 if i==-1:d.qpos[:]=init['qpos'];d.qvel[:]=init['qvel'];d.ctrl[:]=init['ctrl'];d.time=0;stage='initial';clear=None
 else:
  row=s[i];d.time=float(row[0]);d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];d.ctrl[:]=row[-m.nu:];stage=obs[i]['stage'];clear=obs[i]['clearance_m']
 mujoco.mj_forward(m,d)
 for name,camera in [('external',cam),('ego','head_stereo_left')]:
  renderer.update_scene(d,camera=camera);frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);cv2.rectangle(frame,(0,0),(800,82),(15,15,15),-1)
  lines=[r.name,f'NEW CONTROLLER RUN | measured t={d.time:.3f}s | {stage}',f'physics frame {i} | clearance {clear*100:.2f} cm' if clear is not None else 'initial state before first action']
  for j,line in enumerate(lines):cv2.putText(frame,line,(8,20+j*25),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1,cv2.LINE_AA)
  writers[name].write(frame)
  if k in selected:sheets[name].append(cv2.resize(frame,(400,300)))
 mapping.append({'video_frame':k,'measured_poststep_index':i,'time_s':float(d.time),'stage':stage})
for name,w in writers.items():
 w.release();subprocess.run(['ffmpeg','-v','error','-i',str(out/(name+'_raw.mp4')),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/(name+'.mp4'))],check=True)
 cap=cv2.VideoCapture(str(out/(name+'.mp4')));count=0
 while True:
  ok,_=cap.read()
  if not ok:break
  count+=1
 cap.release();assert count==len(indices);cv2.imwrite(str(out/(name+'_contact_sheet.jpg')),np.vstack([np.hstack(sheets[name][:4]),np.hstack(sheets[name][4:])]))
renderer.close();(out/'frame_mapping.json').write_text(json.dumps(mapping,indent=2));(out/'audit.json').write_text(json.dumps({'experiment_id':r.name,'started_at':start,'ended_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'rendering_wall_s':time.perf_counter()-tick,'fps':50,'decoded_frames_per_view':len(indices),'source_measurement_sha256':hashlib.sha256((r/'execution/states.npz').read_bytes()).hexdigest(),'physics_steps_in_render':0,'includes_initial_and_final':True,'state_smoothing':False,'renderer':'MuJoCo EGL OpenGL','source':'new action-driven physics measurements, not original trajectory visualization','training_ready':False},indent=2));print('Rendered',r.name,len(indices),flush=True)
