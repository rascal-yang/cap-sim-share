"""Read only task configuration + initial state. No expert sequence is loaded."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,time,hashlib,inspect
from pathlib import Path
import mujoco,numpy as np
from humano.cli.render_sonic_isaac import _make_env,_reset_episode
root=Path(__file__).resolve().parent;out=root/'prepared';out.mkdir(exist_ok=True);report=[]
for folder in sorted((root/'dataset').glob('G1*')):
 t=time.time();dest=out/folder.name;dest.mkdir(exist_ok=True)
 try:
  meta=json.loads((folder/'sonic/episode_000000_replay.json').read_text());cfg=json.loads(meta['environment_config']);initial=np.load(folder/'sonic/episode_000000_start.npz')
  env=_make_env('humano/'+folder.name+'-v0','mujoco',True,200,None,'dex3');_reset_episode(env,cfg);e=env.unwrapped;m,d=e.mjModel,e.mjData
  if m.nq!=len(initial['qpos']):raise ValueError(f'Initial nq mismatch {m.nq} != {len(initial["qpos"])}')
  d.qpos[:]=initial['qpos'];d.qvel[:]=initial['qvel'];d.ctrl[:]=initial['ctrl'];d.time=0;mujoco.mj_forward(m,d)
  qids=m.jnt_qposadr[e.task.robot.body_joint_index];bodies=['left_wrist_yaw_link','right_wrist_yaw_link','torso_link']
  obj={k:{'position':v.xpos.tolist(),'quaternion':v.xquat.tolist()} for k,v in e.mujoco.mj_objects.items()}
  summary={'task':folder.name,'nq':m.nq,'nv':m.nv,'nu':m.nu,'root_pose':d.qpos[:7].tolist(),'objects':obj,'wrists':{name:{'position':d.body(name).xpos.tolist(),'quaternion':d.body(name).xquat.tolist()} for name in bodies},'initial_hand_positions':{'left':e.task.robot.prepare_obs()['left_hand_q'].tolist(),'right':e.task.robot.prepare_obs()['right_hand_q'].tolist()},'criteria':{k:v for k,v in vars(e.task).items() if any(w in k for w in ['threshold','criteria','hold_steps','max_m','height_m','max_rad']) and isinstance(v,(float,int,str,bool))},'source_file':inspect.getfile(type(e.task)),'wall_s':time.time()-t}
  summary['source_sha256']=hashlib.sha256(Path(summary['source_file']).read_bytes()).hexdigest()
  (dest/'scene.xml').write_text(e.mjSpec.to_xml());(dest/'environment_config.json').write_text(json.dumps(cfg,indent=2));(dest/'initial_observation.json').write_text(json.dumps(summary,indent=2));np.savez(dest/'initial.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy())
  renderer=mujoco.Renderer(m,height=360,width=640);renderer.update_scene(d,camera='head_stereo_left');from PIL import Image;Image.fromarray(renderer.render()).save(dest/'initial_ego.png')
  cam=mujoco.MjvCamera();cam.lookat[:]=[-.25,0,.7];cam.distance=2.8;cam.azimuth=130;cam.elevation=-20;renderer.update_scene(d,camera=cam);Image.fromarray(renderer.render()).save(dest/'initial_external.png');renderer.close();env.close()
  report.append({'task':folder.name,'prepared':True,'wall_s':time.time()-t,'criteria':summary['criteria']});print(report[-1],flush=True)
 except Exception as exc:
  import traceback;traceback.print_exc();report.append({'task':folder.name,'prepared':False,'error':repr(exc),'wall_s':time.time()-t})
 (root/'prepare_report.json').write_text(json.dumps(report,indent=2))
