"""Offline rendering of recorded demo states; no physics advancement."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import subprocess
from pathlib import Path
import numpy as np
from .io import read,write


def render_demo(attempt,out):
    import mujoco
    attempt=Path(attempt);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str((attempt/'scene.xml').resolve()));d=mujoco.MjData(m);data=np.load(attempt/'states.npz')['samples']
    renderer=mujoco.Renderer(m,height=480,width=640);top=float(read(attempt.parent.parent/'task.json')['scene'].get('table_height',.65))
    ids=list(range(3,len(data),4))
    if not ids or ids[-1]!=len(data)-1:ids.append(len(data)-1)
    outputs=[]
    for name,azimuth,elevation in [('external',110,-28),('tool_view',90,-60)]:
        camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,top+.07];camera.distance=.9;camera.azimuth=azimuth;camera.elevation=elevation
        target=out/f'{name}.mp4';process=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','640x480','-framerate','50','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE)
        try:
            for i in ids:
                d.qpos[:]=data[i,1:1+m.nq];d.qvel[:]=data[i,1+m.nq:1+m.nq+m.nv];mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera);frame=renderer.render();process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait(timeout=60):raise RuntimeError('Video encode failed')
        finally:
            if process.poll() is None:process.kill();process.wait()
        subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','null','-'],check=True,timeout=60);outputs.append(str(target))
    renderer.close();report={'videos':outputs,'source_indices':ids,'source_times':[float(data[i,0]) for i in ids],'fps':50,'physics_steps':0,'camera_note':'Demo has no humanoid head: second view is overhead/tool overview, not a claimed robot first-person camera.','training_ready':False}
    write(out/'verification.json',report);return report


def render_native(attempt,out):
    """Native head camera and external camera, measured states, no physics steps."""
    import mujoco
    from PIL import Image
    attempt=Path(attempt);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_binary_path(str(attempt/'scene.mjb'));d=mujoco.MjData(m);states=np.load(attempt/'states.npz')['samples']
    if not len(states):raise ValueError('No executed physical states to render')
    renderer=mujoco.Renderer(m,height=360,width=640);ids=list(range(3,len(states),4))
    if not ids or ids[-1]!=len(states)-1:ids.append(len(states)-1)
    camera=mujoco.MjvCamera();camera.lookat[:]=[-.25,0,.7];camera.distance=2.8;camera.azimuth=130;camera.elevation=-20
    videos=[];calibration={};saved=[];sample_ids=set(np.linspace(0,len(ids)-1,min(7,len(ids)),dtype=int))
    try:
        for name,view in [('ego','head_stereo_left'),('external',camera)]:
            target=out/f'{name}.mp4';proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','640x360','-framerate','50','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE);records=[]
            try:
                for frame,i in enumerate(ids):
                    row=states[i];d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];d.ctrl[:]=row[1+m.nq+m.nv:];mujoco.mj_forward(m,d)
                    renderer.update_scene(d,view);rgb=renderer.render()
                    if rgb.shape!=(360,640,3) or np.std(rgb)<2:raise RuntimeError('Empty rendering')
                    proc.stdin.write(rgb.tobytes());c=renderer.scene.camera[0]
                    records.append({'source_row':i,'time':float(row[0]),'position':c.pos.tolist(),'forward':c.forward.tolist(),'up':c.up.tolist()})
                    if frame in sample_ids:
                        p=out/f'{name}_{frame:05d}.png';Image.fromarray(rgb).save(p);saved.append(str(p))
                proc.stdin.close()
                if proc.wait(timeout=60):raise RuntimeError('Encode failed')
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
            subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','null','-'],check=True,timeout=60)
            videos.append(str(target));calibration[name]=records
    finally:renderer.close()
    report={'renderer':'MuJoCo EGL','videos':videos,'physics_steps':0,'frame_count':len(ids),'fps':50,'resolution':[640,360],'source_rows':ids,'source_times':[float(states[i,0]) for i in ids],'camera_frames':calibration,'preview_images':saved,'training_ready':False,'temporal_review_status':'pending_visual_review','model':'scene.mjb'}
    write(out/'verification.json',report);return {'videos':videos,'frames':len(ids),'physics_steps':0}
