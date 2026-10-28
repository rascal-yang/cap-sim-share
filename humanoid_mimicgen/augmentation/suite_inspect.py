"""Read-only substep contact evidence and unmodified-state multi-task review video."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .render_bootstrap import forbidden,FONT
from .agent import save
from ..workflow.contracts import file_hash,now


def validate_trace(trace,expected):
    n=len(trace['state'])
    if n!=expected or len(trace['next_state'])!=n or len(trace['commands'])!=n:raise ValueError('Incomplete substep evidence')
    return n


def inspect_case(folder,out):
    import mujoco
    saved={key:getattr(mujoco,key) for key in ('mj_step','mj_step1','mj_step2')}
    try:
        for key in saved:setattr(mujoco,key,forbidden)
        return _inspect_case(folder,out)
    finally:
        for key,value in saved.items():setattr(mujoco,key,value)


def _inspect_case(folder,out):
    import mujoco
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=False);tick=time.monotonic();started=now()
    result=json.loads((folder/'result.json').read_text());trace=np.load(folder/'execution/trajectory.npz');n=validate_trace(trace,result['physics_steps'])
    m=mujoco.MjModel.from_binary_path(str(folder/'scene.mjb'));d=mujoco.MjData(m)
    pelvis=m.body('pelvis').id
    def descendant(body,ancestor):
        while body>0:
            if body==ancestor:return True
            body=int(m.body_parentid[body])
        return False
    robot={i for i in range(m.nbody) if descendant(i,pelvis)}
    # Retain every non-robot body identity, including support, rather than guessing task contact.
    m.vis.global_.offwidth=640;m.vis.global_.offheight=480
    m.light_diffuse[:]*=.35;m.light_specular[:]*=.35
    renderer=mujoco.Renderer(m,height=480,width=640)
    camera=mujoco.MjvCamera();camera.azimuth=-65;camera.elevation=-20;camera.distance=2.6
    close=mujoco.MjvCamera();close.azimuth=90;close.elevation=-20;close.distance=1.2
    video=out/'review_zh.mp4'
    proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x600','-r','50','-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,20)
    events=[];frames=[];mapping=[];stills=[];policy_count=(n+3)//4;selected=set(np.linspace(0,policy_count,min(12,policy_count+1),dtype=int))
    def state_at(index):
        state=trace['state'][0] if index<0 else trace['next_state'][index]
        d.time=state[0];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];d.ctrl[:]=trace['commands'][max(index,0),-m.nu:];mujoco.mj_forward(m,d)
    def draw_frame(index,frame):
        camera.lookat[:]=[d.qpos[0]+.2,d.qpos[1],.65]
        others=[i for i in range(1,m.nbody) if i not in robot and 'table' not in (m.body(i).name or '').lower()]
        target=d.xpos[others[-1]].copy() if others else d.qpos[:3].copy()
        close.lookat[:]=target
        images=[]
        for cam in (camera,close,'head_stereo_left'):
            renderer.update_scene(d,camera=cam);rgb=renderer.render().copy()
            if rgb.std()<2:raise ValueError('Blank rendered view')
            images.append(rgb)
        image=Image.new('RGB',(1920,600),(18,22,28));image.paste(Image.fromarray(np.hstack(images)),(0,85));dr=ImageDraw.Draw(image)
        dr.text((12,3),folder.name,font=font,fill='white')
        dr.text((12,32),f'{result["task"]} | {result["mode"]} | 原生物理验收：{result["success"]} | 视觉验收待定',font=small,fill=(220,225,235))
        dr.text((12,58),'全身跟随视角                                      物体近景                                      头部视角',font=small,fill=(180,210,235))
        branch_label=(f"继承前缀至帧{result['branch_start_frame']}；" if result.get('inherited_physics_steps') else '')
        dr.text((12,571),branch_label+f'仿真时间 {d.time:.3f}s | 已记录状态重渲染：无新物理、无平滑 | post_step {index}',font=small,fill='white')
        proc.stdin.write(np.asarray(image).tobytes());mapping.append({'video_frame':frame,'post_step':index,'sim_time_s':float(d.time)})
        if frame in selected:
            image.save(out/f'frame_{frame:04d}.jpg',quality=88);stills.append(image.resize((960,300)))
    try:
        state_at(-1);draw_frame(-1,0);frame=1
        for index in range(n):
            state_at(index);contacts=[]
            for ci,c in enumerate(d.contact):
                b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                if (b1 in robot)==(b2 in robot):continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                if force[0]>.05:contacts.append({'robot_body':m.body(b1 if b1 in robot else b2).name,'other_body':m.body(b2 if b1 in robot else b1).name,'force_N':float(force[0]),'distance_m':float(c.dist),'position':c.pos.tolist()})
            row={'post_step':index,'policy_frame':index//4,'time_s':float(d.time),'root_pose':d.qpos[:7].tolist(),'root_velocity':d.qvel[:6].tolist(),'contacts':contacts}
            frames.append(row)
            for c in contacts:
                if c['other_body'] not in ('world',) and 'table' not in (c['other_body'] or '').lower():events.append({'post_step':index,'policy_frame':index//4,'time_s':float(d.time),**c})
            if index%4==3 or index==n-1:draw_frame(index,frame);frame+=1
        proc.stdin.close()
        if proc.wait(timeout=60):raise RuntimeError('Encoding failed')
    finally:
        renderer.close()
        if proc.poll() is None:proc.kill();proc.wait()
    sheet=Image.new('RGB',(1920,300*((len(stills)+1)//2)))
    for i,im in enumerate(stills):sheet.paste(im,((i%2)*960,(i//2)*300))
    sheet.save(out/'contact_sheet.jpg',quality=90)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,nb_frames,duration','-of','json',str(video)]))
    if int(probe['streams'][0]['nb_frames'])!=len(mapping):raise ValueError('Video frame mismatch')
    save(out/'substeps.json',frames);save(out/'frame_mapping.json',mapping)
    by_pair={}
    for e in events:
        key=e['robot_body']+' / '+str(e['other_body']);v=by_pair.setdefault(key,{'first_policy_frame':e['policy_frame'],'last_policy_frame':e['policy_frame'],'peak_N':0.,'substeps':0})
        v.update(last_policy_frame=e['policy_frame'],peak_N=max(v['peak_N'],e['force_N']),substeps=v['substeps']+1)
    review_images=[str(out/'contact_sheet.jpg')]
    if events:
        anchors=[events[0]['policy_frame']+1,events[-1]['policy_frame']+1,max(events,key=lambda x:x['force_N'])['policy_frame']+1]
        picks=sorted(set(max(0,min(policy_count,a+delta)) for a in anchors for delta in (-2,0,2)))
        focus=Image.new('RGB',(1920,300*((len(picks)+1)//2)))
        for j,k in enumerate(picks):
            dest=out/f'contact_frame_{k:04d}.jpg'
            subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{k})','-vsync','0','-frames:v','1',str(dest)],check=True)
            im=Image.open(dest).resize((960,300));ImageDraw.Draw(im).text((5,280),f'video frame {k}; policy frame {k-1}',fill='yellow');focus.paste(im,((j%2)*960,(j//2)*300))
        focus.save(out/'contact_focus.jpg',quality=90);review_images.append(str(out/'contact_focus.jpg'))
    result={'experiment_id':folder.name,'source':str(folder),'started_at':started,'ended_at':now(),'rendering_and_inspection_wall_s':time.monotonic()-tick,'physics_steps':0,'inspected_substeps':n,'expected_substeps':json.loads((folder/'result.json').read_text())['expected_steps'],'native_result_success':result['success'],'safety_stop':result['failure'],'contact_pairs':by_pair,'object_contact_events':events,'initial_root':frames[0]['root_pose'],'final_root':frames[-1]['root_pose'],'max_root_speed_mps':max(np.linalg.norm(f['root_velocity'][:3]) for f in frames),'final_root_speed_mps':float(np.linalg.norm(frames[-1]['root_velocity'][:3])),'video':str(video),'sheet':str(out/'contact_sheet.jpg'),'review_images':review_images,'probe':probe,'training_ready':False,'temporal_review_status':'pending','source_sha256':{str(p):file_hash(p) for p in (folder/'scene.mjb',folder/'execution/trajectory.npz',folder/'result.json')},'render_only_changes':{'diffuse_and_specular_multiplier':.35,'camera':'root-following external, task-object close, native head'},'state_smoothing':False}
    save(out/'inspection.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=inspect_case(a.case,a.out);print(json.dumps({k:v for k,v in r.items() if k not in ('object_contact_events',)},ensure_ascii=False))
