"""Render saved bootstrap transitions to Chinese-captioned review MP4s. No physics."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ..workflow.contracts import now

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def forbidden(*args,**kwargs):raise RuntimeError('Rendering cannot integrate physics')


def render_case(folder,out,number):
    import mujoco
    for key in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,key,forbidden)
    started=now();tick=time.monotonic();result=json.loads((folder/'result.json').read_text())
    contract=json.loads((folder/'experiment.json').read_text());acceptance=contract['physics']['acceptance_thresholds']
    modelpath=folder/'scene.mjb';tracepath=folder/'execution/trajectory.npz'
    model=mujoco.MjModel.from_binary_path(str(modelpath));data=mujoco.MjData(model)
    trace=np.load(tracepath);pre=trace['state'];post=trace['next_state'];commands=trace['commands']
    metrics=json.loads((folder/'execution/metrics.json').read_text());frames=[-1]+list(range(3,len(post),4))
    if frames[-1]!=len(post)-1:frames.append(len(post)-1)
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    # Rendering-only illumination correction: retain original geometry/materials.
    model.light_diffuse[:]*=.35;model.light_specular[:]*=.35
    renderer=mujoco.Renderer(model,height=480,width=640)
    data.qpos[:]=pre[0,1:1+model.nq];mujoco.mj_forward(model,data)
    external=mujoco.MjvCamera();external.lookat[:]=[-.35,0,.8];external.distance=2.3;external.azimuth=-65;external.elevation=-25
    close=mujoco.MjvCamera();close.lookat[:]=data.body('E_button_01_13').xpos+np.array([0,0,.07]);close.distance=.55;close.azimuth=115;close.elevation=-40
    dest=out/f'{number:02d}_{folder.name}.mp4'
    proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x640','-r','50','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    font=ImageFont.truetype(FONT,25);small=ImageFont.truetype(FONT,21)
    mapping=[];stills=[];selected=set(np.linspace(0,len(frames)-1,8,dtype=int));consecutive=[];count=0
    for row in metrics:
        count=count+1 if row['pressing'] else 0;consecutive.append(count*.02)
    mode=result['mode'];tail=result.get('hold_tail',False)
    shift=result['object_delta_xy_m'];source_count=result.get('source_frames',len(post)//4)
    title=(f'原始 token 精确回放｜示范 {result["episode"]}' if mode=='original_token' and not tail else '原始 token + 接触保持补全' if mode=='original_token' else f'mode1 轨迹重定向｜按钮 X {shift[0]*100:+.0f} cm，Y {shift[1]*100:+.0f} cm')
    outcome='通过原生及独立验收' if result['success'] else '未通过连续按压验收'
    try:
        for frame,index in enumerate(frames):
            state=pre[0] if index<0 else post[index];data.time=float(state[0]);data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:]
            data.ctrl[:]=commands[0 if index<0 else index,-model.nu:];mujoco.mj_forward(model,data)
            panels=[];cameras={}
            for name,camera in [('external',external),('close',close),('ego','head_stereo_left')]:
                renderer.update_scene(data,camera=camera);rgb=renderer.render().copy()
                if rgb.shape!=(480,640,3) or np.std(rgb)<2:raise RuntimeError('Empty frame')
                panels.append(rgb);cam=renderer.scene.camera[0];cameras[name]={'position':cam.pos.tolist(),'forward':cam.forward.tolist(),'up':cam.up.tolist()}
            if index<0:phase='初始状态';force=0.;held=0.;passed=False
            else:
                policy_index=min(index//4,len(metrics)-1);met=metrics[policy_index];force=met['downward_force_N'];held=consecutive[policy_index];passed=met['native_success']
                phase='按压达标，持续保持' if met['pressing'] else '接触按钮，按压力不足' if met['contact_count'] else '接近 / 尚未接触按钮'
                if tail and index//4>=source_count:phase='新增末端保持｜'+phase
                if passed:phase+='｜已满足成功判据'
            img=Image.new('RGB',(1920,640),(18,22,28));img.paste(Image.fromarray(np.hstack(panels)),(0,96));draw=ImageDraw.Draw(img)
            draw.text((15,4),f'{number:02d}  {title}   |   最终结果：{outcome}',font=font,fill=(100,235,155) if result['success'] else (255,205,110))
            draw.text((15,37),folder.name+'  |  已记录状态重渲染，原速，无平滑、无新物理',font=small,fill='white')
            for j,label in enumerate(('全身视角','手与按钮近景','头部视角')):draw.text((15+640*j,68),label,font=small,fill=(180,205,230))
            draw.text((15,579),f'片内 {frame/50:.2f}s｜仿真时钟 {data.time:.3f}s｜{phase}',font=small,fill='white')
            draw.text((15,610),f'向下力 {force:.2f} N（要求 ≥{acceptance["min_downward_force_n"]:g} N）｜连续达标 {held:.2f}s / {acceptance["hold_steps"]*.02:.2f}s｜目标为桌面按钮，地面红球不是目标',font=small,fill=(220,225,235))
            proc.stdin.write(np.asarray(img).tobytes())
            mapping.append({'video_frame':frame,'video_time_s':frame/50,'post_step_index':index,'sim_time_s':float(data.time),'phase':phase,'downward_force_N':force,'continuous_press_s':held,'native_success':passed,'camera_frames':cameras})
            if frame in selected:
                img.save(out/f'{number:02d}_frame_{frame:05d}.jpg',quality=90);stills.append(img.resize((960,320)))
        proc.stdin.close()
        if proc.wait(timeout=60):raise RuntimeError('Encoder failed')
    finally:
        renderer.close()
        if proc.poll() is None:proc.kill();proc.wait()
    sheet=Image.new('RGB',(1920,1280))
    for i,im in enumerate(stills):sheet.paste(im,((i%2)*960,(i//2)*320))
    sheet.save(out/f'{number:02d}_contact_sheet.jpg',quality=92)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(dest),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(dest)]))['streams'][0]
    if int(probe['nb_frames'])!=len(frames):raise RuntimeError('Encoded frame count mismatch')
    save(out/f'{number:02d}_frame_mapping.json',mapping)
    audit={'source_experiment_id':folder.name,'source_study_id':folder.parent.parent.name,'source':str(folder),'video':str(dest),'title':title,'success':result['success'],'started_at':started,'ended_at':now(),'rendering_and_validation_wall_s':time.monotonic()-tick,'physics_steps':0,'source_sha256':{str(p):sha(p) for p in (modelpath,tracepath,folder/'experiment.json')},'includes_initial_and_final':True,'state_smoothing':False,'render_only_changes':{'light_diffuse_multiplier':.35,'light_specular_multiplier':.35,'cameras':'external fixed, button close fixed, original head camera'},'renderer':'MuJoCo 3.3.6 EGL OpenGL','training_ready':False,'probe':probe,'bytes':dest.stat().st_size}
    save(out/f'{number:02d}_render_audit.json',audit);print(json.dumps({'case':number,'video':str(dest),'frames':len(frames)}),flush=True);return audit


def run(studies,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);started=now();tick=time.monotonic()
    folders=[p.parent for study in studies for p in sorted((Path(study)/'experiments').glob('*/result.json'))]
    records=[render_case(folder,out,i+1) for i,folder in enumerate(folders)]
    concat=out/'concat.txt';concat.write_text(''.join("file '"+r['video']+"'\n" for r in records))
    chapters=[';FFMETADATA1'];offset=0;chapter_rows=[]
    for i,r in enumerate(records):
        end=offset+int(r['probe']['nb_frames'])*20
        chapters+=['[CHAPTER]','TIMEBASE=1/1000',f'START={offset}',f'END={end}',f'title={i+1:02d} {r["title"]}']
        chapter_rows.append({'start_s':offset/1000,'end_s':end/1000,'source_experiment_id':r['source_experiment_id'],'title':r['title']});offset=end
    meta=out/'chapters.ffmetadata';meta.write_text('\n'.join(chapters)+'\n');dest=out/'button_all_stages_zh.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(concat),'-i',str(meta),'-map_metadata','1','-map_chapters','1','-c','copy','-movflags','+faststart',str(dest)],check=True)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(dest),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(dest)]))
    save(out/'delivery.json',{'studies':[str(Path(s).resolve()) for s in studies],'started_at':started,'ended_at':now(),'rendering_and_validation_wall_s':time.monotonic()-tick,'physics_steps':0,'video':str(dest),'bytes':dest.stat().st_size,'probe':probe,'chapters':chapter_rows,'cases':records,'training_ready':False})
    print(dest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.studies,a.out)
