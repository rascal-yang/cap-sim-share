"""Read-only contact-control MP4 captions and full-action dataset export."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .agent import save
from .render_bootstrap import FONT
from ..workflow.contracts import file_hash,now


def deliver(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());out=root/'delivery';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return json.loads((out/'manifest.json').read_text())
    started=now();tick=time.monotonic();items=[]
    for index,h in enumerate(summary['attempts']):
        case=Path(h['result']['experiment_path']);src=case/'inspection/review_zh.mp4';dest=out/f'{case.name}.mp4'
        caption=out/f'{index:02d}_caption.png';im=Image.new('RGB',(1920,120),(16,20,26));dr=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,23)
        authority='本地等价复验' if h['review'].get('authority')=='local_exact_seed_revalidation' else 'API复核'
        text=[f"{summary['experiment_id']} | {case.name}",f"原生验收：{h['result']['success']} | 接触复核：{h['result'].get('reviewed_success')} | {authority}：{h['review']['action']} | 验收标准未改写",
              f"原 token 前缀 {h['result']['original_token_steps']*.02:.2f}s；mode1 + 显式关节控制；继承 {h['result'].get('inherited_physics_steps',0)*.005:.2f}s，新执行 {h['result'].get('executed_physics_steps',h['result']['physics_steps'])*.005:.2f}s"]
        for y,line in enumerate(text):dr.text((10,5+36*y),line,font=font,fill='white')
        im.save(caption)
        n=(h['result']['physics_steps']+3)//4+1
        if not dest.exists():subprocess.run(['ffmpeg','-v','error','-i',str(src),'-loop','1','-framerate','50','-i',str(caption),'-filter_complex','[0:v]pad=1920:720:0:0[base];[base][1:v]overlay=0:600:shortest=1[v]','-map','[v]','-frames:v',str(n),'-an','-c:v','libx264','-threads','4','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(dest),'-f','null','-'],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=nb_frames,width,height','-of','json',str(dest)]))['streams'][0]
        if int(probe['nb_frames'])!=n:raise ValueError('Video frame mismatch')
        entry={'case':str(case),'video':str(dest),'video_sha256':file_hash(dest),'frames':n,'fps':50,'accepted':h['accepted'],'source_video':str(src),'source_sha256':file_hash(src),'temporal_review':'pending'}
        if h['accepted']:
            folder=out/'dataset'/case.name;folder.mkdir(parents=True,exist_ok=True)
            trace=np.load(case/'execution/trajectory.npz');raw=np.load(case/'execution/controller.npz')['records'];logs=json.loads((case/'execution/contact_control.json').read_text());num=len(logs)
            if len(trace['state'])!=num*4:raise ValueError('Partial policy tick cannot export')
            token_dim=raw.shape[1]-2-1762-29
            arrays={'state_pre':trace['state'][::4],'state_post':trace['next_state'][3::4],
                'motion_token':raw[:,2:2+token_dim].astype(np.float32),'encoder_mode':raw[:,1].astype(np.int8),
                'body_joint_target_rad':np.asarray([l['executed_target_rad'] for l in logs]),'kp':np.asarray([l['kp'] for l in logs]),'kd':np.asarray([l['kd'] for l in logs]),
                'inherited_prefix':np.arange(num)<h['result'].get('branch_start_frame',0),'commands_200hz':trace['commands'],'state_pre_200hz':trace['state'],'state_post_200hz':trace['next_state']}
            target=folder/'episode.npz'
            if target.exists():
                old=np.load(target)
                if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):raise ValueError('Existing export differs')
            else:np.savez_compressed(target,**arrays)
            metadata={'source_case':str(case),'lineage':h['result'].get('lineage',[]),'fully_new_rollout':h['result'].get('fully_new_rollout',True),'branch_start_frame':h['result'].get('branch_start_frame',0),'control_variant':h['result']['control_variant'],'token_alone_replays_execution':False,'training_ready':False,'acceptance_authority':h['review'].get('authority','api'),'temporal_review':'pending','arrays':{k:list(v.shape) for k,v in arrays.items()},'units':'body targets rad, kp Nm/rad, kd Nm/(rad/s), substep final actuator controls from source commands','state_layout':'sim_time, qpos, qvel; scene.mjb defines dimensions','command_layout':'source_frame, 29 final body targets, final actuator controls including hands','source_scene':str(case/'scene.mjb'),'source_scene_sha256':file_hash(case/'scene.mjb'),'source_controller_log_sha256':file_hash(case/'execution/contact_control.json'),'source_trace_sha256':file_hash(case/'execution/trajectory.npz'),'video':str(dest)}
            save(folder/'metadata.json',metadata);entry['dataset']=str(target);entry['dataset_sha256']=file_hash(target)
        items.append(entry)
    manifest={'experiment_id':summary['experiment_id'],'started_at':started,'ended_at':now(),'delivery_wall_s':time.monotonic()-tick,'physics_steps':0,'items':items}
    save(out/'manifest.json',manifest);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();deliver(a.root)
