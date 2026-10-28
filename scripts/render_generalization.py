#!/usr/bin/env python3
"""Render representative successful and failed saved states; no physics steps."""
import argparse,json,time,subprocess
from pathlib import Path
from generalization_records import roots as study_roots, conditions as study_conditions
from humanoid_mimicgen.augmentation.suite_inspect import inspect_case
from humanoid_mimicgen.augmentation.render_bootstrap import FONT
from humanoid_mimicgen.workflow.contracts import now,file_hash
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
rows=study_conditions(root);videos=[];started=now();tick=time.monotonic()
for label in ('plate','chair','button','apple'):
    rs=[r for r in rows if r['task_label']==label];success=[];fail=[]
    for r in rs:
        for v in r['verification'].values():
            if v['passed']:success.append((r,v['full']));break
        if r.get('seed') and not r['seed']['success']:fail.append((r,r['seed']))
    selected=[]
    if success:selected.append(('成功',max(success,key=lambda x:sum(d*d for d in x[0]['shift']))))
    if fail:selected.append(('失败',max(fail,key=lambda x:(not any(v['passed'] for v in x[0]['verification'].values()),sum(d*d for d in x[0]['shift'])))))
    for status,(r,trial) in selected:
        case=Path(trial['result']['experiment_path']);out=case/'inspection';t=time.monotonic()
        try:
            ins=json.loads((out/'inspection.json').read_text()) if (out/'inspection.json').exists() else inspect_case(case,out)
            caption=root/f'{r["condition_id"]}_{status}_caption.txt'
            caption.write_text(f'{root.name} | {r["condition_id"]} | {status} | 物体平移 {r["shift"]} m\n原生验收 {trial["audit"]["native_success"]}；'+('批准掌面验收 '+str(trial['success']) if label=='button' else '原生任务标准')+'；保存状态重渲染，非新增物理')
            dest=root/f'{r["condition_id"]}_{status}.mp4'
            subprocess.run(['ffmpeg','-v','error','-i',ins['video'],'-vf',f'drawbox=x=0:y=0:w=iw:h=58:color=black:t=fill,drawtext=fontfile={FONT}:textfile={caption}:fontsize=21:fontcolor=white:x=10:y=3','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
            subprocess.run(['ffmpeg','-v','error','-i',str(dest),'-f','null','-'],check=True)
            videos.append({'task':label,'condition_id':r['condition_id'],'status':status,'path':str(dest),'sha256':file_hash(dest),'inspection':ins,'rendering_wall_s':time.monotonic()-t,'full_decode_passed':True})
        except Exception as e:videos.append({'task':label,'condition_id':r['condition_id'],'error':str(e)})
        (root/'videos.json').write_text(json.dumps(videos,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in videos[-1].items() if k!='inspection'},ensure_ascii=False),flush=True)
(root/'rendering_timing.json').write_text(json.dumps({'started_at':started,'ended_at':now(),'wall_s':time.monotonic()-tick},indent=2))
(root/'visual_realism_review.json').write_text(json.dumps({'accepted':False,'training_ready':False,'status':'pending_visual_review','temporal_review_status':'not_inspected','full_decode_checks':'see videos.json','physics_and_visual_quality_separate':True},indent=2))
