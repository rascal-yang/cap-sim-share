#!/usr/bin/env python3
import argparse,json,subprocess,time
from pathlib import Path
from humanoid_mimicgen.augmentation.suite_inspect import inspect_case
from humanoid_mimicgen.augmentation.render_bootstrap import FONT
from humanoid_mimicgen.workflow.contracts import now,file_hash
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);rows=json.loads((root/'summary.json').read_text())['results'];selected={'apple_00','apple_03','apple_05','apple_09'}
selected.update(r['condition_id'] for r in rows if r['arms']['baseline']['success']!=r['arms']['feedback']['success'])
started=now();tick=time.monotonic();videos=[]
for r in rows:
 if r['condition_id'] not in selected:continue
 views=[]
 for arm in ('baseline','feedback'):
  trial=r['arms'][arm];case=Path(trial['result']['experiment_path']);out=case/'inspection'
  ins=json.loads((out/'inspection.json').read_text()) if (out/'inspection.json').exists() else inspect_case(case,out)
  views.append(ins)
 caption=root/(r['condition_id']+'_caption.txt');x=r['arms']['baseline'];y=r['arms']['feedback'];shift=[round((r['shift'][0]-.005)*100,2),round(r['shift'][1]*100,2)]
 caption.write_text(f"{root.name} | {r['condition_id']} | 相对基准偏移XY={shift}厘米\n左：原策略 成功={x['success']} 抬升={x['diagnostics']['max_lift_m']*100:.1f}cm；右：编码前右腕闭环 成功={y['success']} 抬升={y['diagnostics']['max_lift_m']*100:.1f}cm\n保存状态重渲染；不平滑。若安全停止提前截断，该侧保持末帧。")
 dest=root/(r['condition_id']+'_基线_vs_腕部闭环.mp4')
 filt=f'[0:v]crop=1280:600:0:0[a];[1:v]crop=1280:600:0:0[b];[a][b]hstack=inputs=2,drawbox=x=0:y=0:w=iw:h=85:color=black:t=fill,drawtext=fontfile={FONT}:textfile={caption}:fontsize=23:fontcolor=white:x=10:y=3[out]'
 subprocess.run(['ffmpeg','-v','error','-i',views[0]['video'],'-i',views[1]['video'],'-filter_complex',filt,'-map','[out]','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
 subprocess.run(['ffmpeg','-v','error','-i',str(dest),'-f','null','-'],check=True)
 videos.append({'condition_id':r['condition_id'],'path':str(dest),'sha256':file_hash(dest),'views':views,'full_decode_passed':True});(root/'videos.json').write_text(json.dumps(videos,indent=2,ensure_ascii=False));print(json.dumps({'condition':r['condition_id'],'path':str(dest)},ensure_ascii=False),flush=True)
(root/'rendering_timing.json').write_text(json.dumps({'started_at':started,'ended_at':now(),'wall_s':time.monotonic()-tick,'selection':'center,Y+2cm,XY+2cm,X+4cm plus every changed success/failure outcome'},indent=2))
