"""Render every physical diagnostic to a shallow captioned MP4; zero physics."""
import json,subprocess,time,sys
from pathlib import Path
from humanoid_mimicgen.augmentation.suite_inspect import inspect_case
from humanoid_mimicgen.augmentation.render_bootstrap import FONT
from humanoid_mimicgen.workflow.contracts import now,file_hash
from humanoid_mimicgen.augmentation.agent import save
REPO=Path(__file__).resolve().parents[1];BASE=REPO.parent/'experiments/2026-10/26'
items=[]
for rn in ['20261026_0953_apple_navigation_forensics','20261026_0958_apple_navigation_transfer']:
 root=BASE/rn
 for p in sorted(root.glob('*/summary.json')):
  d=json.loads(p.read_text())
  if 'execution' not in d:continue
  items.append((root,p.parent.name,Path(d['prepared']['experiment_path']),'legacy',d['execution']))
for rn in ['20261026_0956_apple_early_handoff','20261026_1005_apple_early_handoff']:
 root=BASE/rn
 for d in json.loads((root/'summary.json').read_text())['results']:
  items.append((root,d['condition_id'],Path(d['trial']['result']['experiment_path']),'current',d['trial']['result']))
start=now();tick=time.monotonic();videos=[]
for root,label,case,kind,res in items:
 if kind=='legacy':
  subprocess.run([str(REPO/'scripts/humano_python.sh'),'-m','humanoid_mimicgen.render_standing',str(case)],check=True,stdout=subprocess.DEVNULL)
  sources=[case/'video/external.mp4',case/'video/ego.mp4'];sheets=[case/'video/external_contact_sheet.jpg',case/'video/ego_contact_sheet.jpg']
  success=res['physical_task_completed'];native=res['native']['success'];failure=res['reason_code'];filt='[0:v][1:v]hstack=inputs=2'
 else:
  ins=inspect_case(case,case/'inspection');sources=[Path(ins['video'])];sheets=[Path(ins['sheet'])];success=native=res['success'];failure=res['failure'];filt='[0:v]crop=1280:600:0:0'
 caption=root/(label+'_caption.txt');caption.write_text(f'{root.name}\n{label} | 原生成功={native} | 完整抓取保持={success} | 终止原因={failure}\n保存实测状态重渲染；无新物理步、无状态平滑；当前任务使用Dex3手。')
 dest=root/(label+'_全过程.mp4');args=['ffmpeg','-v','error']
 for src in sources:args+=['-i',str(src)]
 filt+=f',drawbox=x=0:y=0:w=iw:h=85:color=black:t=fill,drawtext=fontfile={FONT}:textfile={caption}:fontsize=20:fontcolor=white:x=8:y=3[out]'
 subprocess.run(args+['-filter_complex',filt,'-map','[out]','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
 subprocess.run(['ffmpeg','-v','error','-i',str(dest),'-f','null','-'],check=True)
 videos.append({'root':str(root),'label':label,'case':str(case),'path':str(dest),'sha256':file_hash(dest),'sheets':[str(x) for x in sheets],'full_decode_passed':True});save(BASE/'20261026_0953_apple_navigation_forensics/rendering_progress.json',videos);print(label,str(dest),flush=True)
save(BASE/'20261026_0953_apple_navigation_forensics/rendering.json',{'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'physics_steps':0,'videos':videos})
