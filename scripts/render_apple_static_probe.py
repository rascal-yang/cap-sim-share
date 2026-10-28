import json,subprocess,time
from pathlib import Path
from humanoid_mimicgen.workflow.contracts import now,file_hash
from humanoid_mimicgen.augmentation.render_bootstrap import FONT
from humanoid_mimicgen.augmentation.agent import save
REPO=Path(__file__).resolve().parents[1];root=Path('/tmp/static_legacy_root').read_text().strip();root=Path(root);started=now();tick=time.monotonic();videos=[]
for name in ['historical_y8','current_y2']:
 case=root/name;s=json.loads((case/'summary.json').read_text());v=s['verification'];
 if not (case/'video/external.mp4').exists():subprocess.run([str(REPO/'scripts/humano_python.sh'),'-m','humanoid_mimicgen.render_standing',str(case)],check=True,stdout=subprocess.DEVNULL)
 caption=root/(name+'_caption.txt');caption.write_text(f'{root.name} | {name}\n显式历史接口：5秒原token＋3秒mode0＋1秒张手接近，再mode1抓取保持\n原生9cm验收={v["native"]["success"]}；保持段有效比例={s["hold_valid_fraction"]*100:.0f}%；最低离桌={v["hold_min_clearance_m"]*100:.2f}cm\n擦桌峰值={v["max_table_N"]:.2f}N；保存状态重渲染，无新增物理、无平滑')
 dest=root/(name+'_全过程.mp4');f=f'[0:v][1:v]hstack=inputs=2,drawbox=x=0:y=0:w=iw:h=105:color=black:t=fill,drawtext=expansion=none:fontfile={FONT}:textfile={caption}:fontsize=20:fontcolor=white:x=8:y=3[out]'
 subprocess.run(['ffmpeg','-y','-v','error','-i',str(case/'video/external.mp4'),'-i',str(case/'video/ego.mp4'),'-filter_complex',f,'-map','[out]','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True);subprocess.run(['ffmpeg','-v','error','-i',str(dest),'-f','null','-'],check=True);videos.append({'path':str(dest),'case':str(case),'sha256':file_hash(dest),'sheets':[str(case/'video/external_contact_sheet.jpg'),str(case/'video/ego_contact_sheet.jpg')],'full_decode_passed':True});print(name,flush=True)
save(root/'rendering.json',{'started_at':started,'ended_at':now(),'wall_s':time.monotonic()-tick,'physics_steps':0,'videos':videos})
