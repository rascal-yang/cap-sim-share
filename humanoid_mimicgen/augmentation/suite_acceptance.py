"""Read-only acceptance tables and one complete, captioned MP4 per task."""
import argparse,csv,json,subprocess,time,shutil,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .suite_inspect import inspect_case
from ..workflow.contracts import now,file_hash

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LABELS={'BallKick':'踢球','ChairPush':'推椅子','ChairSit':'坐椅子','DollyPush':'推小车','GroundBasketPick':'地面篮子','GroundBottlePick':'地面瓶子','GroundBoxPick':'地面箱子','TableApplePick':'桌面苹果','TableButtonPress':'按按钮','TablePlatePick':'桌面盘子'}
REASONS={
'BallKick':'原始回放、零扰动及四个 ±1 cm 偏移均获 API 接受；本批没有失败案例。',
'ChairPush':'零扰动 3 次均失败；末次位移约 2.3 cm，原生要求 40 cm，持续推行计数不足。',
'ChairSit':'原始回放延长 1 秒后成功；零扰动末端距离 37.0 cm，超过 28 cm 验收半径，未形成坐稳支撑。',
'DollyPush':'零扰动延长 3 秒仍只移动约 23.4 cm，低于 50 cm 要求；API 请求逐手、随时间变化的纠偏。',
'GroundBasketPick':'换源及修复仍失败；最后上移手腕 2.5 cm 后没有手部接触，仅有膝/踝碰触。此前部分抬升不足 9 cm。',
'GroundBottlePick':'三次零扰动均未建立稳定抓握和抬升；末次接触仅集中在少量手指，物体高度基本未变。',
'GroundBoxPick':'四次零扰动中两次曾满足原生成功，但随后掉落/末端高度不足；API 接受 0 次，未放行偏移。',
'TableApplePick':'原始回放成功；零扰动没有有效接触，双腕跟踪 RMSE 约 13.3/14.7 cm；API 请求有符号误差与独立逐手纠偏。',
'TableButtonPress':'原始回放延长 0.8 秒后成功；零扰动三次均失败，末次合格按压仅 0.02 秒，原生要求 1 秒。',
'TablePlatePick':'换源及手腕修复均未通过零扰动；末次抬升约 4 cm，低于 9 cm 要求，仍受桌面支撑。'}


def key(name):return name.removeprefix('G1Wholebody').removesuffix('Teleop')
def stage(c):return '原始回放' if c['mode']=='original_token' else '偏移扩增' if any(c['object_delta_xy_m']) else '零扰动复用'
def verdict(c):
    if not c.get('api_review'):return '待 API 验收'
    action=json.loads(Path(c['api_review']).read_text())['decision']['action']
    return {'accept':'API 接受','retry':'API 要求重试','reject':'API 拒绝','needs_tool':'API 请求补工具'}.get(action,'API '+action)
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def run(args):subprocess.run(args,check=True,stdout=subprocess.DEVNULL)
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,duration,r_frame_rate','-of','json',str(p)]))['streams'][0]
def stamp(path,lines,size=(1920,140),font_size=24):
    im=Image.new('RGB',size,(15,20,29));draw=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,font_size)
    for i,line in enumerate(lines):draw.text((18,8+i*(font_size+7)),line,font=font,fill=(235,239,246))
    im.save(path)
def csv_save(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def build(summary_path,out):
    out=Path(out);out.mkdir(exist_ok=True);tick=time.monotonic();started=now()
    if (out/'acceptance_manifest.json').exists():raise ValueError('Completed package exists; choose a new output directory')
    summary=json.loads(Path(summary_path).read_text());cases=summary['cases'];tasks=summary['tasks']
    save(out/'review_record.json',{'review_id':out.name,'started_at':started,'purpose':'All-task acceptance table and complete success/failure videos','physics_steps':0,'source_summary':str(Path(summary_path).resolve()),'source_summary_sha256':file_hash(summary_path),'code_sha256':file_hash(__file__),'visual_training_acceptance':'not certified','excluded_study':'20261021_2302_api_quality_gate_followup'})
    shutil.copy2(summary_path,out/'source_summary.json')
    video_dir=out/'videos';video_dir.mkdir(exist_ok=True);assets=out/'assets';assets.mkdir(exist_ok=True)
    rendered=0.;encoding=0.;validation=0.;table=[];case_rows=[];videos=[]
    for task in tasks:
        name=task['task'];short=key(name);label=LABELS[short];tc=sorted([c for c in cases if c['task']==name],key=lambda c:c['result']['started_at'])
        work=assets/short;work.mkdir(exist_ok=True);parts=[];timeline=[];cursor=150
        intro=work/'intro.png'
        stamp(intro,[f'{out.name} | {label} 验收视频','',f'原始回放：{task["replay"]["passed"]}/{task["replay"]["attempts"]}  零扰动物理：{task["zero"]["passed"]}/{task["zero"]["attempts"]}  零扰动 API 接受：{task["zero"]["api_accepted"]}',f'偏移扩增：{task["augmentation"]["passed"]}/{task["augmentation"]["attempts"]}  API 接受：{task["augmentation"]["api_accepted"]}',f'本视频包含 {len(tc)} 次有效物理尝试，按时间顺序完整播放。',REASONS[short],'','三视角：全身 / 物体近景 / 头部；原速 50 fps，无平滑。','物理成功与 API 接受分别标注；物理成功不等于训练质量通过。','本页是验收说明页，非仿真帧。'],size=(1920,740),font_size=27)
        intro_mp4=work/'intro.mp4';start=time.monotonic()
        run(['ffmpeg','-v','error','-y','-loop','1','-framerate','50','-i',str(intro),'-frames:v','150','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(intro_mp4)]);encoding+=time.monotonic()-start;parts.append(intro_mp4)
        for i,c in enumerate(tc):
            folder=Path(c['result']['experiment_path']);sources=sorted(folder.rglob('review_zh.mp4'))
            if sources:source=sources[0]
            else:
                target=out/'rendered_missing'/c['experiment_id'];target.parent.mkdir(exist_ok=True)
                inspection=inspect_case(folder,target);rendered+=inspection['rendering_and_inspection_wall_s'];source=Path(inspection['video'])
            metadata=probe(source)
            if (metadata['width'],metadata['height'],metadata['r_frame_rate'])!=(1920,600,'50/1'):raise ValueError('Unexpected source video geometry/timing')
            frames=int(metadata['nb_frames']);r=c['result'];d=json.loads(Path(r['decision_path']).read_text())['decision'];result='物理成功' if c['success'] else '物理失败';api=verdict(c)
            banner=work/f'{i:02d}_caption.png';stamp(banner,[f'{label} | {stage(c)} | 第 {i+1}/{len(tc)} 次尝试 | episode {r["episode"]} | {result} | {api}',c['experiment_id'],f'模式：{c["mode"]}  末段延长：{d["tail_s"]} s  手腕偏移：{d["wrist_delta_m"]} m  物体偏移：{c["object_delta_xy_m"]} m',f'验收包：{out.name} | 保存状态原速重放；不新增物理、不平滑、不改变任务判定。'])
            part=work/f'{i:02d}.mp4';start=time.monotonic()
            run(['ffmpeg','-v','error','-y','-i',str(source),'-loop','1','-framerate','50','-i',str(banner),'-filter_complex','[0:v]pad=1920:740:0:140:color=0x0f141d[base];[base][1:v]overlay=0:0:shortest=1[v]','-map','[v]','-frames:v',str(frames),'-an','-c:v','libx264','-threads','4','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(part)])
            encoding+=time.monotonic()-start;parts.append(part)
            if int(probe(part)['nb_frames'])!=frames:raise ValueError('Composition changed source frame count')
            entry={'experiment_id':c['experiment_id'],'stage':stage(c),'first_video_frame':cursor,'last_video_frame':cursor+frames-1,'start_s':cursor/50,'end_s':(cursor+frames)/50,'source_video':str(source),'source_video_sha256':file_hash(source),'source_frames':frames,'source_frame_mapping':str(source.parent/'frame_mapping.json'),'physical_success':c['success'],'api_verdict':api,'api_review':c.get('api_review')};timeline.append(entry);cursor+=frames
            case_rows.append({'任务':label,'实验ID':c['experiment_id'],'阶段':stage(c),'源episode':r['episode'],'实际开始时间':r['started_at'],'实际结束时间':r['ended_at'],'模式':c['mode'],'物体偏移m':str(c['object_delta_xy_m']),'延长秒':d['tail_s'],'手腕偏移m':str(d['wrist_delta_m']),'物理结果':result,'独立检查':r['independent_success'],'执行覆盖率':r['coverage'],'API结论':api,'API决策':c.get('api_review') or '', '视频开始秒':entry['start_s'],'视频结束秒':entry['end_s'],'源实验目录':str(folder)})
        listing=work/'concat.txt';listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in parts))
        dest=video_dir/f'{short}.mp4';start=time.monotonic();run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(dest)]);encoding+=time.monotonic()-start
        start=time.monotonic();meta=probe(dest);run(['ffmpeg','-v','error','-xerror','-i',str(dest),'-f','null','-']);validation+=time.monotonic()-start
        if int(meta['nb_frames'])!=cursor:raise ValueError('Final video dropped frames')
        save(video_dir/f'{short}.timeline.json',{'intro_frames':150,'fps':50,'segments':timeline,'state_smoothing':False,'physics_steps':0})
        # Contact/result samples from each attempt; kept separately from temporal acceptance.
        picks=[]
        for seg in timeline:
            for frac in (.65,.99):picks.append(seg['first_video_frame']+int((seg['source_frames']-1)*frac))
        sheet=Image.new('RGB',(1920,370*((len(picks)+1)//2)),(15,20,29))
        for j,frame in enumerate(picks):
            png=work/f'preview_{j:02d}.jpg';run(['ffmpeg','-v','error','-y','-ss',str(frame/50),'-i',str(dest),'-frames:v','1',str(png)])
            im=Image.open(png).resize((960,370));sheet.paste(im,((j%2)*960,(j//2)*370))
        sheet_path=video_dir/f'{short}.jpg';sheet.save(sheet_path,quality=90)
        def rate(k):v=task[k];return f'{v["passed"]}/{v["attempts"]}'
        row={'任务':label,'源数据检查':'已通过','原始回放物理成功/尝试':rate('replay'),'原始回放API接受数':task['replay']['api_accepted'],'接触阶段标注':'已完成（含起止）','零扰动物理成功/尝试':rate('zero'),'零扰动API接受数':task['zero']['api_accepted'],'偏移物理成功/尝试':rate('augmentation'),'偏移API接受数':task['augmentation']['api_accepted'],'当前停止阶段':'偏移扩增完成' if task['augmentation']['api_accepted']==4 else '零扰动验收未通过','原因或限制':REASONS[short],'训练就绪':'否：时序视觉及训练导出未验收','有效尝试数':len(tc),'视频秒数':float(meta['duration']),'视频':str(dest)};table.append(row)
        videos.append({'task':name,'label':label,'video':str(dest),'sha256':file_hash(dest),'size_bytes':dest.stat().st_size,'probe':meta,'cases':len(tc),'preview_sheet':str(sheet_path),'timeline':str(video_dir/f'{short}.timeline.json')})
        save(out/'progress.json',{'completed_tasks':len(videos),'videos':videos});print(json.dumps({'task':label,'cases':len(tc),'seconds':meta['duration'],'video':str(dest)},ensure_ascii=False),flush=True)
    csv_save(out/'任务验收表.csv',table);csv_save(out/'逐次实验明细.csv',case_rows)
    lines=[f'# 任务验收表：{out.name}',f'创建时间：{started}；汇总来源：{summary_path}。','','数字均为物理成功次数 / 实际尝试次数，含修复和换源。API 接受单独列示；未执行的偏移不是失败。所有视频完整保留本表覆盖的有效尝试，包括成功和失败；踢球本批没有失败案例，不混入无效实验。','','|任务|源检查|原始回放|接触标注|零扰动物理|零扰动 API 接受|偏移物理 / API 接受|当前停止点|视频|','|---|---|---|---|---|---|---|---|---|']
    for row in table:lines.append(f'|{row["任务"]}|通过|{row["原始回放物理成功/尝试"]}|完成|{row["零扰动物理成功/尝试"]}|{row["零扰动API接受数"]}|{row["偏移物理成功/尝试"]} / {row["偏移API接受数"]}|{row["当前停止阶段"]}|[MP4]({row["视频"]})|')
    lines+=['','[任务验收表 CSV](任务验收表.csv) · [逐次实验明细 CSV](逐次实验明细.csv)','','所有样本 training_ready=false。箱子曾瞬时满足原生判定，但 API 因掉落/末端保持不足拒绝，不能算质量合格。视频为已有状态重渲染/合成，不是新物理实验；保持原速、无插帧平滑。原轨迹可见，属于轨迹重定向，不能据此宣称策略迁移。','']
    for task,row in zip(tasks,table):
        lines.extend([f'## {row["任务"]}',row['原因或限制'],'','|实验ID|阶段|episode|物理结果|API结论|延长秒|手腕偏移m|视频时间|','|---|---|---|---|---|---|---|---|'])
        for c in [c for c in case_rows if c['任务']==row['任务']]:lines.append(f'|{c["实验ID"]}|{c["阶段"]}|{c["源episode"]}|{c["物理结果"]}|{c["API结论"]}|{c["延长秒"]}|{c["手腕偏移m"]}|{c["视频开始秒"]:.2f}–{c["视频结束秒"]:.2f}s|')
        lines.extend(['','最终 API 原文：',task.get('reason') or '', ''])
    manifest={'review_id':out.name,'started_at':started,'ended_at':now(),'total_elapsed_s':time.monotonic()-tick,'new_physics_steps':0,'planning_wall_s':None,'simulation_wall_s':0.,'missing_case_rendering_and_inspection_wall_s':rendered,'video_encoding_wall_s':encoding,'full_video_decode_validation_wall_s':validation,'task_count':len(tasks),'case_count':len(cases),'videos':videos,'training_ready':False,'temporal_review_status':'pending','video_composition':'3-second explanatory title then every recorded case at 50fps, untrimmed and unsmoothed'}
    save(out/'acceptance_manifest.json',manifest);lines+=['','制作时长：',f'新增物理 0 秒；补渲染/接触重建 {rendered:.2f} 秒；视频编码 {encoding:.2f} 秒；完整解码检查 {validation:.2f} 秒；总历时 {manifest["total_elapsed_s"]:.2f} 秒；计划用时未知。',f'实际结束：{manifest["ended_at"]}。']
    (out/'ACCEPTANCE.md').write_text('\n'.join(lines)+'\n');return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.summary,a.out)
