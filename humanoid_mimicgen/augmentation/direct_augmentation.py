"""Direct object perturbation after replay/annotation; zero reuse is diagnostic only."""
import argparse,copy,json,math,time,traceback,subprocess
from pathlib import Path
import numpy as np
from .agent import save
from .quality_suite import PROPERTIES,api_decide,validate_decision,verify_origin,API_CURSOR,same_execution
from .suite_physics import physical_case,validate_guard,source_paths
from .suite_provider import configured_provider
from .suite_evidence import install
from .suite_lock import exclusive
from .suite_inspect import inspect_case
from .trajectory_diagnostics import diagnose
from ..workflow.contracts import now,file_hash

SKILL=Path(__file__).parent/'skills/direct_augmentation/SKILL.md'
PROPS=copy.deepcopy(PROPERTIES)
PROPS['object_shifts_xy_m']={'type':'array','minItems':1,'maxItems':2,'items':{'type':'array','minItems':2,'maxItems':2,'items':{'type':'number'}}}
TOOLS=[{'type':'function','name':'decide','description':'Authorize or review direct perturbation and bounded evidence-based repair.','strict':True,'parameters':{'type':'object','properties':PROPS,'required':list(PROPS),'additionalProperties':False}}]


def validate_direct(d,episodes):
    if set(d)!=set(PROPS):raise ValueError('Unexpected direct decision fields')
    validate_decision({k:v for k,v in d.items() if k!='object_shifts_xy_m'},episodes)
    shifts=d['object_shifts_xy_m']
    if not isinstance(shifts,list) or not 1<=len(shifts)<=2:raise ValueError('Expected one or two shifts')
    for shift in shifts:
        if not isinstance(shift,list) or len(shift)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in shift) or not 0<sum(v*v for v in shift)<=.02**2+1e-12:raise ValueError('Nonzero finite object XY shift must be <=2cm')
    if len({tuple(s) for s in shifts})!=len(shifts):raise ValueError('Duplicate candidates')
    return d


def accepted(r,action):
    return bool(r.get('success') and r.get('independent_success') and r.get('coverage')==1 and not r.get('failure') and action=='accept')


def require_prerequisites(replay,annotation):
    if not accepted(replay,'accept') or annotation['decision']['action']!='accept':raise ValueError('Accepted replay and annotation required')
    d=annotation['decision']
    if not 0<=d['contact_start']<d['contact_end']:raise ValueError('Invalid contact annotation')


def validate_binding(spec,replay,annotation):
    if replay.get('task') != spec['task'] or Path(spec['dataset']).name != spec['task'] or replay.get('episode') != annotation['decision']['episode'] or replay.get('mode') != 'original_token' or replay.get('object_delta_xy_m') != [0,0]:
        raise ValueError('Replay task/episode/interface/scene does not match executed source')
    baseline=Path(spec['baseline']);ann_path=Path(spec['annotation'])
    manifest=json.loads((ann_path.parents[2]/'annotation.json').read_text())
    if Path(manifest['decision']).resolve()!=ann_path.resolve() or Path(manifest['source_result']).resolve()!=(baseline/'result.json').resolve() or manifest['source_sha256']!=file_hash(baseline/'result.json'):
        raise ValueError('Annotation manifest does not bind this replay')
    request=json.loads(Path(annotation['origin']['request']).read_text());content=request['input'][0]['content']
    if isinstance(content,list):content=next(c['text'] for c in content if c['type']=='input_text')
    evidence=json.loads(content)['evidence']
    if evidence.get('metrics')!=replay['evidence_summary'] or evidence.get('frames')!=replay['source_frames']+replay['tail_policy_ticks'] or evidence.get('contact_frames')!=replay['phase_candidates']['contact_frames']:
        raise ValueError('API annotation evidence differs from replay')
    contract=json.loads((baseline/'experiment.json').read_text())
    for path in source_paths(spec['dataset'],replay['episode']).values():
        if contract['preflight']['evidence_sha256'].get(str(path))!=file_hash(path):raise ValueError('Replay source hash differs from current dataset')


def retain_failed_attempt(history,folder,number,error):
    if any(h['attempt']==number for h in history):return
    cache=Path(folder)/'physics.result.json'
    if not cache.exists():return
    r=json.loads(cache.read_text());ip=Path(r['experiment_path'])/'inspection/inspection.json'
    ins=json.loads(ip.read_text()) if ip.exists() else {'video':str(ip.parent/'review_zh.mp4'),'rendering_and_inspection_wall_s':None}
    history.append({'attempt':number,'physical_result':r,'inspection':ins,'inspection_or_review_error':error,'quality_accepted':False})


def compact_case(case):
    case=Path(case);r=json.loads((case/'result.json').read_text());metrics=json.loads((case/'execution/metrics.json').read_text())
    evidence={k:r[k] for k in ('experiment_id','success','independent_success','failure','coverage','mode','object_delta_xy_m','source_frames','tail_policy_ticks','original_token_steps','wrist_error_rmse_m','max_downward_force_N')}
    evidence['native_thresholds']={k:r['native_acceptance'][k] for k in ('parameters','constants')}
    evidence['first']=metrics[0];evidence['last']=metrics[-1]
    evidence['max_hold_counts']=r['evidence_summary']['max_hold_counts'];evidence['longest_press_s']=r['evidence_summary']['longest_press_s'];evidence['peak_hand_object_N']=r['evidence_summary']['peak_hand_object_N']
    evidence['first_native_success_frame']=next((i for i,m in enumerate(metrics) if m['native_success']),None)
    evidence['max_object_delta_xyz_m']={k:(np.asarray([m['object_xyz'][k] for m in metrics])-np.asarray(metrics[0]['object_xyz'][k])).max(axis=0).tolist() for k in metrics[0]['object_xyz']}
    return evidence


def keyframes(case,contact_start):
    from PIL import Image,ImageDraw,ImageFont
    from .render_bootstrap import FONT
    case=Path(case);folder=case/'inspection';out=folder/'api_keyframes.jpg'
    if out.exists():return str(out)
    r=json.loads((case/'result.json').read_text());n=(r['physics_steps']+3)//4
    picks=sorted(set([min(contact_start,n),min((contact_start+n)//2,n),n]))
    canvas=Image.new('RGB',(960,330*len(picks)));font=ImageFont.truetype(FONT,16)
    for index,frame in enumerate(picks):
        p=folder/f'api_frame_{frame:04d}.jpg'
        if not p.exists():subprocess.run(['ffmpeg','-v','error','-i',str(folder/'review_zh.mp4'),'-vf',f'select=eq(n\\,{frame})','-vsync','0','-frames:v','1',str(p)],check=True)
        canvas.paste(Image.open(p).resize((960,300)),(0,index*330));ImageDraw.Draw(canvas).text((5,index*330+302),f'{case.name} | video frame {frame} (initial=0)',font=font,fill='white')
    canvas.save(out,quality=88);return str(out)


def decision(provider,root,context,params,source_frames,shift=None,current=None,retry_left=False):
    root.mkdir(parents=True,exist_ok=True);API_CURSOR[str(root)]=0
    def check(d,episodes):
        validate_direct(d,episodes)
        legal=['run','reject','needs_tool'] if shift is None else ['accept','reject','needs_tool']+(['retry'] if retry_left else [])
        if d['action'] not in legal:raise ValueError('Action outside this stage budget')
        if d['contact_start']!=params['contact_start'] or validate_guard(d.get('safety_guard'))!=validate_guard(params.get('safety_guard')):raise ValueError('Contact start and guard are fixed')
        if not d['contact_start']<=d['contact_end']<source_frames+round(d['tail_s']/.02):raise ValueError('Contact end outside extended source')
        if shift is None and len(d['object_shifts_xy_m'])!=2:raise ValueError('Plan needs two distinct shifts')
        if shift is not None:
            if d['object_shifts_xy_m']!=[shift]:raise ValueError('Review/repair must retain current object shift')
            if d['action']=='retry' and same_execution(d,current):raise ValueError('Repair must change an executable control parameter')
            if d['action']=='accept' and not accepted(context['evidence'],'accept'):raise ValueError('Physical failure cannot be accepted')
        return d
    return api_decide(provider,root,context,[params['episode']],tools=TOOLS,validator=check,skill_path=SKILL)


def physical_storage_root(attempt_folder):
    return Path(attempt_folder).parents[1]


def attempt(spec,folder,d,path,shift):
    folder.mkdir(parents=True,exist_ok=True);verify_origin(path);cached=folder/'physics.result.json'
    if cached.exists():r=json.loads(cached.read_text())
    else:
        marker=folder/'physics.started.json'
        if marker.exists():raise RuntimeError('Unresolved started attempt requires evidence audit; no automatic physical repeat')
        save(marker,{'started_at':now(),'decision':str(path),'object_shift':shift})
        r=physical_case(spec['dataset'],d['episode'],physical_storage_root(folder),'token_prefix_mode1',shift,decision_path=path);save(cached,r)
    if r['object_delta_xy_m']!=shift or r['decision_path']!=str(path):raise ValueError('Cached physics differs from authorized execution')
    case=Path(r['experiment_path']);diagnose(case,spec['dataset'])
    inspection_path=case/'inspection/inspection.json'
    if not inspection_path.exists():
        if inspection_path.parent.exists():inspection_path.parent.rename(case/('inspection.incomplete.'+str(time.time_ns())))
        inspect_case(case,inspection_path.parent)
    ins=json.loads(inspection_path.read_text());e=compact_case(case);e['contact_pairs']=ins['contact_pairs'];e['inspection']={'review_images':[keyframes(case,d['contact_start'])]}
    return r,e,ins


def run(config,provider_config):
    install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root,'study'):
        provider=configured_provider(provider_config,root/'.api_rotation.json');start=now();tick=time.monotonic();results=[]
        for spec in config['tasks']:
            tr=root/spec['task'];tr.mkdir(exist_ok=True);out=tr/'result.json'
            if out.exists() and json.loads(out.read_text())['status']=='evaluated':results.append(json.loads(out.read_text()));continue
            if out.exists():out.rename(tr/('result.history.'+str(time.time_ns())+'.json'))
            candidates=[]
            try:
                ann=verify_origin(Path(spec['annotation']));params=ann['decision'];replay=json.loads((Path(spec['baseline'])/'result.json').read_text());require_prerequisites(replay,ann);validate_binding(spec,replay,ann)
                context={'stage':'direct_plan','task':spec['task'],'reference_parameters':params,'replay':compact_case(spec['baseline']),'zero_diagnostic':compact_case(spec['zero_case']),'budget':{'candidates':2,'max_attempts_per_candidate':2,'object_shift_norm_max_m':.02},'instructions':'Zero diagnostic failure is NOT a prerequisite blocker. Select two nonzero small object shifts and executable initial parameters. Preserve episode, contact_start and guard. For kick/sit disclose no foot/pelvis translation capability.'}
                plan,plan_path=decision(provider,tr/'plan',context,params,replay['source_frames'])
                if plan['action']!='run':
                    result={'task':spec['task'],'status':'blocked','plan':plan,'candidates':[]};save(out,result);results.append(result);continue
                for index,shift in enumerate(plan['object_shifts_xy_m']):
                    cr=tr/f'candidate_{index:02d}';history=[];d=plan;path=plan_path
                    try:
                        for number in range(2):
                            ar=cr/f'attempt_{number:02d}';r,e,ins=attempt(spec,ar,d,path,shift)
                            history.append({'attempt':number,'physical_result':r,'inspection':ins,'executed_decision':str(path)})
                            save(cr/'progress.json',history)
                            review,review_path=decision(provider,ar/'review',{'stage':'direct_review','task':spec['task'],'object_shift_xy_m':shift,'executed_parameters':d,'evidence':e,'retry_left':number==0,'instructions':'Accept only successful task with plausible contact/retention. If failed, choose an evidence-based repair within registered channels or needs_tool/reject. Return exactly this single object shift. No zero success gate.'},params,replay['source_frames'],shift,d,number==0)
                            history[-1].update(review=review,review_path=str(review_path),quality_accepted=accepted(r,review['action']));save(ar/'result.json',history[-1]);save(cr/'progress.json',history)
                            if review['action']!='retry':break
                            d=review;path=review_path
                        candidate={'object_shift_xy_m':shift,'status':'evaluated','success':any(h.get('quality_accepted',False) for h in history),'attempts':history}
                    except Exception as exc:
                        retain_failed_attempt(history,ar,number,str(exc))
                        candidate={'object_shift_xy_m':shift,'status':'error','error':str(exc),'traceback':traceback.format_exc(),'success':False,'attempts':history}
                    save(cr/'result.json',candidate);candidates.append(candidate);save(tr/'progress.json',candidates)
                result={'task':spec['task'],'status':'error' if any(c['status']=='error' for c in candidates) else 'evaluated','candidates':candidates}
            except Exception as exc:result={'task':spec['task'],'status':'error','error':str(exc),'traceback':traceback.format_exc(),'candidates':candidates}
            save(out,result);results.append(result);save(root/'progress.json',results)
        attempts=[h for t in results for c in t['candidates'] for h in c['attempts']];candidates=[c for t in results for c in t['candidates']]
        api=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')]
        export=[{'task':h['physical_result']['task'],'experiment_path':h['physical_result']['experiment_path'],'trajectory':str(Path(h['physical_result']['experiment_path'])/'execution/trajectory.npz'),'controller':str(Path(h['physical_result']['experiment_path'])/'execution/controller.npz'),'video':h['inspection']['video'],'training_ready':False,'temporal_review':'pending'} for h in attempts if h.get('quality_accepted')]
        summary={'study_id':root.name,'started_at':start,'ended_at':now(),'invocation_wall_s':time.monotonic()-tick,'tasks':results,'planned_candidate_budget':2*len(config['tasks']),'candidate_count':len(candidates),'final_successes':sum(c['success'] for c in candidates),'first_attempt_successes':sum(bool(c['attempts'] and c['attempts'][0].get('quality_accepted')) for c in candidates),'retry_candidates':sum(len(c['attempts'])>1 for c in candidates),'retry_recovered':sum(len(c['attempts'])>1 and not c['attempts'][0].get('quality_accepted') and c['success'] for c in candidates),'physical_attempts':len(attempts),'api_requests':len(api),'known_tokens':sum((a.get('usage') or {}).get('total_tokens',0) for a in api),'unknown_usage_calls':sum(a.get('usage') is None for a in api),'api_wall_s':sum(a.get('wall_s') or 0 for a in api),'simulation_wall_s':sum(h['physical_result']['simulation_wall_s'] for h in attempts),'rendering_inspection_wall_s':sum(h['inspection'].get('rendering_and_inspection_wall_s') or 0 for h in attempts),'unknown_rendering_durations':sum(h['inspection'].get('rendering_and_inspection_wall_s') is None for h in attempts),'unresolved_started_attempts':[str(p) for p in root.glob('**/physics.started.json') if not (p.parent/'physics.result.json').exists()],'exports':export}
        save(root/'summary.json',summary);save(root/'accepted_manifest.json',export)
        lines=[f"# {root.name}",f"{start} — {summary['ended_at']}",'','Zero reuse is diagnostic only. Original-token prefix retained; mode1 through end. No native criteria change.','', '|Task|Object XY m|Attempt|Native|API|Quality|Video|','|---|---|---|---|---|---|---|']
        for t in results:
            for c in t['candidates']:
                for h in c['attempts']:
                    r=h['physical_result'];lines.append(f"|{t['task']}|{c['object_shift_xy_m']}|{h['attempt']}|{r['success']}|{h.get('review',{}).get('action','error')}|{h.get('quality_accepted',False)}|[MP4]({h['inspection']['video']})|")
                if c['status']=='error':lines.append('\n'+c['error'])
            if t['status']=='blocked' or t.get('error'):lines.append('\n'+t['task']+': '+str(t.get('error',t.get('plan'))))
        lines+=['',json.dumps({k:v for k,v in summary.items() if k not in ('tasks','exports')},ensure_ascii=False,indent=2)]
        (root/'REPORT.md').write_text('\n'.join(lines))
        from .direct_export import export
        export(root)
        return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),json.loads(Path(a.provider).read_text()))
