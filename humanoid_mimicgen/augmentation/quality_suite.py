"""Stage-gated all-task API orchestration. Concrete repairs originate in API calls."""
import argparse
import json
import math
from pathlib import Path
import time
from . import suite_physics as physics
from .agent import save
from .provider import ResponsesProvider
from .suite_lock import exclusive_task,exclusive
from ..workflow.contracts import now, file_hash, canonical

API_CURSOR={}

PROPERTIES={
    'action':{'type':'string','enum':['run','accept','retry','needs_tool','reject']},
    'rationale':{'type':'string'},'hypothesis':{'type':'string'},
    'episode':{'type':'integer'},'tail_s':{'type':'number'},
    'wrist_delta_m':{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3},
    'contact_start':{'type':'integer','minimum':-1},'contact_end':{'type':'integer','minimum':-1},'safety_guard':{'type':['object','null'],'properties':{'root_height_min_m':{'type':'number','minimum':.1,'maximum':.5},'root_tilt_max_deg':{'type':'number','minimum':30,'maximum':100}},'required':['root_height_min_m','root_tilt_max_deg'],'additionalProperties':False}}
TOOLS=[{'type':'function','name':'decide','description':'Review current stage quality and explicitly select next action or repair. No physical action occurs on acceptance alone.','strict':True,'parameters':{'type':'object','properties':PROPERTIES,'required':list(PROPERTIES),'additionalProperties':False}}]


def validate_decision(d,episodes):
    if set(d) not in (set(PROPERTIES),set(PROPERTIES)-{'safety_guard'}) or d['action'] not in PROPERTIES['action']['enum']:raise ValueError('Invalid decision fields')
    if type(d['episode']) is not int or d['episode'] not in episodes:raise ValueError('Episode outside budget')
    if not isinstance(d['rationale'],str) or not d['rationale'].strip() or not isinstance(d['hypothesis'],str):raise ValueError('Missing rationale')
    tail=d['tail_s'];w=d['wrist_delta_m']
    if not isinstance(tail,(int,float)) or not math.isfinite(tail) or not 0<=tail<=3 or abs(tail/.02-round(tail/.02))>1e-6:raise ValueError('Tail outside 0..3s or not 20ms aligned')
    if not isinstance(w,list) or len(w)!=3 or any(not isinstance(x,(float,int)) or not math.isfinite(x) for x in w) or sum(x*x for x in w)>.05**2:raise ValueError('Wrist offset exceeds 5cm budget')
    placeholder=d['contact_start']==d['contact_end']==-1 and not any(w)
    if any(type(d[k]) is not int for k in ('contact_start','contact_end')) or not (placeholder or 0<=d['contact_start']<=d['contact_end']):raise ValueError('Invalid interval')
    physics.validate_guard(d.get('safety_guard'))
    return d


def same_execution(a,b):
    keys=['episode','tail_s','wrist_delta_m']
    if any(a['wrist_delta_m']) or any(b['wrist_delta_m']):keys+=['contact_start','contact_end']
    return all(a[k]==b[k] for k in keys) and physics.validate_guard(a.get('safety_guard'))==physics.validate_guard(b.get('safety_guard'))


def gate(stage,result,d):
    if stage in ('replay','zero','augmentation'):
        if not (result.get('success') and result.get('independent_success') and result.get('coverage')==1 and not result.get('failure')):raise ValueError('Physical quality gate failed')
    if stage=='annotation':
        a,z=d['contact_start'],d['contact_end'];contacts=result['contact_frames']
        if not result['success'] or not 0<=a<z<result['frames'] or not contacts or not any(a<=i<=z for i in contacts):raise ValueError('Annotation quality gate failed')
    return True


def api_decide(provider,root,context,episodes,*,tools=None,validator=None,skill_path=None):
    """Only persisted completed API function calls become executable decisions."""
    api=root/'api';api.mkdir(exist_ok=True)
    index=API_CURSOR.get(str(root),0);API_CURSOR[str(root)]=index+1
    folder=api/f'{index:04d}';folder.mkdir(exist_ok=True)
    selected_skill=Path(skill_path) if skill_path else physics.SKILL
    selected_tools=TOOLS if tools is None else tools
    selected_validator=validate_decision if validator is None else validator
    instructions=selected_skill.read_text();items=[{'role':'user','content':canonical(context).decode()}]
    image_paths=[]
    def find_images(x):
        if isinstance(x,dict):
            if 'inspection' in x:image_paths.extend(x['inspection'].get('review_images',[]))
            for v in x.values():find_images(v)
        elif isinstance(x,list):
            for v in x:find_images(v)
    find_images(context.get('evidence',context))
    if image_paths:
        import base64
        content=[{'type':'input_text','text':items[0]['content']}]
        for image_path in list(dict.fromkeys(image_paths))[:2]:content.append({'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode(Path(image_path).read_bytes()).decode(),'detail':'high'})
        items=[{'role':'user','content':content}]
    request={'input':items,'tools':selected_tools,'instructions':instructions,'model':provider.config['model'],'reasoning_effort':provider.config.get('reasoning_effort')}
    if (folder/'decision.json').exists():
        if json.loads((folder/'request.json').read_text())!=request:raise ValueError('Resume context changed; use fresh study')
        record=verify_origin(folder/'decision.json');return record['decision'],folder/'decision.json'
    save(folder/'request.json',request)
    prior=len(list(folder.glob('attempt_*.json')))
    schedule=provider.attempt_providers() if hasattr(provider,'attempt_providers') else iter([('primary',provider)]*3)
    validation_failures=0;last_error=None
    for attempt,(slot,transport) in enumerate(schedule,start=prior):
        raw=None
        endpoint=transport.config.get('base_url');sequence=getattr(provider,'routing_sequence',None)
        tick=time.monotonic();audit={'started_at':now(),'usage':None,'status':'started','wall_s':None,'provider_slot':slot,'base_url':endpoint,'routing_sequence':sequence};save(folder/f'attempt_{attempt}.json',audit)
        try:
            actual_request=folder/f'request_attempt_{attempt}.json';save(actual_request,request|{'input':items,'provider_slot':slot,'base_url':endpoint,'routing_sequence':sequence})
            raw=transport.respond(items,selected_tools,instructions)
            save(folder/f'response_{attempt}.json',raw);audit['usage']=raw.get('usage')
            if raw.get('status')!='completed':raise RuntimeError('Incomplete API response')
            calls=[x for x in raw['output'] if x['type']=='function_call']
            if len(calls)!=1 or calls[0]['name']!='decide':raise ValueError('Expected exactly one decide call')
            d=selected_validator(json.loads(calls[0]['arguments']),episodes)
            origin={'kind':'api','provider_slot':slot,'base_url':endpoint,'routing_sequence':sequence,'model':raw['model'],'reasoning_effort':provider.config.get('reasoning_effort'),'response':str(folder/f'response_{attempt}.json'),'response_sha256':file_hash(folder/f'response_{attempt}.json'),'request':str(actual_request),'request_sha256':file_hash(actual_request),'call_id':calls[0]['call_id'],'skill_sha256':file_hash(selected_skill)}
            if skill_path:origin['skill_path']=str(selected_skill.resolve())
            save(folder/'decision.json',{'decision':d,'origin':origin,'created_at':now()})
            audit['status']='completed'
            return d,folder/'decision.json'
        except Exception as exc:
            audit.update(status='error',error_type=type(exc).__name__,detail=str(exc)[:300])
            last_error=exc
            if isinstance(exc,ValueError) and raw is not None and raw.get('status')=='completed':
                validation_failures+=1
                items.extend(raw['output'])
                for call in raw['output']:
                    if call['type']=='function_call':items.append({'type':'function_call_output','call_id':call['call_id'],'output':json.dumps({'error':str(exc),'instruction':'Correct the tool arguments; no action was executed.'})})
            if validation_failures>=3:raise
        finally:
            audit.update(ended_at=now(),wall_s=time.monotonic()-tick);save(folder/f'attempt_{attempt}.json',audit)
    raise last_error


def verify_origin(path):
    record=json.loads(Path(path).read_text());o=record['origin'];d=record['decision']
    if o['kind']!='api' or file_hash(o['response'])!=o['response_sha256'] or file_hash(o['request'])!=o['request_sha256']:raise ValueError('API provenance invalid')
    raw=json.loads(Path(o['response']).read_text())
    calls=[c for c in raw['output'] if c.get('call_id')==o['call_id'] and c.get('name')=='decide']
    if raw.get('status')!='completed' or len(calls)!=1 or json.loads(calls[0]['arguments'])!=d:raise ValueError('Decision differs from API output')
    return record


def execute(dataset,root,stage,d,path,shift=(0,0)):
    verify_origin(path)
    mode='original_token' if stage=='replay' else 'cartesian_source_legs'
    if mode=='original_token' and any(d['wrist_delta_m']):raise ValueError('Original-token replay cannot apply wrist offsets')
    cached=path.parent/'physics.result.json'
    if cached.exists():
        result=json.loads(cached.read_text())
        if result['mode']!=mode or result['object_delta_xy_m']!=list(shift):raise ValueError('Physics cache parameters changed')
        return result
    marker=path.parent/'physics.started.json'
    if marker.exists():raise RuntimeError('Attempt already started; no implicit repeat')
    save(marker,{'started_at':now(),'stage':stage,'shift':shift})
    result=physics.physical_case(dataset,d['episode'],root,mode,shift,decision_path=path)
    save(cached,result)
    from .suite_inspect import inspect_case
    try:
        inspection=inspect_case(Path(result['experiment_path']),Path(result['experiment_path'])/'quality_review')
        result['inspection']={k:v for k,v in inspection.items() if k!='object_contact_events'}
        result['inspection']['review_images']=inspection.get('review_images',[inspection['sheet']])
    except Exception as exc:result['inspection_error']={'error_type':type(exc).__name__,'detail':str(exc)[:200]}
    save(cached,result);return result


def short_result(r):
    return {k:v for k,v in r.items() if k not in ('actual_session_providers','phase_candidates')}


@exclusive_task
def run_task(dataset,root,provider,episodes,max_attempts=3):
    from .suite_evidence import install
    install()
    root.mkdir(exist_ok=True);started=now();tick=time.monotonic();history=[];results=[];decisions=[]
    inventory=physics.inventory(dataset,episodes);save(root/'inventory.json',inventory)
    context={'task':dataset.name,'stage':'source','inventory':inventory,'budget':{'attempts_per_stage':max_attempts,'tail_s_max':3,'wrist_translation_norm_m_max':.05,'offsets_xy_m':[[.01,0],[-.01,0],[0,.01],[0,-.01]]},'available_capabilities':'Every physical result includes 200Hz reconstructed contact evidence and rendered keyframes. See skill. Select episode to inspect, action=run. Each subsequent stage needs API acceptance.'}
    outcome={'task':dataset.name,'started_at':started,'stages':{},'status':'running','cases':results,'decisions':decisions}
    def decide(stage,evidence):
        allowed=['accept','reject','needs_tool'] if stage in ('saved_replay','annotation') else ['run','reject','needs_tool'] if stage in ('source','replay_plan','zero_plan','augmentation_plan') else ['accept','retry','reject','needs_tool']
        for correction in range(3):
            d,p=api_decide(provider,root,{**context,'stage':stage,'legal_actions':allowed,'evidence':evidence,'history':history[-4:],'review_note':'Saved-state review accepts integrity and suitability to attempt fresh physics; fresh success is NOT required and has not been measured at that stage.'},episodes)
            if d['action'] in allowed:break
            history.append({'stage':stage,'protocol_error':f'Action {d["action"]} is not permitted here; allowed {allowed}; no action executed.'})
        else:raise ValueError('API did not select a valid stage action')
        history.append({'stage':stage,'decision':d});decisions.append(str(p))
        print(json.dumps({'task':dataset.name,'stage':stage,'decision':d},ensure_ascii=False),flush=True)
        return d,p
    def finish(status,reason):
        outcome.update(status=status,reason=reason,ended_at=now(),total_wall_s=time.monotonic()-tick)
        save(root/'result.json',outcome);return outcome
    d,p=decide('source',inventory)
    if d['action']!='run':return finish('blocked',d['rationale'])
    ep=d['episode'];saved=physics.saved_replay(dataset,ep,root)
    d,p=decide('saved_replay',saved)
    if d['action']!='accept':return finish('blocked',d['rationale'])
    if d['episode']!=ep:raise ValueError('Cannot accept evidence from another episode')
    outcome['stages']['saved_replay']='accepted'
    # Raw baseline always exact; no repair has been justified yet.
    d,p=decide('replay_plan',{'saved_replay':saved,'requirement':'Use run with tail_s=0 and zero wrist correction for the first exact physical baseline.'})
    if d['action']!='run' or d['episode']!=ep or d['tail_s'] or any(d['wrist_delta_m']):return finish('blocked','API did not authorize exact baseline')
    accepted=None
    for stage in ('replay','zero'):
        for attempt in range(max_attempts):
            try:r=execute(dataset,root,stage,d,p)
            except Exception as exc:
                failure={'success':False,'coverage':0,'independent_success':False,'error_type':type(exc).__name__,'detail':str(exc)[:350]}
                save(p.parent/'tool_error.json',failure)
                d,p=decide(stage+'_tool_error',failure)
                if d['action']!='retry':return finish('tool_error',d['rationale'])
                continue
            results.append(r);save(root/'progress.json',outcome)
            dnext,pnext=decide(stage,{'result':short_result(r),'attempt':attempt+1,'remaining_attempts':max_attempts-attempt-1,'executed_parameters':d})
            if dnext['action']=='accept':
                try:gate(stage,r,dnext)
                except ValueError as exc:return finish('invalid_acceptance',str(exc))
                if dnext['episode']!=d['episode']:return finish('invalid_acceptance','Episode changed on acceptance')
                accepted=r;outcome['stages'][stage]='accepted';break
            if dnext['action']!='retry':return finish('blocked',dnext['rationale'])
            if dnext['episode']!=d['episode']:
                if stage!='replay':
                    restarted=run_task(dataset,root/f'source_restart_{dnext["episode"]:06d}',provider,[dnext['episode']],max_attempts)
                    restarted['cases']=results+restarted['cases'];restarted['prior_source_evidence']=str(root/'progress.json');save(root/'result.json',restarted);return restarted
                saved=physics.saved_replay(dataset,dnext['episode'],root)
                ds,ps=decide('saved_replay',saved)
                if ds['action']!='accept':return finish('blocked',ds['rationale'])
            d,p=dnext,pnext
        else:return finish('budget_exhausted',stage+' failed after bounded attempts')
        if stage=='replay':
            phase=r['phase_candidates'];annotation_evidence={'success':r['success'],'frames':phase['frames'],'contact_frames':phase['contact_frames'],'metrics':r['evidence_summary'],'requirement':'Accept and choose contact_start/end covering the relevant interaction; explain in rationale. Both endpoints required.'}
            da,pa=decide('annotation',annotation_evidence)
            if da['action']!='accept':return finish('blocked',da['rationale'])
            try:gate('annotation',annotation_evidence,da)
            except ValueError as exc:return finish('invalid_annotation',str(exc))
            save(root/'annotation.json',{'source_result':str(Path(r['experiment_path'])/'result.json'),'source_sha256':file_hash(Path(r['experiment_path'])/'result.json'),'decision':str(pa),'segments':[{'name':'approach','start':0,'end':da['contact_start']},{'name':'contact','start':da['contact_start'],'end':da['contact_end']},{'name':'post_contact','start':da['contact_end'],'end':phase['frames']-1}],'shared_state_endpoints':True})
            outcome['stages']['annotation']='accepted'
            d,p=decide('zero_plan',{'accepted_replay':short_result(r),'annotation':da,'executed_replay_parameters':d,'requirement':'Select run for zero-object-offset mode1 reference reuse, or request missing tool. Explicitly preserve successful replay timing unless you justify a change.'})
            if d['action']!='run':return finish('blocked',d['rationale'])
            if d['episode']!=r['episode']:return finish('blocked','New source invalidates annotation')
    # All shifted cases inherit exactly the physically accepted zero configuration.
    zero_params=d.copy();zero_result=accepted
    for shift in ([.01,0],[-.01,0],[0,.01],[0,-.01]):
        plan,pp=decide('augmentation_plan',{'zero_result':short_result(zero_result),'required_matched_parameters':zero_params,'object_delta_xy_m':shift,'requirement':'run with exactly the accepted zero parameters; no hidden repairs across cases.'})
        if plan['action']!='run':return finish('blocked',plan['rationale'])
        if not same_execution(plan,zero_params):return finish('blocked','Unmatched augmentation configuration requires a new zero test')
        r=execute(dataset,root,'augmentation',plan,pp,shift);results.append(r);save(root/'progress.json',outcome)
        review,rp=decide('augmentation',{'result':short_result(r),'object_delta_xy_m':shift,'requirement':'Accept successful case or reject/needs_tool on failure; all cases are exported. A retry requires new zero baseline and is outside this suite budget.'})
        if review['action']=='accept':gate('augmentation',r,review)
        else:
            outcome.setdefault('rejected_offsets',[]).append({'shift':shift,'reason':review['rationale']})
            if review['action']=='needs_tool':return finish('blocked',review['rationale'])
    outcome['stages']['augmentation']='reviewed'
    return finish('completed','All declared offsets tested and reviewed; visual training acceptance pending')


def run(config_path,provider_path):
    cfg=json.loads(Path(config_path).read_text());root=Path(cfg['output_root']);root.mkdir(exist_ok=True)
    provider=ResponsesProvider(json.loads(Path(provider_path).read_text()));started=now();tick=time.monotonic();rows=[]
    for task in cfg['tasks']:
        folder=root/task
        if (folder/'result.json').exists():r=json.loads((folder/'result.json').read_text())
        else:
            try:r=run_task(Path(cfg['dataset_root'])/task,folder,provider,cfg['episodes'],cfg.get('attempts_per_stage',3))
            except Exception as exc:
                folder.mkdir(exist_ok=True);previous=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {};r={'task':task,'status':'error','reason':type(exc).__name__+': '+str(exc)[:350],'cases':previous.get('cases',[]),'stages':previous.get('stages',{}),'ended_at':now()};save(folder/'result.json',r)
        rows.append(r);save(root/'suite_progress.json',rows)
    attempts=[json.loads(p.read_text()) for p in root.glob('*/api/*/attempt_*.json')]
    cases=[c for r in rows for c in r['cases']];usage={k:sum((a.get('usage') or {}).get(k,0) for a in attempts) for k in ('input_tokens','output_tokens','total_tokens')}
    result={'experiment_id':root.name,'started_at':started,'ended_at':now(),'wall_s':time.monotonic()-tick,'tasks':rows,'api_calls':len(attempts),'known_usage':usage,'unknown_usage_attempts':sum(a.get('usage') is None for a in attempts),'api_wall_s':sum(a['wall_s'] for a in attempts),'physics_wall_s':sum(c['simulation_wall_s'] for c in cases),'physical_cases':len(cases),'physical_successes':sum(c['success'] for c in cases),'training_ready':False,'rendering_wall_s':None}
    save(root/'result.json',result)
    lines=[f'# {root.name}',f'{started} → {result["ended_at"]}','','API owns stage acceptance and concrete repairs. Local gates reject failed upstream results. Original source actions remain visible: trajectory retargeting, not demonstrated task-strategy transfer.','','|任务|状态|已验收阶段|物理成功/尝试|阻塞/结论|','|---|---|---|---|---|']
    for r in rows:lines.append(f'|{r["task"]}|{r["status"]}|{", ".join(r["stages"])}|{sum(c["success"] for c in r["cases"])}/{len(r["cases"])}|{r.get("reason", "")}|')
    lines+=['',f'API requests: {len(attempts)}; known tokens: {usage}; unknown usage attempts: {result["unknown_usage_attempts"]}.',f'API wall seconds: {result["api_wall_s"]:.3f}; physics wall seconds: {result["physics_wall_s"]:.3f}; total run wall seconds: {result["wall_s"]:.3f}. Planning implementation and rendering times unknown.', 'All records remain training_ready=false pending visual review. See each task API request/response/decision and frozen case contract.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);a=p.parse_args()
    with exclusive(json.loads(Path(a.config).read_text())['output_root'],'study'):r=run(a.config,a.provider)
    print(json.dumps({k:v for k,v in r.items() if k!='tasks'},ensure_ascii=False))
