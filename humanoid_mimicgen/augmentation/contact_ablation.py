"""Serial, API-owned matched contact-entry comparison. No implicit repairs."""
import argparse,json,time,traceback
from pathlib import Path
from .agent import save
from .quality_suite import api_decide,verify_origin,API_CURSOR
from .suite_provider import configured_provider
from .suite_physics import physical_case,validate_guard
from .suite_lock import exclusive
from .suite_inspect import inspect_case
from .trajectory_diagnostics import diagnose
from .suite_evidence import install
from ..workflow.contracts import now


def validate_matched(decision, params):
    for key in ('episode','tail_s','wrist_delta_m','contact_start','contact_end'):
        if decision[key] != params[key]:raise ValueError('API changed matched parameter '+key)
    if validate_guard(decision.get('safety_guard')) != validate_guard(params.get('safety_guard')):
        raise ValueError('API changed matched parameter safety_guard')


def run(config,provider_config,resume_errors=False):
    with exclusive(config['output_root'],'study'):
        return _run(config,provider_config,resume_errors)


def _run(config,provider_config,resume_errors=False):
    install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if resume_errors:
        for name in ('summary.json','REPORT.md','progress.json'):
            path=root/name
            if path.exists():path.rename(root/(name+'.history.'+str(time.time_ns())))
    provider=configured_provider(provider_config,root/'.api_rotation.json');started=now();tick=time.monotonic();results=[]
    for spec in config['tasks']:
        taskroot=root/spec['task'];taskroot.mkdir(exist_ok=True)
        cached=taskroot/'result.json'
        if cached.exists():
            previous=json.loads(cached.read_text())
            if not (resume_errors and previous['status']=='error'):
                results.append(previous);continue
            cached.rename(taskroot/('result.error.'+str(time.time_ns())+'.json'))
        API_CURSOR[str(taskroot)]=0
        try:
            annotation=verify_origin(Path(spec['annotation']));params=annotation['decision']
            baseline=json.loads((Path(spec['baseline'])/'result.json').read_text())
            if not baseline['success']:raise ValueError('Baseline must pass native replay')
            context={'stage':'contact_ablation_plan','legal_actions':['run','needs_tool','reject'],'task':spec['task'],'matched_parameters':params,'baseline':baseline,'annotation':annotation,'budget':{'fresh_physical_arms':2,'retries':0,'object_shift_m':[0,0]},'instructions':'Authorize full mode1 and token-prefix then mode1 through end as a diagnostic comparison. Preserve episode, tail_s, wrist_delta_m=[0,0,0], guard and contact bounds exactly from matched_parameters. No repairs. Each arm gets independent review; no downstream augmentation. Explain annotation limitations, especially whether prefix already completes contact.'}
            d,decision=api_decide(provider,taskroot,context,[params['episode']])
            if d['action']!='run':
                blocked={'task':spec['task'],'status':'blocked','decision':d};save(taskroot/'result.json',blocked);results.append(blocked);continue
            validate_matched(d,params)
            arms=[]
            for arm_index,mode in enumerate(('cartesian_source_legs','token_prefix_mode1')):
                arm=None
                try:
                    marker=taskroot/(mode+'.json')
                    if marker.exists():arm=json.loads(marker.read_text())
                    else:
                        started_marker=taskroot/(mode+'.started.json')
                        if started_marker.exists():raise RuntimeError('Unresolved started physics case; audit before resume')
                        save(started_marker,{'at':now(),'decision':str(decision)})
                        arm=physical_case(spec['dataset'],d['episode'],taskroot,mode,decision_path=decision);save(marker,arm)
                    case=Path(arm['experiment_path'])
                    diag=diagnose(case,spec['dataset'])
                    inspection_path=case/'inspection'/'inspection.json'
                    if not inspection_path.exists() and inspection_path.parent.exists():
                        inspection_path.parent.rename(case/('inspection.incomplete.'+str(time.time_ns())))
                    inspection=json.loads(inspection_path.read_text()) if inspection_path.exists() else inspect_case(case,case/'inspection')
                    API_CURSOR[str(taskroot)]=1+arm_index
                    review,review_path=api_decide(provider,taskroot,{'stage':'contact_ablation_review','legal_actions':['accept','reject','needs_tool'],'task':spec['task'],'mode':mode,'evidence':{'physical_result':arm,'diagnostics':diag,'inspection':inspection},'instructions':'Review this diagnostic arm only. No retry budget or downstream augmentation. Explain earliest measured divergence, whether prefix already succeeds, final retention, and capability requests. Keep matching parameters unchanged in response.'},[d['episode']])
                    arms.append({'mode':mode,'physical_result':arm,'diagnostics':diag,'inspection':inspection,'review':review,'quality_accepted':bool(arm['success'] and arm.get('independent_success',False) and arm.get('coverage')==1 and not arm.get('failure') and review['action']=='accept'),'review_path':str(review_path)})
                    save(taskroot/(mode+'_review.json'),arms[-1])
                except Exception as exc:
                    failed={'mode':mode,'status':'error','error':str(exc),'traceback':traceback.format_exc()}
                    if arm is not None:failed['physical_result']=arm
                    arms.append(failed);save(taskroot/(mode+'_error.json'),failed)
            result={'task':spec['task'],'status':'error' if any(a.get('status')=='error' for a in arms) else 'measured','arms':arms}
        except Exception as exc:
            result={'task':spec['task'],'status':'error','error':str(exc),'traceback':traceback.format_exc()}
        save(taskroot/'result.json',result);results.append(result);save(root/'progress.json',results)
    usage=[]
    for path in root.glob('*/api/*/attempt_*.json'):
        row=json.loads(path.read_text());usage.append(row)
    total={'study_id':root.name,'started_at':started,'ended_at':now(),'total_wall_s':time.monotonic()-tick,'tasks':results,'api_calls':len(usage),'known_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in usage),'unknown_usage_calls':sum(r.get('usage') is None for r in usage),'api_wall_s':sum(r.get('wall_s') or 0 for r in usage)}
    save(root/'summary.json',total)
    lines=[f"# {root.name}",f"{started} — {total['ended_at']}",'','Fresh matched arms; zero object shift; source hands unchanged; original token prefix disclosed. No training-ready claim.','', '|Task|Mode|Native|API|Prefix ticks|Final native|Video|','|---|---|---|---|---|---|---|']
    for r in results:
        for a in r.get('arms',[]):
            if a.get('status')=='error':
                lines.append(f"|{r['task']}|{a['mode']}|unknown|error|unknown|unknown|{a['error']}|");continue
            p=a['physical_result'];lines.append(f"|{r['task']}|{a['mode']}|{p['success']}|{a['review']['action']}|{p['original_token_steps']}|{a['diagnostics']['native_success_at_end']}|[MP4]({p['experiment_path']}/inspection/review_zh.mp4)|")
        if r['status']!='measured':lines.append(f"\n{r['task']}: {r.get('error',r['status'])}")
    lines+=['',f"API calls: {len(usage)}; known tokens: {total['known_tokens']}; unknown usage: {total['unknown_usage_calls']}; total wall: {total['total_wall_s']:.1f}s. Planning/API, physics and render durations are preserved separately in individual records; unmeasured components unknown."]
    (root/'REPORT.md').write_text('\n'.join(lines));return total

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);p.add_argument('--resume-errors',action='store_true');a=p.parse_args();run(json.loads(Path(a.config).read_text()),json.loads(Path(a.provider).read_text()),resume_errors=a.resume_errors)
