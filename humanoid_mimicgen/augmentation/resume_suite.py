"""Resume a quality-blocked task only after an API-authored supplementary acceptance."""
import argparse,json,time
from pathlib import Path
from . import quality_suite as q
from .agent import save
from .provider import ResponsesProvider
from ..workflow.contracts import now,file_hash


def run(config_path,provider_path):
    cfg=json.loads(Path(config_path).read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    acceptance=q.verify_origin(cfg['acceptance_decision']);zero=json.loads(Path(cfg['zero_result']).read_text())
    if acceptance['decision']['action']!='accept':raise ValueError('Supplement did not authorize continuation')
    q.gate('zero',zero,acceptance['decision'])
    matched=q.verify_origin(zero['decision_path'])['decision'];provider=ResponsesProvider(json.loads(Path(provider_path).read_text()))
    history=[];rows=[];started=now();tick=time.monotonic()
    context={'task':zero['task'],'accepted_zero':q.short_result(zero),'supplementary_api_acceptance':acceptance,'matched_execution_parameters':matched,'annotation_note':'Supplement contact_start/end revises annotation, not the zero case control parameters. Keep execution parameters identical to the actually successful zero case; no repair is requested.','resolved_evidence':[json.loads(Path(x).read_text()) for x in cfg.get('resolved_reviews',[])],'budget':{'attempts_per_offset':1,'offsets':cfg['offsets']}}
    for shift in cfg['offsets']:
        d,p=q.api_decide(provider,root,{**context,'stage':'augmentation_plan','object_delta_xy_m':shift,'history':history,'requirement':'run with matched execution parameters; otherwise reject/needs_tool. Successful zero and supplementary review were both accepted by API.'},[zero['episode']])
        history.append({'stage':'plan','shift':shift,'decision':d})
        if d['action']!='run':break
        if not q.same_execution(d,matched):raise ValueError('Offset parameters differ from accepted zero')
        q.verify_origin(p);save(p.parent/'physics.queued.json',{'at':now(),'shift':shift})
        r=q.execute(cfg['dataset'],root,'augmentation',d,p,shift)
        save(p.parent/'physics.result.json',r);rows.append(r)
        review,rp=q.api_decide(provider,root,{**context,'stage':'augmentation','object_delta_xy_m':shift,'result':q.short_result(r),'history':history,'requirement':'Review this measured result. Accept only if appropriate; reject or request a capability otherwise. No physical retry authorized by this resume budget.'},[zero['episode']])
        if review['action']=='accept':q.gate('augmentation',r,review)
        history.append({'stage':'review','shift':shift,'decision':review})
        save(root/'progress.json',{'cases':rows,'history':history})
    result={'task':zero['task'],'started_at':started,'ended_at':now(),'wall_s':time.monotonic()-tick,'supplementary_acceptance':cfg['acceptance_decision'],'cases':rows,'history':history,'training_ready':False,'status':'completed' if len(rows)==len(cfg['offsets']) else 'blocked'}
    save(root/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);a=p.parse_args();r=run(a.config,a.provider);print(json.dumps({'status':r['status'],'cases':len(r['cases'])}))
