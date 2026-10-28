"""API review of an explicitly requested read-only evidence supplement; no physics."""
import argparse,base64,json,time
from pathlib import Path
from .provider import ResponsesProvider
from .quality_suite import TOOLS,validate_decision
from .suite_physics import SKILL
from .agent import save
from ..workflow.contracts import now,file_hash,canonical


def run(config_path,provider_path):
    cfg=json.loads(Path(config_path).read_text());out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
    provider=ResponsesProvider(json.loads(Path(provider_path).read_text()));inspection=json.loads(Path(cfg['inspection']).read_text());original=json.loads(Path(cfg['requesting_decision']).read_text())
    task_result=json.loads((Path(inspection['source'])/'result.json').read_text())
    context={'stage':cfg.get('stage','supplementary_zero_review'),'previous_api_request':original,'inspection':inspection,'physical_result':task_result,'new_read_only_capability':'All saved substeps reconstructed without integration. Contact forces are reconstructed; body identities and state alignment preserved. Review attached uniform views and contact-focused frames. No physics correction was made. Decide whether this case can be accepted now and whether its evidence blocker is resolved. accept authorizes next stage, not training-ready; needs_tool/reject keeps blocked. Only call decide.'}
    content=[{'type':'input_text','text':canonical(context).decode()}]
    for name in cfg['images']:
        content.append({'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode(Path(name).read_bytes()).decode(),'detail':'high'})
    items=[{'role':'user','content':content}];instructions=SKILL.read_text()
    save(out/'request.json',{'input':items,'tools':TOOLS,'instructions':instructions,'model':provider.config['model'],'reasoning_effort':provider.config.get('reasoning_effort')})
    for attempt in range(3):
        tick=time.monotonic();audit={'started_at':now(),'usage':None}
        try:
            raw=provider.respond(items,TOOLS,instructions);save(out/f'response_{attempt}.json',raw);audit['usage']=raw.get('usage')
            if raw.get('status')!='completed':raise ValueError('Incomplete response')
            calls=[x for x in raw['output'] if x['type']=='function_call']
            if len(calls)!=1 or calls[0]['name']!='decide':raise ValueError('Expected one decision')
            d=validate_decision(json.loads(calls[0]['arguments']),[task_result['episode']])
            r={'decision':d,'origin':{'kind':'api','model':raw['model'],'reasoning_effort':provider.config.get('reasoning_effort'),'response':str(out/f'response_{attempt}.json'),'response_sha256':file_hash(out/f'response_{attempt}.json'),'request':str(out/'request.json'),'request_sha256':file_hash(out/'request.json'),'call_id':calls[0]['call_id'],'skill_sha256':file_hash(SKILL)},'inspection_sha256':file_hash(cfg['inspection']),'images':{p:file_hash(p) for p in cfg['images']},'created_at':now()};save(out/'decision.json',r);audit['status']='completed';return r
        except Exception as exc:
            audit.update(status='error',error_type=type(exc).__name__)
            if attempt==2:raise
        finally:
            audit.update(ended_at=now(),wall_s=time.monotonic()-tick);save(out/f'attempt_{attempt}.json',audit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);a=p.parse_args();print(json.dumps(run(a.config,a.provider),ensure_ascii=False))
