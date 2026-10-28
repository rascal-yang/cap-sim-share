#!/usr/bin/env python3
"""Compact real API decisions across eight positions; all nine credential slots used."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from humanoid_mimicgen.augmentation.provider import ResponsesProvider
from humanoid_mimicgen.augmentation.api_pool import RotatingResponsesProvider
from humanoid_mimicgen.workflow.contracts import now,file_hash,digest,canonical


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp");temporary.write_bytes(canonical(value));temporary.replace(path)


def validate_decision(raw,task):
    if raw.get('status')!='completed':raise ValueError('Incomplete API response')
    text='\n'.join(c['text'] for x in raw.get('output',[]) if x.get('type')=='message' for c in x.get('content',[]) if c.get('type')=='output_text')
    value=json.loads(text)
    if set(value)!={'action','feedback_gain','rationale'} or value['action']!='execute':raise ValueError('Decision schema')
    gain=value['feedback_gain']
    if type(gain) not in (int,float) or not 0<=gain<=3 or not isinstance(value['rationale'],str) or not value['rationale'].strip():raise ValueError('Bounded gain/rationale')
    if task=='chair' and gain!=0:raise ValueError('Chair has no active wrist feedback; gain must0')
    return value


def main():
    import tomllib
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--credentials-pool',nargs=9,required=True);args=parser.parse_args();root=args.root.resolve();repo=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'provider.json').read_text());profiles=[]
    for i,name in enumerate(args.credentials_pool):
        path=Path(name);entry=tomllib.loads(path.read_text())['model_providers']['OpenAI'];key=f'FOUR_TASK_{i}_KEY';actor=f'FOUR_TASK_{i}_ACTOR';os.environ[key]=entry['experimental_bearer_token'];os.environ[actor]=entry['http_headers']['x-openai-actor-authorization'];profiles.append(dict(label=path.name,config=cfg|dict(base_url=entry['base_url'],key_env=key,headers_env={'x-openai-actor-authorization':actor})))
    save(root/'api_pool.json',profiles);provider=RotatingResponsesProvider.from_profiles(cfg,profiles,root/'api_routing.json');plans=[Path(x) for x in json.loads((root/'plans.json').read_text())];rows={};attempts=[];clock=time.monotonic();started=now();api=root/'api';api.mkdir()
    instructions='You choose a bounded pre-encoder proportional feedback gain for one physically simulated task at a new lateral position. Original successful full-body operation is translated with target, then precision navigation starts from original far robot state; all SONIC mode1, no original-token prefix, no images. This is trajectory retargeting. Return ONLY JSON with action="execute", feedback_gain number0..3, rationale concise. Chair has no active wrist feedback: gain0. Other tasks feedback applies active hand(s); source orientation/fingers preserved, no other policy edits. A physical failure is valid evidence; do not invent success.'
    def request(context,instruction,kind):
        number=len(attempts);folder=api/f'{number:03d}';folder.mkdir();payload=dict(input=[dict(role='user',content=json.dumps(context,ensure_ascii=False))],tools=[],instructions=instruction,decision_protocol='json_action');save(folder/'request.json',payload);t=time.monotonic();a=dict(number=number,kind=kind,started_at=now(),status='started');save(folder/'attempt.json',a)
        try:
            raw=provider.respond(payload['input'],[],instruction);save(folder/'response.json',raw);a.update(status='responded',usage=raw.get('usage'),reported_model=raw.get('model'));return folder,raw
        except Exception as exc:a.update(status='failed',reason_code=getattr(exc,'code',type(exc).__name__));return folder,None
        finally:
            a.update(ended_at=now(),wall_s=time.monotonic()-t,credential_slot=provider.last_provider_label,credential_sequence=provider.last_provider_sequence,credential_endpoint=provider.last_provider_endpoint);save(folder/'attempt.json',a);attempts.append(a);save(root/'api_attempts.json',attempts)
    def condition(plan):
        p=json.loads(plan.read_text());task=p['execution_task']['name'];ident=plan.parent.name;context=dict(task=task,offset_world_m=[0,p['offset_m'],0],candidate_id=ident,baseline_native_or_approved_success=True,active_hands=p['execution_task']['strategy']['bindings']['hands'],suggested_gain=2 if task=='apple' else 0,task_criterion=p['execution_task']['acceptance_profile'],trial='one physical attempt; no tuning; collision separately measured',source_operation_samples=json.loads(Path(p['generated_reference']['metadata_path']).read_text())['operation_interval'])
        folder,raw=request(context,instructions,'condition_'+ident)
        row=dict(condition=ident,task=task,offset_m=p['offset_m'],plan=str(plan),api_folder=str(folder),status='api_blocked')
        if raw:
            try:decision=validate_decision(raw,task)
            except Exception as exc:row.update(status='api_invalid',reason=str(exc));rows[ident]=row;save(root/'rows.json',list(rows.values()));return
            save(folder/'decision.json',decision)
            evidence={str(x):file_hash(x) for x in [root/'protocol.json',root/'provider.json',root/'api_pool.json',folder/'request.json',folder/'response.json',folder/'attempt.json',folder/'decision.json',Path(__file__).resolve(),plan]}
            proof=dict(transport='responses',decision_protocol='json_action',native_function_calling=False,perception_mode='privileged_only',model=cfg['model'],reported_model=raw.get('model'),call_id=digest(raw),credential_slot=provider.last_provider_label,credential_sequence=provider.last_provider_sequence,api_calls_before_launch=len(attempts),evidence_sha256=evidence,decision=decision)
            job=root/(ident+'_execution_request.json');save(job,dict(root=str(root),plan=str(plan),decision=decision,provenance=proof));t=time.monotonic()
            with (folder/'physical.log').open('w') as log:run=subprocess.run([str(repo/'scripts/humano_python.sh'),'-m','humanoid_mimicgen.augmentation.four_task_navigation','run',str(job)],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
            row.update(status='executed' if run.returncode==0 else 'preparation_or_execution_error',execution_wall_s=time.monotonic()-t,decision=decision)
            if job.with_suffix('.result.json').exists():row['result']=json.loads(job.with_suffix('.result.json').read_text())
        rows[ident]=row;save(root/'rows.json',list(rows.values()));print(ident,row['status'],flush=True)
    for plan in plans:condition(plan)
    # Ninth real request sees actual results/errors, not a redundant health probe.
    folder,raw=request(dict(results=list(rows.values()),scope='Eight primary condition attempts; later retries may unblock API-only failures. Explain only actual evidence and distinguish blockers from physical failures.'),'Return one concise JSON object with findings and limitations for these measured four-task position trials. No new physical action is requested. Do not count API outages as task failures.','measured_batch_reflection')
    if raw:save(root/'api_batch_reflection.json',dict(response_path=str(folder/'response.json'),response_sha256=file_hash(folder/'response.json')))
    # One further pass only for conditions with no physical launch; never rerun completed/error physical jobs.
    for plan in plans:
        if rows[plan.parent.name]['status'] in ('api_blocked','api_invalid'):condition(plan)
    save(root/'batch_timing.json',dict(started_at=started,ended_at=now(),wall_s=time.monotonic()-clock,api_attempts=len(attempts),actual_slots=sorted({a['credential_slot'] for a in attempts}),physical_records=sum(r['status']=='executed' for r in rows.values())))


if __name__=='__main__':main()
