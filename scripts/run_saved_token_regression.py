#!/usr/bin/env python3
"""Run one independent token replay per task from a stable-seed study.

Usage: PYTHONPATH=. scripts/humano_python.sh scripts/run_saved_token_regression.py NEW_ROOT BASELINE_ROOT
NEW_ROOT must already exist and contain no task directories. No API calls or retries.
Each case freezes its contract before physical execution; source evidence is never changed.
"""
import json,time,sys
from pathlib import Path
import numpy as np
from humanoid_mimicgen.augmentation.suite_evidence import install
from humanoid_mimicgen.augmentation.suite_physics import physical_case
from humanoid_mimicgen.augmentation.saved_token_replay import extract_tokens
from humanoid_mimicgen.augmentation.suite_inspect import inspect_case
from humanoid_mimicgen.workflow.contracts import now,file_hash
from humanoid_mimicgen.augmentation.agent import save
install();root=Path(sys.argv[1]);baseline=Path(sys.argv[2]);cfg=json.loads((baseline/'config.json').read_text());results=[];start=now();tick=time.monotonic()
for spec in cfg['tasks']:
    paths=list((baseline/'tasks'/spec['task']).glob('**/attempt_00/result.json'));assert len(paths)==1
    source=json.loads(paths[0].read_text())['result'];case=Path(source['experiment_path']);tr=root/spec['task'];tr.mkdir()
    r=physical_case(spec['dataset'],source['episode'],tr,'saved_token_replay',source['object_delta_xy_m'],decision_path=source['decision_path'],saved_token_case=case)
    dest=Path(r['experiment_path']);a=np.load(case/'execution/controller.npz')['records'];b=np.load(dest/'execution/controller.npz')['records'];tokens=extract_tokens(a,len(a));actual=extract_tokens(b,len(b));assert np.array_equal(tokens[:len(b)],actual)
    old=np.load(case/'execution/trajectory.npz');new=np.load(dest/'execution/trajectory.npz');n=len(new['state']);delta=new['next_state']-old['next_state'][:n];target=new['commands'][:,1:30]-old['commands'][:n,1:30]
    audit={'tokens_exact':True,'no_encoder_calls':bool(np.all(b[:,1]==-2) and np.all(b[:,2+tokens.shape[1]:-29]==0)),'actual_initial_state_exact':bool(np.array_equal(old['state'][0],new['state'][0])),'state_trace_exact':bool(n==len(old['state']) and np.array_equal(old['next_state'],new['next_state'])),'commands_exact':bool(n==len(old['state']) and np.array_equal(old['commands'],new['commands'])),'common_substeps':n,'root_xyz_error_max_m':float(np.linalg.norm(delta[:,1:4],axis=1).max()),'q_target_error_rmse_rad':float(np.sqrt(np.mean(target**2))),'q_target_error_max_rad':float(np.abs(target).max()),'source_success':source['success'],'source_case':str(case)}
    assert audit['tokens_exact'] and audit['no_encoder_calls'] and audit['actual_initial_state_exact']
    save(dest/'token_replay_audit.json',audit);print(json.dumps({'task':spec['task'],'success':r['success'],'failure':r['failure'],'audit':audit}),flush=True)
    ins=inspect_case(dest,dest/'inspection')
    results.append({'task':spec['task'],'result':r,'audit':audit,'inspection':ins});save(root/'progress.json',results)
summary={'experiment_id':root.name,'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'api_requests':0,'tokens':0,'results':results};save(root/'summary.json',summary)
print('COMPLETE',summary['wall_s'],flush=True)
