"""Independent saved-token regression: no encoder, planner, residual, or API calls."""
import json
from pathlib import Path
import numpy as np

def extract_tokens(records,frames):
    a=np.asarray(records)
    if a.ndim!=2 or len(a)!=frames or a.shape[1]<=2+1762+29 or not np.isfinite(a).all():
        raise ValueError('Invalid or incomplete recorded controller rows')
    return a[:,2:-(1762+29)].astype(np.float32)

def verify_source_contract(case):
    from ..workflow.contracts import file_hash
    case=Path(case).resolve()
    if file_hash(case/'experiment.json')!=(case/'experiment.sha256').read_text():raise ValueError('Source contract modified')
    contract=json.loads((case/'experiment.json').read_text())
    if contract['record_status']!='frozen_ready' or not contract['preflight']['passed']:raise ValueError('Source not frozen')
    for path,digest in contract['preflight']['evidence_sha256'].items():
        original=Path(path)
        if original.exists() and file_hash(original)==digest:continue
        snapshots=list((case/'source_snapshot').glob('*_'+original.name))
        if not any(file_hash(p)==digest for p in snapshots):raise ValueError('Source evidence changed: '+path)
    return contract

def load_source(case,frames,task,episode,shift):
    case=Path(case).resolve();verify_source_contract(case)
    r=json.loads((case/'result.json').read_text())
    if (r['task']!=task or r['episode']!=episode or r['object_delta_xy_m']!=list(shift) or r['coverage']!=1 or r.get('branch_start_frame',0)!=0):
        raise ValueError('Saved-token source identity/coverage mismatch')
    tokens=extract_tokens(np.load(case/'execution/controller.npz')['records'],frames)
    names=['result.json','experiment.json','experiment.sha256','scene.mjb','initial.npz','physics.json','execution/controller.npz','execution/trajectory.npz']
    return tokens,r,[case/n for n in names]+[p for p in [case/'execution/contact_control.json',case/'execution/preencoder.json',case/'execution/hand_commands.npz',case/'execution/finger_control.json'] if p.exists()]
