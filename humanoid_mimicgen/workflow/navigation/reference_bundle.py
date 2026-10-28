"""Immutable generated references, separate from measured replay evidence."""
import json
from pathlib import Path
import numpy as np
from ..contracts import WorkflowError, file_hash, canonical
from .archive import verify_evidence


def validate_arrays(arrays, metadata):
    try:
        q=np.asarray(arrays['qpos']);t=np.asarray(arrays['time_s']);rows=np.asarray(arrays['source_rows'])
        n=len(t);a,z=metadata['operation_interval'];b=np.asarray(arrays['bridge_qpos']);bt=np.asarray(arrays['bridge_time_s'])
        valid=(n>1 and q.ndim==2 and q.shape[0]==n and q.shape[1]>=7 and
               t.shape==(n,) and np.all(np.diff(t)>0) and rows.dtype.kind in 'iu' and
               np.array_equal(rows,np.arange(a,z+1)) and len(rows)==n and
               np.asarray(arrays['fingers']).shape==(n,14) and b.ndim==2 and b.shape[1]==q.shape[1] and
               bt.shape==(len(b),) and len(b)>1 and bt[0]==0 and np.all(np.diff(bt)>0) and
               np.allclose(b[-1],q[0],atol=1e-8,rtol=0))
        if not valid or any(not np.isfinite(np.asarray(v)).all() for v in arrays.values()):raise ValueError()
        for poses in (q,b):
            if not np.allclose(np.linalg.norm(poses[:,3:7],axis=1),1,atol=1e-5):raise ValueError()
    except (KeyError,ValueError,TypeError):raise WorkflowError('invalid_reference_bundle','Invalid generated arrays/index/timing') from None


def save_reference_bundle(directory, arrays, metadata):
    validate_arrays(arrays,metadata)
    p=Path(directory).resolve();p.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(p/'reference.npz',**arrays)
    meta=dict(metadata,kind='generated_operation_reference',operation_kind='trajectory_retargeting',
              dynamic_execution_verified=False,arrays_sha256=file_hash(p/'reference.npz'))
    (p/'manifest.json').write_bytes(canonical(meta))
    return dict(arrays_path=str(p/'reference.npz'),metadata_path=str(p/'manifest.json'),
                evidence_sha256={str(f):file_hash(f) for f in (p/'reference.npz',p/'manifest.json')})


def load_reference_bundle(reference):
    verify_evidence(reference['evidence_sha256'])
    for k in ('arrays_path','metadata_path'):
        if str(Path(reference[k]).resolve()) not in reference['evidence_sha256']:
            raise WorkflowError('integrity_error','Unbound generated reference '+k)
    metadata=json.loads(Path(reference['metadata_path']).read_text())
    if metadata.get('kind')!='generated_operation_reference' or metadata.get('dynamic_execution_verified') is not False:
        raise WorkflowError('invalid_reference_bundle','Generated reference provenance')
    verify_evidence(metadata.get('source_evidence_sha256',{}))
    if file_hash(reference['arrays_path'])!=metadata['arrays_sha256']:
        raise WorkflowError('integrity_error','Reference hash differs from manifest')
    with np.load(reference['arrays_path'],allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    validate_arrays(arrays,metadata)
    return dict(arrays,metadata=metadata)
