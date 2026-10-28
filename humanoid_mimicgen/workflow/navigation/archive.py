"""Read explicit archive manifests. Frozen source files are never modified."""
import json
import time
from pathlib import Path
import numpy as np
from ..contracts import WorkflowError, file_hash, positive
from .prior import finite


def verify_evidence(evidence):
    for name,expected in evidence.items():
        if not Path(name).is_file() or file_hash(name)!=expected:
            raise WorkflowError('integrity_error','Evidence changed: '+name)


def load_archive(manifest_path,deadline=None):
    import mujoco
    from ...privileged import no_physics
    from ..native import descendants, geom_z_bounds
    path=Path(manifest_path).resolve()
    evidence={str(path):file_hash(path)};manifest=json.loads(path.read_text())
    files={k:(path.parent/v).resolve() for k,v in manifest['files'].items()}
    for p in files.values():evidence[str(p)]=file_hash(p)
    b=manifest['bindings'];result=json.loads(files['result'].read_text())
    with no_physics():
        m=mujoco.MjModel.from_binary_path(str(files['model']));d=mujoco.MjData(m)
        with np.load(files['trajectory'],allow_pickle=False) as z:states=z[manifest.get('state_key','state')].copy()
        if states.ndim!=2 or states.shape[1]!=1+m.nq+m.nv or not np.isfinite(states).all():
            raise WorkflowError('invalid_input','Expected time,qpos,qvel archive layout')
        n=len(states);entry=manifest['entry_index'];end=manifest['end_index']
        if type(entry) is not int or type(end) is not int or not 0<entry<=end<n:
            raise WorkflowError('invalid_input','Archive phase indices')
        root=m.body(b['root_body']).id;rootjoint=int(m.body_jntadr[root])
        if rootjoint<0 or m.jnt_qposadr[rootjoint]!=0 or m.jnt_type[rootjoint]!=mujoco.mjtJoint.mjJNT_FREE:
            raise WorkflowError('unsupported_capability','Expected first free root')
        object_id=m.body(b['object']).id
        feet=[descendants(m,m.body(name).id) for name in b['feet']]
        ground={m.geom(name).id for name in b['ground_geoms']}
        if not feet or not ground:raise WorkflowError('needs_annotation','Foot and ground bindings required')
        tol=manifest['support_distance_m'];positive(tol,'Support geometry tolerance',zero=True)
        objects=[];support=[];environment={}
        body_poses={name:[] for name in b.get('capture_bodies',[])}
        for i,state in enumerate(states):
            if deadline is not None and time.monotonic()>deadline:
                raise WorkflowError('planning_budget_exhausted','Source FK deadline')
            d.qpos[:]=state[1:1+m.nq];mujoco.mj_forward(m,d)
            objects.append(np.r_[d.xpos[object_id],d.xquat[object_id]])
            for name,poses in body_poses.items():
                bid=m.body(name).id;poses.append(np.r_[d.xpos[bid],d.xquat[bid]])
            supported=[]
            for bodies in feet:
                supported.append(any(c.dist<=tol and
                    ((int(c.geom1) in ground and int(m.geom_bodyid[c.geom2]) in bodies) or
                     (int(c.geom2) in ground and int(m.geom_bodyid[c.geom1]) in bodies)) for c in d.contact))
            support.append(all(supported))
            if i==entry:
                for name in b.get('support_geoms',[]):
                    g=m.geom(name).id;center=d.geom_xpos[g].copy();rotation=d.geom_xmat[g].reshape(3,3)
                    info={'center_world_m':center.tolist(),'rotation_world':rotation.tolist(),'height_bounds_m':list(geom_z_bounds(m,d,g)),
                          'root_in_geom_m':(rotation.T@(d.qpos[:3]-center)).tolist()}
                    if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                        info['half_extents_m']=m.geom_size[g].tolist()
                        info['root_to_box_edge_xy_m']=(abs(rotation.T@(d.qpos[:3]-center))[:2]-m.geom_size[g,:2]).tolist()
                    environment[name]=info
    success_document=result
    if 'acceptance' in files:success_document=json.loads(files['acceptance'].read_text())
    success=success_document
    for part in manifest['success_field'].split('.'):
        success=success.get(part) if isinstance(success,dict) else None
    trace={'time_s':states[:,0],'qpos':states[:,1:1+m.nq],'root_pose':states[:,1:8],
           'root_velocity':states[:,1+m.nq:1+m.nq+6], 'object_pose':np.asarray(objects),
           'support_ok':np.asarray(support,bool),'support_evidence_kind':'geometric_contact_not_force',
           'success':success,'success_criterion':manifest['success_criterion'],
           'task':result.get('task'),'entry_index':entry,'end_index':end,
           'environment_relations':environment,'body_poses':{k:np.asarray(v) for k,v in body_poses.items()},
           'manifest':manifest,'files':{k:str(v) for k,v in files.items()}}
    verify_evidence(evidence)
    return trace,b,evidence
