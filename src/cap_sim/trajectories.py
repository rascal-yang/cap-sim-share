"""Reference import preserves success provenance and useful incomplete prefixes."""
import hashlib
from pathlib import Path
import numpy as np
from .core import evaluate
from .io import digest, read, write


def import_reference(source,out,kind='cap',scene_xml=None):
    source=Path(source).resolve()
    if kind=='cap':
        ref=read(source);attempt=Path(ref['source_attempt']);evidence=source.parent/'samples.json';samples=read(evidence if evidence.exists() else attempt/'samples.json')
        if digest(samples)!=ref['samples_sha256']:raise ValueError('Reference evidence hash mismatch')
        acceptance=evaluate(samples,ref['task']['acceptance'])
        result={'kind':'cap_reference','source':str(source),'actions':ref['actions'],'source_task':ref['task'],'source_acceptance':acceptance,'verified_from_physics':True}
    elif kind=='g1-attempt':
        initial=read(source/'initial.json');variant=read(source/'variant.json');samples=[];actions=[];prefix=[];stages=[]
        from scipy.spatial.transform import Rotation
        for f in sorted(source.glob('phase_*_states.npz')):
            s=np.load(f)['samples'];c=read(f.with_name(f.name.replace('_states.npz','_contacts.json')));meta=read(f.with_name(f.name.replace('_states.npz','.json')))
            if len(s)!=len(c):raise ValueError('Contact/state length mismatch')
            for row,contact in zip(s,c):
                q=row[1:58];tilt=Rotation.from_quat(q[[4,5,6,3]]).as_matrix()[2,2]
                samples.append({'time':float(row[0]),'dt':.005,'lift_m':float(q[52]-initial['apple_position'][2]),'support_force':contact['apple_table_N'],'hand_force':contact['apple_right_finger_N'],'stable':bool(q[2]>.6 and tilt>np.cos(np.deg2rad(25)))})
            actions.append(meta['command']);stages.append({'action':meta['command'],'measured_wrist':meta['wrist_position'],'object_position':meta['apple_position'],'opposed_contact_s':meta['opposed_contact_s']})
            if not meta.get('pause') and not meta.get('halt') and meta['stable'] and not meta['blocked'] and meta.get('executed_duration_s',0)>=meta['command']['duration_s']-1e-8 and (meta['command']['stage'] in ('recover','approach') or meta['opposed_contact_s']>=.12):prefix=list(actions)
        result={'kind':'g1_reference','source':str(source),'source_scene':variant,'actions':actions,'valid_prefix':prefix,'stages':stages,'source_acceptance':evaluate(samples,{'lift_m':.1,'hold_s':1.}),'verified_from_physics':True}
    elif kind=='humano':
        if not scene_xml:raise ValueError('HUMANO import requires compatible G1/Dex3 scene XML for forward kinematics')
        import pandas as pd
        import mujoco
        from scipy.spatial.transform import Rotation
        data=pd.read_parquet(source);m=mujoco.MjModel.from_xml_path(str(Path(scene_xml).resolve()));d=mujoco.MjData(m);w=m.body('right_wrist_yaw_link').id
        if m.nq!=57:raise ValueError('Expected G1/Dex3 57-position model')
        rows=[]
        for i in np.linspace(0,len(data)-1,min(32,len(data)),dtype=int):
            q=np.asarray(data.iloc[i]['sim.qpos'])[:57];d.qpos[:]=q;mujoco.mj_forward(m,d);ar=Rotation.from_quat(q[[54,55,56,53]])
            rows.append({'frame':int(i),'time':float(data.iloc[i]['timestamp']),'object_position':q[50:53].tolist(),'wrist_object_local':ar.inv().apply(d.xpos[w]-q[50:53]).tolist(),'wrist_quaternion_wxyz':d.xquat[w].tolist(),'fingers':q[43:50].tolist()})
        result={'kind':'humano_geometric_reference','source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'stages':rows,'actions':[],'verified_from_physics':False,'source_acceptance':None,'limitations':['Forward kinematics on supplied approximate assets; not original contact/force ground truth.','Sparse geometry samples are experience, not a verified successful current rollout.']}
    else:raise ValueError('Unknown reference format')
    result['current_task_success']=False
    write(out,result);return result
