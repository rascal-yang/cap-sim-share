"""Standalone worker, invoked with simulator Python; no installation in that env needed."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

req=json.loads(Path(sys.argv[1]).read_text());task=req['task'];out=Path(req['out']);res=task['resources'];source=Path(res['experiment']).resolve()
for p in res.get('python_paths',[]):sys.path.insert(0,p)
sys.path.insert(0,str(source/'harness'))
import sonic_controller
sonic_controller.MODELS=Path(res['sonic_models']).resolve()
from runtime import Runtime

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))
files=[source/'harness'/n for n in ['base_runtime.py','runtime.py','sonic_controller.py','policy.py']]+[source/'setup/deployment_parameters.json',source/'setup/initial_v1.npz',source/'assets/scene.xml']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
if req['op'] != 'inspect':
    tree0=ET.parse(source/'assets/scene.xml');compiler0=tree0.getroot().find('compiler')
    meshdir0=Path(compiler0.get('meshdir','')) if compiler0 is not None else Path('.')
    dependencies=files+[sonic_controller.MODELS/n for n in ['planner/target_vel/V2/planner_sonic.onnx','policy/release/model_encoder.onnx','policy/release/model_decoder.onnx']]
    dependencies += [(source/'assets'/meshdir0/e.get('file')).resolve() for e in tree0.findall('.//mesh') if e.get('file')]
    def file_sha(p):
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    frozen={str(p):file_sha(p) for p in dependencies};manifest=out.parents[1]/'external_resources.json'
    if manifest.exists() and json.loads(manifest.read_text())!=frozen:raise RuntimeError('External resources changed across attempts')
    dump(manifest,frozen)
if req['op'] in ('reset','inspect'):
    if set(task['scene'])-{'scale','table_delta','translation'}:raise ValueError('Unknown G1 scene variation')
    dump(out/'runtime_hashes.json',hashes)
    tree=ET.parse(source/'assets/scene.xml');root=tree.getroot();compiler=root.find('compiler');meshdir=Path(compiler.get('meshdir','')) if compiler is not None else Path('.')
    for elem in root.findall('.//mesh'):
        if elem.get('file'):elem.set('file',str((source/'assets'/meshdir/elem.get('file')).resolve()))
    if compiler is not None:compiler.set('meshdir','')
    v=task['scene'];scale=float(v.get('scale',1.));dz=float(v.get('table_delta',0.));translation=np.asarray(v.get('translation',[0,0,0]),float)
    if not .5<=scale<=1.5 or abs(dz)>.2 or translation.shape!=(3,) or not np.isfinite(translation).all() or np.linalg.norm(translation)>.2:raise ValueError('Scene outside supported G1 variation bounds')
    root.find(".//geom[@name='task_table']").set('pos',f'.3 0 {.60+dz}')
    geom=root.find(".//geom[@name='apple_collision']")
    for attr in ('pos','size'):geom.set(attr,' '.join(str(float(x)*scale) for x in geom.get(attr).split()))
    root.find(".//mesh[@name='approx_apple_mesh']").set('scale',f'{scale} {scale} {scale}')
    tree.write(out/'scene.xml')
    r=Runtime(out/'scene.xml');r.load(source/'setup/initial_v1.npz');r.d.qpos[50:53]+=translation
    ar=Rotation.from_quat(r.d.qpos[[54,55,56,53]]).as_matrix();g=r.m.geom('apple_collision').id
    r.d.qpos[52]=.65+dz-(ar@r.m.geom_pos[g])[2]+np.linalg.norm(ar[2]*r.m.geom_size[g])+.0001;r.d.qvel[49:55]=0
    mujoco.mj_forward(r.m,r.d);r.z0=float(r.d.qpos[52]);r.save_all(out)
    answer={'observation':r.observe()}
else:
    if json.loads((out/'runtime_hashes.json').read_text())!=hashes:raise RuntimeError('External runtime changed mid-attempt')
    r=Runtime(out/'scene.xml');r.load_all(out);a=req['action'];ob=r.observe()
    if r.halt or r.success:raise RuntimeError('Cannot advance terminal state')
    invalid=None
    hj=r.m.actuator_trnid[r.hand,0];f=np.asarray(a.get('fingers',r.finger_target))
    if np.any(f<r.m.jnt_range[hj,0]-.001) or np.any(f>r.m.jnt_range[hj,1]+.001):invalid='finger joint limits'
    if a['stage'] in ('lift','hold') and np.linalg.norm(np.asarray(a.get('wrist_position',r.d.xpos[r.wrist]))-r.d.xpos[r.wrist])>.031:invalid='lift increments limited to 3 cm'
    if invalid:
        np.savez_compressed(out/'worker_states.npz',samples=np.empty((0,113)))
        dump(out/'worker_response.json',{'completed':False,'reusable':False,'failure':'invalid_action: '+invalid,'samples':[],'observation':ob,'contacts':[]})
        sys.exit(0)
    result=r.phase(a);r.save_all(out);samples=[]
    for s,c in zip(r.frames,r.contacts):
        q=np.asarray(s[1:58]);tilt=float(np.arccos(np.clip(Rotation.from_quat(q[[4,5,6,3]]).as_matrix()[2,2],-1,1)))
        samples.append({'time':float(s[0]),'dt':.005,'lift_m':float(q[52]-r.z0),'support_force':c['apple_table_N'],'hand_force':c['apple_right_finger_N'],'stable':bool(q[2]>.6 and tilt<np.deg2rad(25))})
    np.savez_compressed(out/'worker_states.npz',samples=np.asarray(r.frames))
    completed=result.get('executed_duration_s',0)>=a['duration_s']-1e-8
    failed=result.get('rejection') or result.get('pause') or result.get('halt') or (None if result['stable'] and not result['blocked'] else 'unstable_or_blocked')
    answer={'completed':completed,'reusable':bool(completed and not failed and (a['stage'] in ['approach','recover'] or result['opposed_contact_s']>=.12)),'failure':failed,'samples':samples,'observation':result,'contacts':r.contacts}
dump(out/'worker_response.json',answer)
