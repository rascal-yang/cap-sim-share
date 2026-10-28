"""Read-only comparison of replay and retarget contact/command evidence."""
import json,subprocess
from pathlib import Path
import numpy as np
from .agent import save
from .direct_augmentation import compact_case,keyframes
from ..workflow.contracts import file_hash


def metric_state(trace,timestamp):
    """Resolve identity at the metric's time, including truncated policy ticks."""
    for key in ('next_state','state'):
        rows=trace[key];matches=np.flatnonzero(np.isclose(rows[:,0],timestamp,rtol=0,atol=1e-9))
        if len(matches)==1:return rows[matches[0]]
        if len(matches)>1:raise ValueError('Ambiguous metric timestamp in recorded states')
    raise ValueError('No recorded state at metric timestamp')


def inspect_contact_case(case,contact_start,evidence_root=None,inspection_path=None):
    import mujoco
    case=Path(case);r=json.loads((case/'result.json').read_text())
    m=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));d=mujoco.MjData(m)
    trace=np.load(case/'execution/trajectory.npz');metrics=json.loads((case/'execution/metrics.json').read_text())
    qids=[i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE and '_hand_' not in m.joint(i).name][:29]
    state=trace['next_state'][-1];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];d.ctrl[:]=trace['commands'][-1,-m.nu:];mujoco.mj_forward(m,d)
    pelvis=m.body('pelvis').id
    def robot(body):
        while body>0:
            if body==pelvis:return True
            body=int(m.body_parentid[body])
        return False
    bodies=[m.body(i).name for i in range(1,m.nbody) if robot(i)]
    positions=metrics[-1]['object_xyz'];target_key='target' if 'target' in positions else 'articulated'
    if target_key not in positions:raise ValueError('Unresolved task target identity')
    target_pos=np.asarray(positions[target_key])
    identity_state=metric_state(trace,metrics[-1]['time_s'])
    d.qpos[:]=identity_state[1:1+m.nq];d.qvel[:]=identity_state[1+m.nq:];mujoco.mj_forward(m,d)
    matches=[i for i in range(1,m.nbody) if not robot(i) and np.linalg.norm(d.xpos[i]-target_pos)<1e-6]
    def depth(body):
        n=0
        while body>0:n+=1;body=int(m.body_parentid[body])
        return n
    if not matches:raise ValueError('No target body matches recorded task position')
    target=min(matches,key=depth)
    def target_child(body):
        while body>0:
            if body==target:return True
            body=int(m.body_parentid[body])
        return False
    if not all(target_child(i) for i in matches):raise ValueError('Ambiguous task target position')
    candidates=[i for i in range(1,m.nbody) if target_child(i)]
    # Identity is fixed; terminal geometry still uses the actual final substep.
    d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];mujoco.mj_forward(m,d)
    distances={}
    object_geoms=[g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in candidates]
    for g in range(m.ngeom):
        b=int(m.geom_bodyid[g])
        if not robot(b) or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        for og in object_geoms:
            if not (m.geom_contype[og] or m.geom_conaffinity[og]):continue
            dist=mujoco.mj_geomDistance(m,d,g,og,.5,None)
            name=m.body(b).name
            distances[name]=min(distances.get(name,.5),float(dist))
    ip=Path(inspection_path) if inspection_path else case/'inspection/inspection.json'
    if not ip.exists():
        ip=next(case.glob('*/inspection.json'))
    ins=json.loads(ip.read_text())
    if Path(ins['source']).resolve()!=case.resolve():raise ValueError('Inspection source identity mismatch')
    for source,digest in ins['source_sha256'].items():
        if file_hash(source)!=digest:raise ValueError('Inspection source hash mismatch')
    joints=[{'name':m.joint(i).name,'range_rad':m.jnt_range[i].tolist(),'joint_axis_local':m.jnt_axis[i].tolist(),'final_measured_rad':float(d.qpos[m.jnt_qposadr[i]])} for i in qids]
    summary=compact_case(case)
    executed=json.loads(Path(r['decision_path']).read_text())['decision']
    summary['executed_parameters']={k:executed[k] for k in ('episode','contact_start','contact_end','tail_s','wrist_delta_m','contact_control') if k in executed}
    physics=json.loads((case/'physics.json').read_text())
    summary['baseline_joint_kp']=physics['kp'];summary['baseline_joint_kd']=physics['kd']
    summary.update(contact_pairs=ins['contact_pairs'],robot_bodies=bodies,joints=joints,target_body=m.body(target).name,target_key=target_key,final_collision_distance_to_object_m=distances,
        distance_caveat='mj_geomDistance zero does not prove loaded contact; rely on contact pairs/forces',
        inspection={'review_images':ins.get('review_images',[ins['sheet']])[:1]},source_hashes={str(p):file_hash(p) for p in [case/'result.json',case/'scene.mjb',case/'execution/trajectory.npz']})
    control=case/'execution/contact_control.json'
    if control.exists():
        rows=json.loads(control.read_text());summary['force_controller']={'final':rows[-1],'max_force_N':max(x['contact_force_N'] for x in rows),'max_abs_force_offset_rad':np.max(np.abs([x['force_offset_rad'] for x in rows]),axis=0).tolist()}
    if evidence_root is not None:
        summary['inspection']={'review_images':[compact_frames(Path(ins['video']),contact_start,len(metrics),Path(evidence_root))]}
    return summary


def compact_frames(video,start,frames,folder):
    from PIL import Image,ImageDraw,ImageFont
    from .render_bootstrap import FONT
    folder=folder/(video.parent.parent.name+'_'+file_hash(video)[:10])
    folder.mkdir(parents=True,exist_ok=True);out=folder/'keyframes.jpg'
    if out.exists():return str(out)
    picks=sorted(set([min(start,frames),min((start+frames)//2,frames),frames]))
    canvas=Image.new('RGB',(960,320*len(picks)));font=ImageFont.truetype(FONT,16)
    for i,f in enumerate(picks):
        dest=folder/f'frame_{f:04d}.jpg'
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{f})','-vsync','0','-frames:v','1',str(dest)],check=True)
        canvas.paste(Image.open(dest).resize((960,300)),(0,320*i));ImageDraw.Draw(canvas).text((4,320*i+301),f'{video.parent.parent.name}: video frame {f}',font=font,fill='white')
    canvas.save(out,quality=88);save(folder/'mapping.json',{'source_video':str(video),'source_sha256':file_hash(video),'frames':picks,'no_new_physics':True})
    return str(out)


def acceptance_sources(case):
    sources=sorted((Path(case)/'source_snapshot').glob('*_g1_wholebody*.py'))
    if not sources:raise ValueError('Native task source snapshot missing')
    return {'code':'\n\n'.join('# '+p.name+'\n'+p.read_text() for p in sources),'hashes':{str(p):file_hash(p) for p in sources}}


def reference_differences(ra,rb,names,start):
    n=min(len(ra),len(rb))
    if start>=n:return []
    delta=np.asarray(ra)[start:n]-np.asarray(rb)[start:n]
    return [{'name':j,'replay_minus_mode1_mean_rad':float(delta[:,i].mean()),'replay_minus_mode1_final_rad':float(delta[-1,i]),'max_abs_rad':float(abs(delta[:,i]).max())} for i,j in enumerate(names)]


def compare(spec,out):
    out=Path(out)
    a=inspect_contact_case(spec['baseline'],spec['contact_start'],out.parent/'evidence/replay',spec.get('baseline_inspection'))
    b=inspect_contact_case(spec['failed_case'],spec['contact_start'],out.parent/'evidence/diagnostic',spec.get('diagnostic_inspection'))
    source=acceptance_sources(spec['baseline'])
    ra=np.load(Path(spec['baseline'])/'execution/controller.npz')['records'][:,-29:]
    rb=np.load(Path(spec['failed_case'])/'execution/controller.npz')['records'][:,-29:]
    result={'replay':a,'diagnostic':b,'acceptance_source':source['code'],'acceptance_source_sha256':source['hashes'],
        'joint_reference_difference':reference_differences(ra,rb,[j['name'] for j in a['joints']],spec['contact_start']),
        'evidence_semantics':'Contacts and forces recomputed from saved physical states/commands, no new physics. Diagnostic may have failed or passed: read native result. Context/controller/object offset can differ. Empty joint difference means diagnostic truncated before contact. Normal-force sum is not task-directional force.'}
    if spec.get('diagnostic_review'):
        review_path=Path(spec['diagnostic_review']);result['diagnostic_api_review']=json.loads(review_path.read_text())['decision'];result['diagnostic_api_review_sha256']=file_hash(review_path)
    save(out,result);return result
