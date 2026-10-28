"""Read-only, pre-control aligned source residuals; no physics integration."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .suite_physics import source_paths, extend_contact_tail
from .agent import save


def rotation_residual(actual, reference):
    a=Rotation.from_quat(np.asarray(actual)[..., [1,2,3,0]])
    r=Rotation.from_quat(np.asarray(reference)[..., [1,2,3,0]])
    return (r.inv()*a).as_rotvec()


def summarize(values):
    x=np.asarray(values)
    if not len(x): return None
    norms=np.linalg.norm(x.reshape(len(x),-1),axis=1)
    return {'first':x[0].tolist(),'last':x[-1].tolist(),'rmse_norm':float(np.sqrt(np.mean(norms**2))), 'p95_norm':float(np.percentile(norms,95)), 'max_norm':float(norms.max()),'max_frame_relative':int(norms.argmax())}


def diagnose(case, dataset):
    import mujoco
    import pandas as pd
    case=Path(case);result=json.loads((case/'result.json').read_text())
    decision=json.loads(Path(result['decision_path']).read_text())['decision']
    df=pd.read_parquet(source_paths(dataset,result['episode'])['parquet'])
    if decision['tail_s']:df=extend_contact_tail(df,round(decision['tail_s']/.02),.02,.02)
    m=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));actual=mujoco.MjData(m);ref=mujoco.MjData(m)
    trace=np.load(case/'execution/trajectory.npz');states=trace['state'][::4];n=len(states)
    q=states[:,1:1+m.nq];v=states[:,1+m.nq:];rq=np.stack(df['sim.qpos'])[:n,:m.nq];rv=np.stack(df['sim.qvel'])[:n,:m.nv]
    if not np.allclose(states[:,0],df['sim.time'].to_numpy()[:n],atol=1e-7,rtol=0):raise ValueError('Misaligned pre-control states')
    joints=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
    legs=[j for j in joints if any(k in m.joint(j).name for k in ('hip','knee','ankle'))]
    hands=[j for j in joints if any(k in m.joint(j).name for k in ('thumb','index','middle','ring','pinky'))]
    def joint_delta(ids,positions):
        adr=m.jnt_qposadr[ids] if positions else m.jnt_dofadr[ids]
        return (q-rq)[:,adr] if positions else (v-rv)[:,adr]
    data={'time_s':states[:,0],'root_xyz_error_m':q[:,:3]-rq[:,:3],'root_rotation_error_rad':rotation_residual(q[:,3:7],rq[:,3:7]),'leg_q_error_rad':joint_delta(legs,True),'leg_dq_error_rad_s':joint_delta(legs,False),'hand_q_error_rad':joint_delta(hands,True)}
    ay=Rotation.from_quat(q[:,[4,5,6,3]]).as_euler('xyz')[:,2];ry=Rotation.from_quat(rq[:,[4,5,6,3]]).as_euler('xyz')[:,2]
    data['root_heading_error_rad']=((ay-ry+np.pi)%(2*np.pi)-np.pi)[:,None]
    for side in ('left','right'):
        body=m.body(side+'_wrist_yaw_link').id;dp=[];dr=[]
        for i in range(n):
            actual.qpos[:]=q[i];ref.qpos[:]=rq[i];mujoco.mj_kinematics(m,actual);mujoco.mj_kinematics(m,ref)
            dp.append(actual.xpos[body]-ref.xpos[body]);dr.append(rotation_residual(actual.xquat[body],ref.xquat[body]))
        data[side+'_wrist_xyz_error_m']=np.array(dp);data[side+'_wrist_rotation_error_rad']=np.array(dr)
        data[side+'_hand_command_rad']=np.stack(df['teleop.'+side+'_hand_joints'])[:n]
    out=case/'diagnostics';out.mkdir(exist_ok=True);np.savez_compressed(out/'residuals.npz',**data)
    a=decision['contact_start'];z=decision['contact_end'];metrics=json.loads((case/'execution/metrics.json').read_text())
    segments={'approach':(0,min(a,n)),'contact':(min(a,n),min(z+1,n)),'after_contact':(min(z+1,n),n)}
    report={'alignment':'pre-control source row versus first substep; world XYZ signed actual minus source; rotation vector in source local frame','physics_steps':0,'source_hand_commands_unchanged':True,'leg_joint_names':[m.joint(j).name for j in legs],'hand_joint_names':[m.joint(j).name for j in hands],'frames':n,'contact_bounds':[a,z], 'segments':{name:{key:summarize(value[lo:hi]) for key,value in data.items() if 'error' in key} for name,(lo,hi) in segments.items()},'first_native_success_frame':next((i for i,r in enumerate(metrics) if r['native_success']),None),'native_success_before_switch':any(r['native_success'] for r in metrics[:a]),'native_success_at_end':bool(metrics[-1]['native_success']) if metrics else False,'first_contact_frame':next((i for i,r in enumerate(metrics) if r['robot_object_N']>.1),None),'reporting_note':'Contact 0.1N is a diagnostic reporting threshold, not acceptance. No synthetic closure event: hand joint commands and measured residuals are exported per frame.'}
    save(out/'summary.json',report);return report
