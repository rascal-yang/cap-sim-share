"""Read-only conversion of jointly accepted, complete physical traces to named arrays."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
from humanoid_mimicgen.augmentation.suite_physics import source_paths,extend_contact_tail
from humanoid_mimicgen.workflow.contracts import now

def export(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());cfg=json.loads((root/'config.json').read_text());datasets={s['task']:s['dataset'] for s in cfg['tasks']};out=root/'accepted_dataset';out.mkdir(exist_ok=True);items=[]
    for item in summary['exports']:
        case=Path(item['experiment_path']);r=json.loads((case/'result.json').read_text());assert r['success'] and r['independent_success'] and r['coverage']==1 and not r['failure']
        trace=np.load(case/'execution/trajectory.npz');controller=np.load(case/'execution/controller.npz')['records'];model=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));n=len(controller);assert len(trace['state'])==n*4
        d=json.loads(Path(r['decision_path']).read_text())['decision'];df=pd.read_parquet(source_paths(datasets[r['task']],r['episode'])['parquet'])
        if d['tail_s']:df=extend_contact_tail(df,round(d['tail_s']/.02),.02,.02)
        token_width=len(df['action.motion_token'].iloc[0]);assert controller.shape[1]==2+token_width+1762+29
        pre=trace['state'][::4];post=trace['next_state'][3::4];assert len(pre)==len(post)==n==len(df)
        np.testing.assert_allclose(pre[:,0],controller[:,0],atol=1e-8,rtol=0)
        np.testing.assert_allclose(post[:,0]-pre[:,0],.02,atol=1e-8,rtol=0)
        folder=out/(case.name+'__'+case.parents[2].name+'_'+case.parents[1].name);folder.mkdir(exist_ok=True)
        arrays={'time_s':pre[:,0],'next_time_s':post[:,0],'qpos':pre[:,1:1+model.nq],'qvel':pre[:,1+model.nq:],'next_qpos':post[:,1:1+model.nq],'next_qvel':post[:,1+model.nq:],'motion_token':controller[:,2:2+token_width].astype(np.float32),'encoder_mode':controller[:,1].astype(np.int32),'body_joint_target_rad':controller[:,-29:],'left_hand_command_rad':np.stack(df['teleop.left_hand_joints']),'right_hand_command_rad':np.stack(df['teleop.right_hand_joints']),'native_success_after_action':np.array([x['native_success'] for x in json.loads((case/'execution/metrics.json').read_text())],bool)}
        if not (folder/'episode.npz').exists():np.savez_compressed(folder/'episode.npz',**arrays)
        loaded=np.load(folder/'episode.npz');assert all(np.array_equal(loaded[k],v) for k,v in arrays.items())
        meta={'experiment_id':case.name,'exported_at':now(),'task':r['task'],'source_episode':r['episode'],'source_case':str(case),'object_delta_xy_m':r['object_delta_xy_m'],'policy_dt_s':.02,'physics_dt_s':.005,'policy_frames':n,'original_token_prefix_frames':r['original_token_steps'],'encoder_mode_semantics':{'-1':'stored original token; no encoder','1':'new mode1 encoded token'},'state_alignment':'pre-control qpos/qvel -> action -> four physics steps -> next_qpos/next_qvel; source hand commands unchanged','video':item['video'],'video_alignment':'video frame 0 is initial state; frame i+1 is post-control state for policy action i','video_frame_mapping':str(case/'inspection/frame_mapping.json'),'array_shapes':{k:list(v.shape) for k,v in arrays.items()},'quality_accepted':True,'training_ready':False,'temporal_review':'pending','original_actions_visible':True,'generation_type':'trajectory retargeting','artifacts_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (case/'execution/trajectory.npz',case/'execution/controller.npz',case/'experiment.json',folder/'episode.npz')}}
        if (folder/'metadata.json').exists():
            prior=json.loads((folder/'metadata.json').read_text());assert prior['source_case']==str(case) and prior['array_shapes']==meta['array_shapes']
        else:(folder/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        items.append({'episode':str(folder/'episode.npz'),'metadata':str(folder/'metadata.json'),'frames':n,'task':r['task']})
    (out/'manifest.json').write_text(json.dumps(items,indent=2));print(json.dumps({'accepted_episodes':len(items),'policy_frames':sum(x['frames'] for x in items),'output':str(out)},ensure_ascii=False))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();export(a.root)
