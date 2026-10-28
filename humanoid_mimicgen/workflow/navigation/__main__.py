"""JSON CLI; native geometry and execution use scripts/humano_python.sh."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from ..contracts import WorkflowError, canonical, file_hash, now
from ..service import write_new
from ..__main__ import diagnostics_to_stderr
from .archive import load_archive, verify_evidence
from .prior import extract_prior, task_frame, yaw


def extract(manifest,options):
    trace,bindings,evidence=load_archive(manifest)
    opt=Path(options).resolve();evidence[str(opt)]=file_hash(opt)
    started=now();tick=time.monotonic()
    prior=extract_prior(trace,bindings,json.loads(opt.read_text()))
    prior.update(source_manifest=str(Path(manifest).resolve()),evidence_sha256=evidence,
                 support_evidence_kind=trace['support_evidence_kind'],source_trajectory_sha256=file_hash(trace['files']['trajectory']),
                 started_at=started,ended_at=now(),analysis_wall_s=time.monotonic()-tick)
    verify_evidence(evidence);return prior


def plan(prior_path,scene_path):
    import mujoco
    from ...privileged import no_physics
    from .native import NativeEvaluator
    from .planner import plan_stances
    pp=Path(prior_path).resolve();sp=Path(scene_path).resolve()
    evidence={str(pp):file_hash(pp),str(sp):file_hash(sp)}
    prior=json.loads(pp.read_text());verify_evidence(prior['evidence_sha256']);evidence.update(prior['evidence_sha256'])
    scene=json.loads(sp.read_text());files={k:str((sp.parent/v).resolve()) for k,v in scene['files'].items()}
    for path in files.values():evidence[path]=file_hash(path)
    trace,binding,source_evidence=load_archive(prior['source_manifest']);evidence.update(source_evidence)
    with no_physics():
        model=mujoco.MjModel.from_binary_path(files['model']);data=mujoco.MjData(model)
        source_model=mujoco.MjModel.from_binary_path(trace['files']['model'])
        if model.nq!=source_model.nq or model.nv!=source_model.nv or any(model.joint(j).name!=source_model.joint(j).name for j in range(model.njnt)) or not np.array_equal(model.jnt_type,source_model.jnt_type):
            raise WorkflowError('unsupported_capability','Source/scene articulation layout differs')
        with np.load(files['initial'],allow_pickle=False) as z:current=z['qpos'].copy()
        data.qpos[:]=current;mujoco.mj_forward(model,data);obj=model.body(binding['object']).id
        frame=task_frame(np.r_[data.xpos[obj],data.xquat[obj]],scene['frame'])
        evaluator=NativeEvaluator(model,current,trace['qpos'],prior,frame,binding,scene['native_options'])
        result=plan_stances(prior,{'task_frame_xyzyaw':frame},[*current[:2],float(yaw(current[:7]))],scene['search'],evaluator)
    result.update(evidence_sha256=evidence,scene_files=files,source_manifest=prior['source_manifest'],
                  native_options=scene['native_options'],source_trajectory_sha256=prior['source_trajectory_sha256'],
                  scene_input_sha256={k:file_hash(v) for k,v in files.items()},created_at=now())
    verify_evidence(evidence);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Replay relation extraction and deterministic navigation')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('extract');p.add_argument('--manifest',required=True);p.add_argument('--options',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('plan');p.add_argument('--prior',required=True);p.add_argument('--scene',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('execute');p.add_argument('--plan',required=True);p.add_argument('--experiment',required=True)
    p=sub.add_parser('lateral-extract');p.add_argument('--manifest',required=True);p.add_argument('--options',required=True);p.add_argument('--out',required=True)
    for name in ('lateral-plan','lateral-grid'):
        p=sub.add_parser(name);p.add_argument('--manifest',required=True);p.add_argument('--options',required=True)
        p.add_argument('--initial',required=True);p.add_argument('--out-dir',required=True)
        if name=='lateral-plan':p.add_argument('--offset-m',type=float,required=True)
    args=parser.parse_args(argv)
    try:
        if hasattr(args,'out') and Path(args.out).exists():raise WorkflowError('output_exists','Refusing overwrite')
        with diagnostics_to_stderr():
            if args.command=='extract':result=extract(args.manifest,args.options)
            elif args.command=='plan':result=plan(args.prior,args.scene)
            elif args.command=='lateral-extract':
                from .lateral_prior import load_lateral_source
                from .lateral import surface_frame
                import mujoco
                trace,result,_,_=load_lateral_source(args.manifest,json.loads(Path(args.options).read_text()))
                model=mujoco.MjModel.from_binary_path(trace['files']['model'])
                result['surface']=surface_frame(model,trace['qpos'][0],result['bindings'])
                result['evidence_sha256'][str(Path(args.options).resolve())]=file_hash(args.options)
            elif args.command=='lateral-plan':
                from .lateral import plan_lateral
                result=plan_lateral(args.manifest,args.initial,json.loads(Path(args.options).read_text()),args.out_dir,args.offset_m)
            elif args.command=='lateral-grid':
                from .lateral import plan_lateral
                root=Path(args.out_dir).resolve();root.mkdir(parents=True,exist_ok=False)
                options=json.loads(Path(args.options).read_text());start=now();clock=time.monotonic();rows=[]
                for index,offset in enumerate([0.,-.05,.05,-.10,.10,-.15,.15,-.20,.20]):
                    case=root/f'{index:02d}_y_{offset:+.2f}'
                    value=plan_lateral(args.manifest,args.initial,options,case,offset)
                    row={k:value.get(k) for k in ('offset_m','status','reason','planning_wall_s','search_incomplete')}
                    row.update(plan_file=str(case/'plan.json'),selected_id=(value.get('selected') or {}).get('id'),
                               candidates=[{k:c.get(k) for k in ('id','reason','feasible','planning_wall_s')} for c in value['candidates']])
                    rows.append(row);write_new(root/f'progress_{index:02d}.json',row)
                result=dict(kind='apple_lateral_offline_grid',status='completed',cases=rows,physics_steps=0,api_calls=0,
                    dynamic_execution_verified=False,started_at=start,ended_at=now(),planning_wall_s=time.monotonic()-clock,
                    completed_means='all declared planning conditions recorded; not physical task success')
                write_new(root/'summary.json',result)
            else:
                from .integration import execute_plan
                result=execute_plan(args.plan,args.experiment)
            if hasattr(args,'out'):write_new(args.out,result)
        print(canonical(result).decode());return 0 if result.get('status')=='completed' else 1
    except (WorkflowError,OSError,ValueError,KeyError,TypeError) as exc:
        result={'status':'failed','reason_code':exc.code if isinstance(exc,WorkflowError) else 'invalid_input',
                'detail':exc.detail if isinstance(exc,WorkflowError) else str(exc)}
        print(canonical(result).decode());return 2

if __name__=='__main__':sys.exit(main())
