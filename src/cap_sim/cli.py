import argparse
import importlib
import json
import os
from pathlib import Path
from .pipeline import run_pipeline, augment_pipeline
from .io import read,write
from .planners import DemoPlanner,FilePlanner,CommandPlanner,OpenAIPlanner


def make_backend(task):
    names={'humano':'cap_sim.backends.humano:Backend','mujoco_gripper':'cap_sim.backends.demo:Backend','g1_sonic':'cap_sim.backends.g1:Backend'}
    path=names.get(task['backend'],task['backend'])
    if ':' not in path:raise ValueError('Unknown backend; use registered name or trusted module:class')
    mod,name=path.split(':');return getattr(importlib.import_module(mod),name)()


def _make_planner(args,out):
    if args.planner=='demo':return DemoPlanner()
    if args.planner=='file':
        if not args.plans:raise ValueError('--plans required')
        return FilePlanner(args.plans)
    if args.planner=='command':
        if not args.command:raise ValueError('--command must be a JSON argv array')
        command=json.loads(args.command)
        if not isinstance(command,list) or not all(isinstance(x,str) for x in command):raise ValueError('Command must be argv list')
        return CommandPlanner(command)
    if not args.model:raise ValueError('--model required for OpenAI; no model name silently chosen')
    return OpenAIPlanner(args.model,Path(out).parent/(Path(out).name+'_model_calls'),args.base_url)


def make_planner(args,out):
    planner=_make_planner(args,out)
    roots=getattr(args,'evolving_skill',[])
    if roots:
        if getattr(args,'op',None) not in ('run','augment'):raise ValueError('--evolving-skill applies to run/augment attempts only')
        from .task_skills import EvolvingSkillPlanner
        planner=EvolvingSkillPlanner(planner,roots,Path(out)/'skill_versions')
    return planner


def load_task(path):
    path=Path(path).resolve();task=read(path)
    for key in ('references','skills'):
        task[key]=[str((path.parent/Path(p)).resolve()) for p in task.get(key,[])]
    for key in ('experiment','sonic_models','python','asset_manifest','prepared','initial_state'):
        if key in task.get('resources',{}):task['resources'][key]=str((path.parent/Path(task['resources'][key])).resolve())
    if 'python_paths' in task.get('resources',{}):task['resources']['python_paths']=[str((path.parent/Path(p)).resolve()) for p in task['resources']['python_paths']]
    return task


def add_task_skills(task, paths):
    from .task_skills import load_skill
    for path in paths:
        p=Path(path).resolve();load_skill(p)
        task.setdefault('skills',[]).append(str(p))


def planner_args(p):
    p.add_argument('--evolving-skill',action='append',default=[],help='Task skill registry directory: select latest before each attempt, freeze during feedback')
    p.add_argument('--skill',action='append',default=[],help='Exact versioned task skill JSON; repeat to add skills')
    p.add_argument('--planner',choices=['demo','file','command','openai'],default='openai');p.add_argument('--plans');p.add_argument('--command');p.add_argument('--model');p.add_argument('--base-url',default=os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1'))


def main():
    p=argparse.ArgumentParser(description='Evidence-gated CaP simulation pipeline');sub=p.add_subparsers(dest='op',required=True)
    run=sub.add_parser('run');run.add_argument('--task',required=True);run.add_argument('--out',required=True);run.add_argument('--resume',action='store_true');run.add_argument('--no-reference',action='store_true');run.add_argument('--reference-task',help='Original scene task used for physical reference recovery');planner_args(run)
    aug=sub.add_parser('augment');aug.add_argument('--task',required=True);aug.add_argument('--variants',required=True);aug.add_argument('--out',required=True);aug.add_argument('--reference-task',help='Original scene task used for physical reference recovery');aug.add_argument('--resume',action='store_true');planner_args(aug)
    imp=sub.add_parser('import-reference');imp.add_argument('--source',required=True);imp.add_argument('--out',required=True);imp.add_argument('--format',choices=['cap','g1-attempt','humano'],default='cap');imp.add_argument('--scene-xml')
    summary=sub.add_parser('summarize-reference');summary.add_argument('--reference',required=True);summary.add_argument('--out',required=True);summary.add_argument('--video');summary.add_argument('--interface');summary.add_argument('--contact-prior');planner_args(summary)
    asset=sub.add_parser('assets-from-video');asset.add_argument('--video',required=True);asset.add_argument('--out',required=True);asset.add_argument('--scene-plan');asset.add_argument('--mesh-command');planner_args(asset)
    build=sub.add_parser('assets-build');build.add_argument('--plan',required=True);build.add_argument('--out',required=True)
    bridge=sub.add_parser('task-from-assets');bridge.add_argument('--manifest',required=True);bridge.add_argument('--object-id',required=True);bridge.add_argument('--support-id',required=True);bridge.add_argument('--out',required=True)
    render=sub.add_parser('render-demo');render.add_argument('--attempt',required=True);render.add_argument('--out',required=True)
    native_render=sub.add_parser('render-native');native_render.add_argument('--attempt',required=True);native_render.add_argument('--out',required=True)
    contact=sub.add_parser('extract-contact-prior');contact.add_argument('--reference',required=True);contact.add_argument('--object-body',required=True);contact.add_argument('--wrist-body',required=True);contact.add_argument('--object-key',default='target');contact.add_argument('--out',required=True)
    skill_publish=sub.add_parser('publish-task-skill');skill_publish.add_argument('--draft',required=True);skill_publish.add_argument('--registry',required=True);skill_publish.add_argument('--parent',help='Current revision sha256; required after first version')
    args=p.parse_args()
    if args.op=='run':
        task=load_task(args.task)
        if args.no_reference:task['references']=[];task['skills']=[]
        add_task_skills(task,args.skill)
        result=run_pipeline(task,lambda out:make_planner(args,out),make_backend,args.out,reference_task=load_task(args.reference_task) if args.reference_task else None,resume=args.resume)
    elif args.op=='augment':
        task=load_task(args.task)
        add_task_skills(task,args.skill)
        result=augment_pipeline(task,read(args.variants),lambda out:make_planner(args,out),make_backend,args.out,reference_task=load_task(args.reference_task) if args.reference_task else None,resume=args.resume)
    elif args.op=='publish-task-skill':
        from .task_skills import publish_skill
        result=publish_skill(read(args.draft),args.registry,args.parent)
    elif args.op=='extract-contact-prior':
        from .contact_prior import extract_prior
        result=extract_prior(args.reference,object_body=args.object_body,wrist_body=args.wrist_body,object_key=args.object_key);write(args.out,result)
    elif args.op=='import-reference':
        from .trajectories import import_reference
        result=import_reference(args.source,args.out,args.format,args.scene_xml)
    elif args.op=='summarize-reference':
        from .experience import summarize
        result=summarize(args.reference,args.out,make_planner(args,args.out),args.video,interface=read(args.interface) if args.interface else None,contact_prior=read(args.contact_prior) if args.contact_prior else None)
    elif args.op=='assets-from-video':
        from .assets import from_video
        result=from_video(args.video,args.out,None if args.scene_plan else make_planner(args,args.out),args.scene_plan,json.loads(args.mesh_command) if args.mesh_command else None)
    elif args.op=='task-from-assets':
        from .assets import task_from_assets
        result=task_from_assets(args.manifest,args.object_id,args.support_id);write(args.out,result)
    elif args.op=='assets-build':
        from .assets import build_scene,validate_scene
        build_scene(read(args.plan),args.out);result=validate_scene(args.out)
    elif args.op=='render-native':
        from .render import render_native
        result=render_native(args.attempt,args.out)
    else:
        from .render import render_demo
        result=render_demo(args.attempt,args.out)
    print(json.dumps(result,ensure_ascii=False,indent=2))
