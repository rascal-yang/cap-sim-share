"""API-selected read-only evidence queries; no task strategies or physics steps."""
import json
from pathlib import Path
import numpy as np
from .agent import save
from ..workflow.contracts import file_hash

ARTIFACTS={'controller':'execution/controller.npz','trajectory':'execution/trajectory.npz',
 'metrics':'execution/metrics.json','contact_control':'execution/contact_control.json',
 'physics':'physics.json','result':'result.json','model':'scene.mjb','aligned':'execution/trajectory.npz'}

def query_case(cases,q):
    if set(q)!={'case','artifact','start','stop','stride'}:raise ValueError('Invalid query fields')
    if q['case'] not in cases or q['artifact'] not in ARTIFACTS:raise ValueError('Unknown case/artifact')
    a,z,s=(q[k] for k in ('start','stop','stride'))
    if any(type(x) is not int for x in (a,z,s)) or not 0<=a<z or not 1<=s<=1000 or z-a>4096 or len(range(a,z,s))>32:raise ValueError('Query permits at most 32 rows across 4096 indices')
    p=Path(cases[q['case']])/ARTIFACTS[q['artifact']]
    if not p.is_file():raise FileNotFoundError('Artifact not recorded: '+q['artifact'])
    out={'case':q['case'],'artifact':q['artifact'],'source':str(p),'source_sha256':file_hash(p),'physics_steps':0,'range':[a,z,s]}
    if q['artifact']=='aligned':
        import mujoco
        case=Path(cases[q['case']]);model=case/'scene.mjb';controller=case/'execution/controller.npz'
        m=mujoco.MjModel.from_binary_path(str(model))
        ids=[i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE and '_hand_' not in m.joint(i).name][:29]
        logs_path=case/'execution/contact_control.json';metrics_path=case/'execution/metrics.json'
        logs=json.loads(logs_path.read_text()) if logs_path.exists() else []
        logs={row['frame']:row for row in logs};metrics=json.loads(metrics_path.read_text())
        out['additional_source_sha256']={str(f):file_hash(f) for f in (model,controller,metrics_path,logs_path) if f.exists()}
        out['joint_names']=[m.joint(i).name for i in ids];out['frames']=[]
        with np.load(p,allow_pickle=False) as trace, np.load(controller,allow_pickle=False) as ctrl:
            for i in range(a,min(z,len(ctrl['records'])),s):
                lo=i*4;hi=min(lo+4,len(trace['state']));state=trace['state'][lo];row=ctrl['records'][i]
                if not np.isclose(state[0],row[0],atol=1e-8,rtol=0):raise ValueError('Controller/state timestamps disagree')
                command=trace['commands'][lo];measured=state[1+m.jnt_qposadr[ids]];end=trace['next_state'][hi-1]
                metric=next((x for x in metrics if abs(x['time_s']-end[0])<1e-8),None)
                out['frames'].append({'frame':i,'pre_time_s':float(state[0]),'post_time_s':float(end[0]),'root_xyz_pre_m':state[1:4].tolist(),
                    'root_quat_wxyz_pre':state[4:8].tolist(),'root_xyz_post_m':end[1:4].tolist(),'root_quat_wxyz_post':end[4:8].tolist(),
                    'measured_q_pre_rad':measured.tolist(),'raw_decoder_target_rad':row[-29:].tolist(),'executed_target_rad':command[1:30].tolist(),
                    'target_minus_measured_rad':(command[1:30]-measured).tolist(),'actuator_ctrl_substeps':trace['commands'][lo:hi,-m.nu:].tolist(),
                    'pre_control_log':logs.get(i),'post_metrics':metric,'clipping_flags':None})
        out['alignment']='Policy frame i: pre-control measured state and decoded/final targets, up to four 200Hz actuator commands, metrics matched to final substep timestamp. Missing pre_control_log/clipping flags are unknown, not zero; no reconstruction or physics integration.'
    elif q['artifact']=='model':
        import mujoco
        m=mujoco.MjModel.from_binary_path(str(p))
        out['model']={'nq':m.nq,'nv':m.nv,'nu':m.nu,'joints':[{'name':m.joint(i).name,'qpos_address':int(m.jnt_qposadr[i]),'dof_address':int(m.jnt_dofadr[i]),'range':m.jnt_range[i].tolist(),'axis':m.jnt_axis[i].tolist()} for i in range(m.njnt)],'actuators':[m.actuator(i).name for i in range(m.nu)]}
    elif p.suffix=='.npz':
        with np.load(p,allow_pickle=False) as data:
            out['shapes']={k:list(data[k].shape) for k in data.files}
            if q['artifact']=='controller':
                rows=data['records'][a:z:s];out['arrays']={'records':np.concatenate([rows[:,:2],rows[:,-29:]],axis=1).tolist() if rows.shape[1]>31 else rows.tolist()}
                out['alignment']='50Hz pre-control; returned columns time, mode, 29 raw decoder targets; latent/encoder omitted.'
            else:
                out['arrays']={k:data[k][a:z:s].tolist() for k in data.files if k!='wrist_error_m'}
                out['alignment']='200Hz substeps. state/next_state columns time,qpos(nq),qvel(nv); commands source row,29 final targets,actuator ctrl(nu). Query model for addresses. Policy row i begins at substep 4*i; use timestamps, truncated final ticks possible.'
    else:
        d=json.loads(p.read_text());out['rows']=d[a:z:s] if isinstance(d,list) else d
        out['alignment']='metrics: post-control 50Hz (time_s); contact_control: pre-control 50Hz. Match timestamps, not equal row indices.'
    return out


def explore_decision(provider,root,context,episodes,*,decide,decision_tools,validator,skill_path,cases,budget):
    """The model selects queries/proceed/stop, then issues unchanged signed decisions."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);results=[];blocker=None
    props={'action':{'type':'string','enum':['inspect','proceed','stop']},'rationale':{'type':'string'},'query':{'anyOf':[{'type':'null'},{'type':'object','additionalProperties':False,'properties':{'case':{'type':'string','enum':list(cases)},'artifact':{'type':'string','enum':list(ARTIFACTS)},'start':{'type':'integer'},'stop':{'type':'integer'},'stride':{'type':'integer'}},'required':['case','artifact','start','stop','stride']}]}}
    tools=[{'type':'function','name':'decide','description':'Choose read-only evidence query, proceed to stage decision, or stop with concrete missing capability.','strict':True,'parameters':{'type':'object','properties':props,'required':list(props),'additionalProperties':False}}]
    def valid(d,ep):
        if set(d)!=set(props) or d['action'] not in ('inspect','proceed','stop') or not d['rationale'].strip():raise ValueError('Invalid exploration decision')
        if (d['action']=='inspect')!=(d['query'] is not None):raise ValueError('Query required only for inspect')
        return d
    while True:
        if budget['remaining']<=0:raise TimeoutError('API exploration decision budget exhausted')
        budget['remaining']-=1
        step=root/f'exploration_{len(results):03d}';step.mkdir(exist_ok=True)
        from .quality_suite import API_CURSOR
        API_CURSOR[str(step)]=0
        c=context|{'exploration_stage':context.get('stage'),'stage':'explore','cases':cases,'artifacts':ARTIFACTS,'tool_results':results,'last_blocker':blocker,'remaining_exploration_decisions':budget['remaining'],
          'instructions':'You own exploration. Inspect any available replay or own-run evidence. 32 rows/query maximum; stride enables scans then zoom. Missing fields are unknown, not zero. Proceed to submit an audit, executable experimental plan or review. Diagnostic experiments use the normal plan and physical budget. needs_tool returns here; stop only for a concrete unavailable capability or exhausted useful exploration. Never require supervisor tuning.'}
        d,dp=decide(provider,step,c,episodes,tools=tools,validator=valid,skill_path=skill_path)
        if d['action']=='stop':
            save(root/'exploration_blocked.json',{'decision_path':str(dp),'rationale':d['rationale']})
            raise RuntimeError('API exploration blocked: '+d['rationale'])
        if d['action']=='inspect':
            try:r=query_case(cases,d['query'])
            except (ValueError,FileNotFoundError,KeyError) as exc:r={'error':str(exc),'physics_steps':0,'missing_data_is_unknown':True}
            save(step/'tool_result.json',{'query':d['query'],'result':r});results.append({'query':d['query'],'result':r});continue
        decision,path=decide(provider,root,context|{'tool_results':results,'last_blocker':blocker},episodes,tools=decision_tools,validator=validator,skill_path=skill_path)
        if decision.get('action',decision.get('assessment'))!='needs_tool':return decision,path
        blocker=decision;results.append({'decision_requested_more_evidence':decision})
