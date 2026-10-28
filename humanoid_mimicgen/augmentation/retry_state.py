"""Contact-route checkpoint provenance and API-selected branch initialization."""
import copy,json,math
from pathlib import Path
import numpy as np
from .agent import save
from ..workflow.contracts import file_hash

RESUME={'type':'object','properties':{'kind':{'type':'string','enum':['restart','checkpoint']},'case':{'type':'string'},'frame':{'type':'integer','minimum':0},'rationale':{'type':'string'}},'required':['kind','case','frame','rationale'],'additionalProperties':False}


def budgets(config,limit):
    initial=config.get('initial_wall_budget_s',900);retry=config.get('retry_wall_budget_s',1200);total=config.get('task_wall_budget_s',initial+retry*(limit-1))
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in (initial,retry,total)):raise ValueError('Invalid retry time budget')
    return initial,retry,total


def phase_deadline(current,total_deadline,phase_budget):return min(total_deadline,current+phase_budget)


def validate_resume(value,offers):
    if not isinstance(value,dict) or set(value)!=set(RESUME['properties']) or value['kind'] not in ('restart','checkpoint') or type(value['frame']) is not int or value['frame']<0 or not isinstance(value['rationale'],str) or not value['rationale'].strip():raise ValueError('Invalid resume decision')
    if value['kind']=='restart':
        if value['case'] or value['frame']!=0:raise ValueError('Restart requires empty case and frame 0')
    elif not any(x['case']==value['case'] and 0<=value['frame']<=x['last_frame'] for x in offers):raise ValueError('Checkpoint not offered or frame outside completed attempt')
    return value


def pack(x):
    if isinstance(x,np.ndarray):return {'__array__':x.tolist(),'dtype':str(x.dtype)}
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    if x is None or isinstance(x,(str,int,float,bool)):return x
    raise TypeError('Unsupported checkpoint state '+type(x).__name__)


def unpack(x):
    if isinstance(x,dict):return np.asarray(x['__array__'],dtype=x['dtype']) if '__array__' in x else {k:unpack(v) for k,v in x.items()}
    if isinstance(x,list):return [unpack(v) for v in x]
    return x


def task_state(task):
    # Mutable acceptance state used by the ten supported HUMANO task checkers.
    return {k:copy.deepcopy(v) for k,v in vars(task).items() if k.startswith('_init_') or k.endswith('_hold_count') or k in ('_contact_started','reward','_target_body_name')}


def capture(b,s,adapter,frame,ever):
    import mujoco
    kind=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(b.m,kind));mujoco.mj_getState(b.m,b.d,state,kind)
    meta=pack({'frame':frame,'ever':ever,'task':task_state(b.e.task),'sonic':{'history':list(s.history),'last':s.last,'tick':s.tick},'adapter':{k:getattr(adapter,k) for k in ('servo','previous','hold','shadow_ever')}})
    return state,meta


def restore(b,s,adapter,state,meta):
    import mujoco
    meta=unpack(meta);kind=mujoco.mjtState.mjSTATE_INTEGRATION
    if len(state)!=mujoco.mj_stateSize(b.m,kind):raise ValueError('Checkpoint physics dimension mismatch')
    mujoco.mj_setState(b.m,b.d,state,kind);warm=b.d.qacc_warmstart.copy();mujoco.mj_forward(b.m,b.d);b.d.qacc_warmstart[:]=warm
    for k,v in meta['task'].items():setattr(b.e.task,k,v)
    s.history.clear();s.history.extend(meta['sonic']['history']);s.last=meta['sonic']['last'];s.tick=meta['sonic']['tick']
    for k,v in meta['adapter'].items():setattr(adapter,k,v)
    actual=np.empty_like(state);mujoco.mj_getState(b.m,b.d,actual,kind)
    if not np.array_equal(actual,state):raise ValueError('Checkpoint integration-state restore mismatch')
    return bool(meta['ever'])


def write_checkpoints(case,states,metadata):
    case=Path(case);out=case/'execution';np.savez_compressed(out/'checkpoints.npz',integration_state=np.asarray(states));save(out/'checkpoints.json',metadata)
    files=[case/n for n in ['scene.mjb','physics.json','task.json','result.json','experiment.json','experiment.sha256']]+[out/n for n in ['checkpoints.npz','checkpoints.json','trajectory.npz','controller.npz','metrics.json','contact_control.json']]
    save(out/'checkpoint_manifest.json',{'version':1,'boundary':'pre_control_50Hz','last_frame':len(states)-1,'sha256':{str(p):file_hash(p) for p in files}})


def offer(case):
    case=Path(case).resolve();p=case/'execution/checkpoint_manifest.json'
    if not p.exists():return None
    m=json.loads(p.read_text());return {'case':str(case),'last_frame':m['last_frame'],'boundary':'before executing frame; 50Hz','manifest_sha256':file_hash(p)}


def load(case,frame):
    case=Path(case).resolve();p=case/'execution/checkpoint_manifest.json';m=json.loads(p.read_text())
    if m['version']!=1 or type(frame) is not int or not 0<=frame<=m['last_frame']:raise ValueError('Unsupported checkpoint')
    for path,digest in m['sha256'].items():
        if file_hash(path)!=digest:raise ValueError('Checkpoint source changed '+path)
    if file_hash(case/'experiment.json')!=(case/'experiment.sha256').read_text():raise ValueError('Parent frozen contract changed')
    out=case/'execution';states=np.load(out/'checkpoints.npz')['integration_state'];metadata=json.loads((out/'checkpoints.json').read_text())
    if len(states)!=len(metadata) or metadata[frame]['frame']!=frame:raise ValueError('Checkpoint frame mapping mismatch')
    trace=np.load(out/'trajectory.npz');records=np.load(out/'controller.npz')['records'];logs=json.loads((out/'contact_control.json').read_text());metrics=json.loads((out/'metrics.json').read_text())
    if len(trace['state'])<frame*4:raise ValueError('Incomplete parent prefix')
    return {'case':case,'frame':frame,'states':states,'metadata':metadata,'trace':trace,'records':records,'logs':logs,'metrics':metrics,'manifest':m,'evidence':[p,*map(Path,m['sha256'])],'result':json.loads((case/'result.json').read_text())}
