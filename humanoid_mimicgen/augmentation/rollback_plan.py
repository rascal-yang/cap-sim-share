"""Compact authenticated API patches compiled into immutable local execution plans."""
import copy,json,math
from pathlib import Path
import numpy as np
from .agent import save
from .quality_suite import verify_origin
from ..workflow.contracts import file_hash,now

FIELDS={'chair':{'tail_s','feedback_timescale_s','max_offset_rad','max_rate_rad_s'},'button':{'tail_s','translation_m','rotation_vector_rad','feedback_gain_z','index_offset_rad'},'apple':{'tail_s','wrist_delta_m'}}


def compile_parameters(profile,base,p):
    if profile not in FIELDS or set(p)!=FIELDS[profile]:raise ValueError('Unexpected patch fields')
    d=copy.deepcopy(base)
    if type(p['tail_s']) not in (int,float) or not math.isfinite(p['tail_s']) or not 0<=p['tail_s']<=(5 if profile=='button' else 3) or abs(p['tail_s']/.02-round(p['tail_s']/.02))>1e-6:raise ValueError(f'tail_s={p["tail_s"]}: require20ms multiple within allowed range')
    d['tail_s']=p['tail_s']
    if profile=='chair':
        for k,lo,hi in [('feedback_timescale_s',.4,2),('max_offset_rad',.05,.3),('max_rate_rad_s',.05,.4)]:
            if type(p[k]) not in (int,float) or not math.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError('Invalid '+k)
            d['leg_response'][k]=p[k]
    elif profile=='button':
        from .encoder_wrist import validate_parameters
        for k in ('translation_m','rotation_vector_rad'):d['preencoder'][k]=p[k]
        z=p['feedback_gain_z'];o=p['index_offset_rad']
        if type(z) not in (int,float) or not math.isfinite(z) or not -.005<=z<=0 or type(o) not in (int,float) or not math.isfinite(o) or not 0<=o<=d['finger_control']['max_offset_rad']:raise ValueError('Invalid force gain/index offset')
        d['preencoder']['force_gain_m_per_N_s']=[0,0,z]
        for j in d['finger_control']['joints']:j['offset_rad']=o
        for key,limit in [('translation_m',.05),('rotation_vector_rad',.35)]:
            norm=float(np.linalg.norm(np.asarray(d['preencoder'][key],float)))
            if norm>limit+1e-12:raise ValueError(f'{key} Euclidean norm {norm:.8f} exceeds {limit}; vector={d["preencoder"][key]}; fix vector rather than resubmit unchanged')
        validate_parameters(d['preencoder'])
    else:
        v=np.asarray(p['wrist_delta_m'],float)
        if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)>.05+1e-12:raise ValueError('Invalid wrist offset')
        d['wrist_delta_m']=v.tolist()
    if d['contact_control']['joints']:raise ValueError('Body residual forbidden')
    return d


def patch_from_plan(profile,d):
    if profile=='chair':return {'tail_s':d['tail_s'],**{k:d['leg_response'][k] for k in FIELDS[profile]-{'tail_s'}}}
    if profile=='button':return {'tail_s':d['tail_s'],'translation_m':d['preencoder']['translation_m'],'rotation_vector_rad':d['preencoder']['rotation_vector_rad'],'feedback_gain_z':d['preencoder']['force_gain_m_per_N_s'][2],'index_offset_rad':d['finger_control']['joints'][0]['offset_rad']}
    return {'tail_s':d['tail_s'],'wrist_delta_m':d['wrist_delta_m']}


def assemble(base,patch,profile,resume,rationale):
    d=compile_parameters(profile,base,patch);d.update(action='run',rationale=rationale,hypothesis='Execute authenticated shared API candidate under paired retry benchmark',resume=copy.deepcopy(resume));return d


def write_compiled(path,batch_path,index,spec_path,resume):
    batch=verify_origin(batch_path);spec=json.loads(Path(spec_path).read_text());candidate=batch['decision']['candidates'][index]
    d=assemble(spec['base_plan'],candidate['parameters'],spec['profile'],resume,candidate['rationale'])
    origin=copy.deepcopy(batch['origin']);origin.update(kind='api_compiled',api_batch_path=str(batch_path),api_batch_sha256=file_hash(batch_path),candidate_index=index,spec_path=str(spec_path),spec_sha256=file_hash(spec_path),compiler_sha256=file_hash(__file__),local_resume=resume)
    save(path,{'decision':d,'origin':origin,'created_at':now()});verify_compiled(path);return d


def verify_compiled(path):
    rec=json.loads(Path(path).read_text());o=rec['origin']
    if o['kind']!='api_compiled' or file_hash(o['api_batch_path'])!=o['api_batch_sha256'] or file_hash(o['spec_path'])!=o['spec_sha256'] or file_hash(__file__)!=o['compiler_sha256']:raise ValueError('Compiled provenance changed')
    batch=verify_origin(o['api_batch_path']);spec=json.loads(Path(o['spec_path']).read_text());c=batch['decision']['candidates'][o['candidate_index']]
    if assemble(spec['base_plan'],c['parameters'],spec['profile'],o['local_resume'],c['rationale'])!=rec['decision']:raise ValueError('Compiled parameters differ from authenticated patch')
    return rec
