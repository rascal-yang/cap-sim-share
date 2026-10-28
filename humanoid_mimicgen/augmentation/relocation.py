"""Create new runnable inputs from moved historical evidence; never rewrite originals."""
import copy
import json
from pathlib import Path
from .contracts import require,verify_inputs
from ..workflow.contracts import canonical,file_hash


def rebase_baseline(source,destination,path_map):
    source=Path(source).resolve();destination=Path(destination).resolve();path_map=Path(path_map).resolve()
    mapping=json.loads(path_map.read_text());prefixes=sorted(mapping,key=len,reverse=True)
    def relocate(value):
        if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
        if isinstance(value,list):return [relocate(v) for v in value]
        if isinstance(value,str):
            for old in prefixes:
                if value==old or value.startswith(old+'/'):return mapping[old]+value[len(old):]
        return value
    request=relocate(json.loads(source.read_text()))
    old_task=Path(request['files']['task']);old_record=Path(request['files']['record'])
    record=relocate(json.loads(old_record.read_text()))
    # Relocation is authorized only when the original bound bytes still match.
    verify_inputs(record['preflight']['evidence_sha256'])
    task=relocate(json.loads(old_task.read_text()))
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'task.json').write_bytes(canonical(task))
    proof={'source_request':str(source),'source_request_sha256':file_hash(source),
           'source_record':str(old_record),'source_record_sha256':file_hash(old_record),
           'source_task':str(old_task),'source_task_sha256':file_hash(old_task),
           'path_map':str(path_map),'path_map_sha256':file_hash(path_map),
           'operation':'New derived request/template/task; historical bytes verified and left unchanged','physics_steps':0}
    (destination/'relocation.json').write_bytes(canonical(proof))
    record['pipeline']['relocation']=proof
    record['preflight']['evidence_sha256'].update({str(destination/'task.json'):file_hash(destination/'task.json'),
                                                  str(destination/'relocation.json'):file_hash(destination/'relocation.json')})
    (destination/'record.json').write_bytes(canonical(record))
    request['files'].update(task=str(destination/'task.json'),record=str(destination/'record.json'),relocation=str(destination/'relocation.json'))
    (destination/'request.json').write_bytes(canonical(request))
    return destination/'request.json'
