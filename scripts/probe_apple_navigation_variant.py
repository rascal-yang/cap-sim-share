"""Run an explicit local navigation-only variation of a frozen diagnostic request."""
import argparse,json,copy,time
from pathlib import Path
from humanoid_mimicgen.workflow import service
from humanoid_mimicgen.workflow.contracts import now,file_hash
from humanoid_mimicgen.augmentation.agent import save
p=argparse.ArgumentParser();p.add_argument('request',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False);r=json.loads(a.request.read_text());start=now();tick=time.monotonic()
r['strategy']['policy']['navigation'].update(profile='distance_velocity',acceleration_m_s2=.5,response_s=.6)
r['experiment_root']=str(a.output/'runs');r['strategy_root']=str(a.output/'strategies');r['files']['navigation_variant_script']=str(Path(__file__).resolve())
study=json.loads(Path(r['files']['study_case']).read_text());study['purpose'].update(hypothesis='Does smooth navigation and standing handoff improve capture at the identical apple position and stance target?',baseline_paths=[str(a.request)],planned_changes=['Navigation distance velocity profile: max0.5m/s, command acceleration0.5m/s2, response0.6s; smooth standing reference transition'],known_confounders=['Combined velocity shaping and handoff blend, cannot attribute to either alone','No wrist/finger/contact changes; actual achieved stance and replanned approach can differ'])
save(a.output/'study_case.json',study);r['files']['study_case']=str(a.output/'study_case.json');save(a.output/'request.json',r);result={'started_at':start,'api_calls':0}
try:
 plan=service.plan(r);result['plan']=plan;prep=service.prepare(plan['strategy_id'],r);result['prepared']=prep;result['execution']=service.execute(prep['experiment_path'])
except Exception as e:
 import traceback
 result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 result.update(ended_at=now(),wall_s=time.monotonic()-tick);save(a.output/'summary.json',result);print(json.dumps(result,ensure_ascii=False))
