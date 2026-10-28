"""Explicit historical mixed-interface baseline; never a silent mode0 fallback."""
import argparse,copy,json,subprocess,time,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,mujoco
from probe_historical_apple import resolve_tree
from humanoid_mimicgen import transfer_feedback_direct as ex
from humanoid_mimicgen.workflow.contracts import now,file_hash
from humanoid_mimicgen.augmentation.agent import save
REPO=Path(__file__).resolve().parents[1];BASE=REPO.parent
p=argparse.ArgumentParser();p.add_argument('--condition',choices=['historical_y8','current_y2'],required=True);p.add_argument('--root',type=Path);a=p.parse_args();stamp=datetime.now(ZoneInfo('Asia/Shanghai'));root=a.root or BASE/'experiments'/stamp.strftime('%Y-%m/%d/%Y%m%d_%H%M_apple_static_legacy');root.mkdir(parents=True,exist_ok=True);Path('/tmp/static_legacy_root').write_text(str(root));case=root/a.condition;case.mkdir(exist_ok=False);started=now();tick=time.monotonic()
origin=BASE/'experiments/2026-10/18/20261018_1904_apple_contact_direct_y_plus8';old=json.loads((origin/'experiment.json').read_text());idx=json.loads((BASE/'experiments/_index/latest-migration.json').read_text());mapping=json.loads(Path(idx['path_map']).read_text());r=resolve_tree(copy.deepcopy(old),mapping);inputs=r['inputs'];initial=dict(np.load(inputs['initial']));targets=dict(np.load(inputs['targets']));manifest=json.loads(Path(inputs['transfer_manifest']).read_text());delta=np.zeros(3)
for key in ('initial','targets','model','reference_states','controls','source_controller','program'):
 assert file_hash(inputs[key])==old['preflight']['evidence_sha256'][old['inputs'][key]],key
if a.condition=='current_y2':
 current=BASE/'experiments/2026-10/26/20261026_0053_apple_wrist_position_ablation/apple_03/summary.json';d=json.loads(current.read_text());newcase=Path(d['arms']['baseline']['result']['experiment_path']);m=mujoco.MjModel.from_binary_path(str(newcase/'scene.mjb'));md=mujoco.MjData(m);bid=m.body('apple_001').id;adr=int(m.jnt_qposadr[m.body_jntadr[bid]]);md.qpos[:]=np.load(newcase/'initial.npz')['qpos'];mujoco.mj_forward(m,md);desired=md.xpos[bid].copy();md.qpos[:]=initial['qpos'];mujoco.mj_forward(m,md);delta=desired-md.xpos[bid];assert abs(delta[2])<1e-8
 initial['qpos'][adr:adr+3]+=delta;targets['right_pose'][:,:3]+=delta;manifest['target_object_anchor_pose'][:3]=(np.array(manifest['target_object_anchor_pose'][:3])+delta).tolist();manifest['translation_m']=(np.array(manifest['translation_m'])+delta).tolist()
np.savez_compressed(case/'initial.npz',**initial);np.savez_compressed(case/'targets.npz',**targets);manifest.update(experiment_id=root.name,standing='direct contact after1s approach; no pose gate');save(case/'transfer_manifest.json',manifest)
for k in ('initial','targets','transfer_manifest'):inputs[k]=str(case/(k+('.npz' if k!='transfer_manifest' else '.json')))
task=resolve_tree(json.loads(Path(inputs['task']).read_text()),mapping);task['resources']['initial_state']=inputs['initial'];save(case/'task.json',task);inputs['task']=str(case/'task.json')
subprocess.run([str(REPO/'scripts/humano_python.sh'),str(REPO/'scripts/check_humano_runtime.py'),'--load-models','--out',str(case/'runtime.json')],check=True,stdout=subprocess.DEVNULL);runtime=json.loads((case/'runtime.json').read_text())
r.update(experiment_id=root.name+'_'+a.condition,created_at=started,record_status='frozen_ready');r['runtime'].update(actual_loaded_session_evidence=str(case/'runtime.json'),model_hashes=runtime['model_sha256'])
r['purpose']={'hypothesis':'Reproduce the successful historical wide-position static retargeting method, then test exact current Y+2cm apple location','baseline_paths':[str(origin)],'planned_changes':[{'condition':a.condition,'relative_to_historical_y8_delta':delta.tolist()},'Derived path/provenance records only; explicitly retain original mixed interface'],'invariants':['Source hands, all left/torso/root reference targets, initial robot, physics and original validator','250 original token ticks +150 mode0 ticks, then fixed-source-leg mode1','No postdecoder body correction or API'],'known_confounders':['This full-method diagnostic is not an isolated controller comparison or the pending ten-condition benchmark','Historical held-object criterion differs from native9cm criterion; report both','Source is recovered measured contact trajectory; current API pipeline uses raw dataset reference']}
r['control']['object_translation_m']=manifest['translation_m'];r['pipeline'].update(entrypoint='scripts/probe_apple_static_legacy.py -> transfer_feedback_direct.run',external_model_calls='0; local diagnostic only');r['budget']['attempts_per_case']=1
bound=[Path(v) for v in inputs.values() if isinstance(v,str) and Path(v).is_file()]+[origin/'experiment.json',Path(__file__).resolve(),REPO/'scripts/probe_historical_apple.py',case/'runtime.json',Path(r['physics']['physics_parameters_manifest'])]
bound+=list((REPO/'humanoid_mimicgen').glob('*.py'))+list((REPO/'src/cap_sim/backends').glob('*.py'))+[REPO/'scripts/humano_python.sh',REPO/'scripts/check_humano_runtime.py']
models=Path(task['resources']['sonic_models']);bound+=list(models.glob('policy/release/*.onnx'))+[models/'planner/target_vel/V2/planner_sonic.onnx']
for name in ('grasp-prior-refinement','g1-grasp-runtime-review','sim-visual-realism-review'):bound.append(Path('/home/kinetra-4090-01/.codex/skills')/name/'SKILL.md')
from cap_sim.backends.humano import Backend
from humanoid_mimicgen.privileged import no_physics
with no_physics():
 b=Backend();b.describe(task)
 try:
  physics=json.loads(Path(r['physics']['physics_parameters_manifest']).read_text());assert all(np.array_equal(getattr(b.m,k),np.array(v)) for k,v in physics['arrays'].items());import inspect
  bound+= [Path(inspect.getfile(cls)) for cls in type(b.e.task).__mro__ if cls.__module__.startswith('humano.')]
 finally:b.close()
snapshot=case/'source_snapshot';snapshot.mkdir()
for i,path in enumerate(bound):
 if path.suffix=='.py':shutil.copy2(path,snapshot/f'{i:03d}_{path.name}')
proof={'physics_steps':0,'historical_input_hashes_checked':True,'actual_physics_arrays_equal':True,'declared_interface':'250 original tokens +150 mode0 then mode1','delta_from_y8_m':delta.tolist(),'new_api_calls':0};save(case/'preflight_evidence.json',proof);bound.append(case/'preflight_evidence.json')
hashes={str(f):file_hash(f) for f in bound};r['preflight'].update(passed=True,evidence_paths=list(hashes),evidence_sha256=hashes);r['pipeline']['source_snapshot_and_hashes']=hashes;save(case/'comparison_diff.json',r['purpose']);r['preflight']['comparison_diff_path']=str(case/'comparison_diff.json');save(case/'experiment.json',r);(case/'experiment.sha256').write_text(file_hash(case/'experiment.json'))
result={'started_at':started,'condition':a.condition,'case':str(case)}
try:
 result['audit']=ex.run(case/'experiment.json');result['verification']=ex.verify(case/'experiment.json');ex.check_contract(case/'experiment.json');obs=json.loads((case/'execution/observations.json').read_text());held=[o for o in obs if o['stage']=='hold'];result['hold_valid_fraction']=sum(o['clearance_m']>=.01 and o['contacts']['hand_object_N']>.1 and o['contacts']['support_N']<.1 for o in held)/len(held) if held else 0
 if a.condition=='historical_y8':
  x=np.load(case/'execution/states.npz')['samples'];y=np.load(origin/'execution/states.npz')['samples'];result['historical_state_exact']=np.array_equal(x,y)
except Exception as e:
 import traceback
 result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 result.update(ended_at=now(),wall_s=time.monotonic()-tick,api_calls=0);save(case/'summary.json',result);print(json.dumps(result,ensure_ascii=False))
