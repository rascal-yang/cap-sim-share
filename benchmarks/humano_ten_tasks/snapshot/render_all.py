import json,time,subprocess,sys,shutil
from pathlib import Path
r=Path(__file__).resolve().parent;timings=[]
for p in sorted((r/'runs').glob('*/*/summary.json')):
 s=json.loads(p.read_text());out=r/'videos'/p.parent.parent.name/p.parent.name
 if (out/'verification.json').exists():continue
 if out.exists():
  archive=r/'render_failures'/p.parent.parent.name/(p.parent.name+'_'+str(int(time.time())));archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(archive))
 out.parent.mkdir(parents=True,exist_ok=True)
 attempt=p.parent/'attempts'/f"{s['accepted_attempt'] or s['attempts']:03d}"
 if not (attempt/'states.npz').exists():print('MISSING_STATES',attempt,flush=True);continue
 start=time.time();log=r/(p.parent.parent.name+'_'+p.parent.name+'_render.log')
 with log.open('w') as f:proc=subprocess.run([sys.executable,'-m','cap_sim','render-native','--attempt',str(attempt),'--out',str(out)],stdout=f,stderr=subprocess.STDOUT)
 t={'case':str(p.parent),'wall_s':time.time()-start,'returncode':proc.returncode,'source_attempt':str(attempt)};timings.append(t);print(t,flush=True)
 if out.exists():(out/'render_timing.json').write_text(json.dumps(t,indent=2))
(r/'render_batch_timing.json').write_text(json.dumps(timings,indent=2))
