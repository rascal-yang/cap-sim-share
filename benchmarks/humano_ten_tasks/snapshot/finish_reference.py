import subprocess,sys,time,json
from pathlib import Path
r=Path(__file__).resolve().parent
assert (r/'no_reference_frozen.json').exists()
report=[]
for ref in sorted((r/'references').glob('G1*.json')):
 task=ref.stem;out=r/'runs/reference_augmented'/task
 if (out/'summary.json').exists():continue
 args=[sys.executable,str(r/'run_case.py'),'--task',task,'--group','reference_augmented','--auto-policy']
 if out.exists():args+=['--resume']
 start=time.time()
 with (r/('reference_'+task+'.log')).open('a') as log:p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
 report.append({'task':task,'returncode':p.returncode,'wall_s':time.time()-start});(r/'reference_execution.json').write_text(json.dumps(report,indent=2));print(report[-1],flush=True)
 if p.returncode:break
