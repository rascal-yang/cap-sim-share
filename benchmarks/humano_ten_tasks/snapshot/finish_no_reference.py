import subprocess,sys,time,json,os
from pathlib import Path
r=Path(__file__).resolve().parent;report=[]
for name in ['TableApplePick','TableButtonPress','GroundBottlePick','GroundBoxPick','TablePlatePick']:
 start=time.time();task='G1Wholebody'+name+'Teleop'
 with (r/('direct_'+name+'.log')).open('w') as log:p=subprocess.run([sys.executable,str(r/'run_case.py'),'--task',task,'--group','no_reference','--resume','--auto-policy'],stdout=log,stderr=subprocess.STDOUT)
 report.append({'task':task,'returncode':p.returncode,'wall_s':time.time()-start});(r/'direct_execution.json').write_text(json.dumps(report,indent=2));print(report[-1],flush=True)
 if p.returncode:break
