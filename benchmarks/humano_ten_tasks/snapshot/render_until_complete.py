import subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;start=time.time()
while time.time()-start<7200:
 subprocess.run([sys.executable,str(r/'render_all.py')],check=True)
 n=len(list((r/'runs').glob('*/*/summary.json')));v=len(list((r/'videos').glob('*/*/verification.json')))
 print({'completed_cases':n,'video_pairs':v},flush=True)
 if n==20 and v==20:break
 time.sleep(20)
else:raise TimeoutError('Video watcher deadline; case processes are not restarted')
