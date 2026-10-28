"""Subprocess adapter for the preserved G1/Dex3 SONIC runtime.
External assets/weights stay outside Git. No execution from restored success checkpoints.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from ..io import read, write

class Backend:
    interface = {'backend':'g1_sonic','action':{'stage':'approach/contact/lift/hold/recover','duration_s':'.1..8','wrist_position':'world xyz','wrist_quaternion_wxyz':'unit quaternion','fingers':'7 joint targets','pelvis_height_m':'.64..74'},'gates':'original SONIC runtime; opposed contact >= .12s before lift; lift displacement <= .031m'}
    def describe(self,task):
        with tempfile.TemporaryDirectory(prefix='cap_g1_inspect_') as tmp:
            self.task=task;self.out=Path(tmp)
            return self._call('inspect')['observation']

    def reset(self,task,out):
        if task['acceptance']!={'lift_m':.1,'hold_s':1.}:raise ValueError('Current G1 runtime supports exactly .10 m / 1 s acceptance')
        if task['budget']['sim_seconds']>30:raise ValueError('G1 runtime is limited to 30 simulation seconds')
        self.task=task;self.out=Path(out).resolve();self.states=[];self.contacts=[]
        write(self.out/'worker_task.json',task)
        return self._call('reset')['observation']
    def _call(self,op,action=None):
        req=self.out/'worker_request.json';write(req,{'op':op,'task':self.task,'out':str(self.out),'action':action})
        resources=self.task['resources'];python=resources.get('python',sys.executable)
        worker=Path(__file__).with_name('g1_worker.py')
        p=subprocess.run([python,str(worker),str(req)],capture_output=True,text=True,timeout=180)
        with (self.out/'worker.log').open('a') as f:f.write(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError('G1 runtime failed; see '+str(self.out/'worker.log'))
        return read(self.out/'worker_response.json')
    def execute(self,action,remaining):
        if set(action)-{'stage','duration_s','wrist_position','wrist_quaternion_wxyz','fingers','pelvis_height_m','name'}:raise ValueError('Unknown G1 control channel')
        if action.get('stage') not in ('approach','contact','lift','hold','recover'):raise ValueError('Invalid G1 stage')
        duration=action.get('duration_s',0)
        if not isinstance(duration,(int,float)) or not np.isfinite(duration) or not .1<=duration<=8:raise ValueError('Invalid G1 duration')
        if abs(duration/.005-round(duration/.005))>1e-7:raise ValueError('G1 duration must align to 5 ms steps')
        height=action.get('pelvis_height_m',.68)
        if not isinstance(height,(int,float)) or not np.isfinite(height) or not .64<=height<=.74:raise ValueError('Invalid pelvis height')
        for key,size in [('wrist_position',3),('wrist_quaternion_wxyz',4),('fingers',7)]:
            if key in action:
                v=np.asarray(action[key],dtype=float)
                if v.shape!=(size,) or not np.isfinite(v).all():raise ValueError('Invalid '+key)
                if size==4 and abs(np.linalg.norm(v)-1)>.001:raise ValueError('Wrist quaternion must be unit length')
        if action['duration_s']>remaining+1e-9:return {'completed':False,'reusable':False,'failure':'execution_deadline','samples':[],'observation':{}}
        r=self._call('phase',action)
        if (self.out/'worker_states.npz').exists():
            s=np.load(self.out/'worker_states.npz')['samples'];self.states.extend(s.tolist())
        self.contacts.extend(r.pop('contacts',[]))
        return r
    def export(self,out):
        np.savez_compressed(Path(out)/'states.npz',samples=np.asarray(self.states))
        write(Path(out)/'contacts.json',self.contacts)
    def metrics(self):
        s=np.asarray(self.states)
        if len(s)<2:return {'roughness':0.,'root_omega_rms':0.,'peak_table_force':0.}
        # All measured actuated joint velocities; exclude initial settle when enough data.
        s=s[s[:,0]>3] if np.sum(s[:,0]>3)>2 else s
        acc=np.diff(s[:,64:107],axis=0)/.005
        return {'roughness':float(np.sqrt(np.mean(acc*acc))), 'root_omega_rms':float(np.sqrt(np.mean(np.sum(s[:,61:64]**2,axis=1)))), 'peak_table_force':max((c['all_robot_table_N'] for c in self.contacts),default=0.)}
