from pathlib import Path
import numpy as np
from cap_sim.io import read,write
from cap_sim.backends.humano import Backend
from cap_sim.pipeline import run_pipeline
from policy import Planner
r=Path(__file__).resolve().parent
class DurableBackend(Backend):
 def execute(self,a,remaining):
  result=super().execute(a,remaining);np.savez_compressed(self.out/'states.npz',samples=np.asarray(self.states));write(self.out/'contacts.json',self.contact_records);return result
backends=[]
def factory(t):
 b=DurableBackend();backends.append(b);return b
try:
 result=run_pipeline(read(r/'target_task.json'),Planner,factory,r/'run',reference_task=read(r/'original_task.json'),resume=(r/'run').exists());print(result,flush=True)
finally:
 for b in backends:b.close()
