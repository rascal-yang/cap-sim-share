import json
import pytest
from cap_sim.core import evaluate, run_task
from cap_sim.io import read

def task():
    return {'name':'lift','backend':'test','scene':{},'acceptance':{'lift_m':.1,'hold_s':1.},'budget':{'attempts':3,'sim_seconds':10,'optimization_attempts':1}}

def trace(lift=.11):
    return [{'time':i*.1,'dt':.1,'lift_m':lift,'support_force':0.,'hand_force':1.,'stable':True} for i in range(1,12)]

def test_acceptance_measured_and_continuous():
    assert evaluate(trace(),task()['acceptance'])['success']
    assert not evaluate(trace(.09),task()['acceptance'])['success']
    data=trace();data[5]['support_force']=2
    assert not evaluate(data,task()['acceptance'])['success']

def test_reject_bad_timestamps():
    data=trace();data[3]['time']=0
    with pytest.raises(ValueError):evaluate(data,task()['acceptance'])

class Backend:
    interface={'actions':'test'}
    def __init__(self):self.resets=0;self.played=[]
    def reset(self, task, out):self.resets+=1;self.played.append([]);return {}
    def execute(self,a,remaining):
        self.played[-1].append(a['name'])
        if a['name']=='bad':return {'completed':False,'reusable':False,'failure':'contact_lost','samples':[], 'observation':{}}
        return {'completed':True,'reusable':True,'failure':None,'samples':trace() if a['name']=='win' else [],'observation':{}}
    def export(self,out):pass
    def metrics(self):return {'roughness':float(self.resets),'root_omega_rms':0.,'peak_table_force':0.}

class Planner:
    def __init__(self):self.requests=[]
    def plan(self,req):
        self.requests.append(req)
        if req['attempt']==1:return {'analysis':{'evidence':['initial'],'hypothesis':'approach'},'actions':[{'name':'reach'},{'name':'bad'}]}
        if req['attempt']==2:return {'analysis':{'evidence':['contact_lost'],'hypothesis':'repair suffix'},'reuse_prefix':req['prefixes'][0]['id'],'actions':[{'name':'win'}]}
        return {'analysis':{'evidence':['success'],'hypothesis':'smoother'},'actions':[{'name':'bad'}]}

def test_prefix_survives_failure_and_optimization_cannot_erase_success(tmp_path):
    b=Backend();p=Planner();result=run_task(task(),p,b,tmp_path/'run')
    assert result['success'] and result['accepted_attempt']==2
    assert b.played[1]==['reach','win']
    assert b.resets==3
    assert p.requests[2]['phase']=='optimize'
    assert (tmp_path/'run/success/attempt_002/trajectory.json').exists()
    assert (tmp_path/'run/attempts/001/result.json').exists()
    assert json.loads((tmp_path/'run/attempts/002/decision.json').read_text())['reuse_prefix']

def test_no_reference_request_has_no_expert_data(tmp_path):
    p=Planner();run_task(task(),p,Backend(),tmp_path/'run')
    assert p.requests[0]['references']==[]

def test_missing_analysis_prevents_physics(tmp_path):
    class Bad:
        def plan(self,request):return {'actions':[{'name':'win'}]}
    b=Backend()
    with pytest.raises(ValueError):run_task(task(),Bad(),b,tmp_path/'run')
    assert b.resets==0

def test_initial_observation_reaches_planner(tmp_path):
    class Observed(Backend):
        def describe(self,task):return {'object_position':[1,2,3]}
    p=Planner();run_task(task(),p,Observed(),tmp_path/'run')
    assert p.requests[0]['observation']['object_position']==[1,2,3]

def test_resume_cannot_claim_missing_success_artifact(tmp_path):
    out=tmp_path/'run';run_task(task(),Planner(),Backend(),out)
    (out/'success/attempt_002/trajectory.json').unlink()
    with pytest.raises(RuntimeError):run_task(task(),Planner(),Backend(),out,resume=True)

def test_invalid_action_records_failure(tmp_path):
    class Invalid(Backend):
        def execute(self,a,remaining):raise ValueError('out of bounds')
    t=task();t['budget']['attempts']=1
    result=run_task(t,Planner(),Invalid(),tmp_path/'run')
    assert not result['success']
    assert json.loads((tmp_path/'run/attempts/001/result.json').read_text())['failure'].startswith('invalid_action')

def test_stage_feedback_is_bounded_and_received(tmp_path):
    class P:
        def plan(self,request):
            if request.get('feedback_index'):
                assert request['observation']=={'at':'reach'}
                return {'analysis':{'evidence':['reach'],'hypothesis':'continue'},'actions':[{'name':'win'}]}
            return {'analysis':{'evidence':['initial'],'hypothesis':'reach'},'actions':[{'name':'reach','replan_after':True}]}
    class B(Backend):
        def execute(self,a,remaining):
            result=super().execute(a,remaining);result['observation']={'at':a['name']};return result
    t=task();t['budget']['attempts']=1
    assert run_task(t,P(),B(),tmp_path/'run')['success']


def test_feedback_budget_stops_replanning(tmp_path):
 class P:
  def plan(self,req):return {'analysis':{'evidence':['observed'],'hypothesis':'step'},'actions':[{'name':'reach','replan_after':True}]}
 t=task();t['budget'].update(attempts=1,feedback_calls=0)
 r=run_task(t,P(),Backend(),tmp_path/'run')
 assert not r['success']
 assert read(tmp_path/'run/attempts/001/result.json')['failure']=='feedback_budget'


def test_resume_keeps_success_after_bad_optimization(tmp_path):
 t=task();run_task(t,Planner(),Backend(),tmp_path/'run')
 b=Backend();r=run_task(t,Planner(),b,tmp_path/'run',resume=True)
 assert r['accepted_attempt']==2 and b.resets==0


def test_optimization_receives_complete_success_even_if_stage_stopped_early(tmp_path):
 class B(Backend):
  def execute(self,action,remaining):
   r=super().execute(action,remaining);r['completed']=False;r['reusable']=False;return r
 class P:
  def plan(self,req):
   if req['phase']=='optimize':
    assert req['successful_trajectory']['actions']==[{'name':'win'}]
    assert not req['prefixes']
   return {'analysis':{'evidence':['state'],'hypothesis':'test'},'actions':[{'name':'win'}]}
 t=task();r=run_task(t,P(),B(),tmp_path/'run')
 assert r['success']
 assert (tmp_path/'run/success/attempt_001/samples.json').exists()

@pytest.mark.parametrize('action',[None,{},42])
def test_malformed_model_actions_become_failures(tmp_path,action):
 class P:
  def plan(self,req):return {'analysis':{'evidence':['initial'],'hypothesis':'test'},'actions':[action]}
 t=task();t['budget']['attempts']=1
 run_task(t,P(),Backend(),tmp_path/'run')
 assert read(tmp_path/'run/attempts/001/result.json')['failure'].startswith('invalid_action')


def test_native_acceptance_requires_measured_contiguous_evidence():
 criteria={'kind':'native','hold_s':.005,'validator_id':'frozen task checker'}
 rows=[{'time':.005,'dt':.005,'native_success':True,'stable':True}]
 assert evaluate(rows,criteria)['success']
 rows[0]['stable']=False
 assert not evaluate(rows,criteria)['success']
 with pytest.raises(ValueError):evaluate([{'time':.005,'dt':.005,'success':True}],criteria)


def test_disagreement_with_offline_checker_cannot_export_success(tmp_path):
 class B(Backend):
  def verify(self,out):return {'success':False}
 class P:
  def plan(self,req):return {'analysis':{'evidence':['measured'],'hypothesis':'test'},'actions':[{'name':'win'}]}
 with pytest.raises(RuntimeError,match='disagrees'):run_task(task(),P(),B(),tmp_path/'run')
 assert not (tmp_path/'run/success').exists()
