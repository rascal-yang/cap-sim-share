import copy
import pytest
from cap_sim.core import run_task
from cap_sim.io import read,digest
from test_core import Backend,task


def strict_task():
    t=task();t['budget'].update(attempts=2,optimization_attempts=0)
    t['policy_checks']={'require_reflection':True}
    return t


def reflection(q):
    return {'source_attempt':q['attempt'],'result_sha256':q['result_sha256'],
            'failed_stage':'contact','observed_facts':[{'path':'result.failure','value':'contact_lost'}],
            'hypothesis':'Contact was not retained; exact mechanism unknown.',
            'alternative_causes':['tracking error','insufficient opposed contact'],
            'next_change':{'description':'Test changed closure','controls':['right_fingers']},
            'expected_effect':'Retain contact in the first lift segment',
            'reject_if':'Same early contact loss','hypothesis_outcome':'inconclusive'}


class P:
    def plan(self,q):
        analysis={'evidence':['measured state'],'hypothesis':'test'}
        if q['attempt']>1:
            assert q['previous_reflection']['source_attempt']==1
            analysis['reflection_sha256']=q['previous_reflection_sha256']
            analysis['change']='Test changed closure'
        return {'analysis':analysis,'actions':[{'name':'bad' if q['attempt']==1 else 'win'}]}
    def reflect(self,q):return reflection(q)


def test_reflection_is_bound_to_prior_failure_and_reaches_retry(tmp_path):
    b=Backend();result=run_task(strict_task(),P(),b,tmp_path/'run')
    assert result['success'] and b.resets==2
    r=read(tmp_path/'run/attempts/001/reflection.json')
    assert r['result_sha256']==digest(read(tmp_path/'run/attempts/001/result.json'))
    assert read(tmp_path/'run/attempts/002/request.json')['previous_reflection']==r


@pytest.mark.parametrize('corruption',['hash','invented_fact','missing_fact'])
def test_invalid_reflection_prevents_next_physical_reset(tmp_path,corruption):
    class Bad(P):
        def reflect(self,q):
            r=reflection(q)
            if corruption=='hash':r['result_sha256']='0'*64
            elif corruption=='invented_fact':r['observed_facts'][0]['value']='robot_fall'
            else:r['observed_facts']=[{'path':'result.nonexistent','value':0}]
            return r
    b=Backend()
    with pytest.raises(ValueError,match='reflection'):
        run_task(strict_task(),Bad(),b,tmp_path/'run')
    assert b.resets==1
    assert read(tmp_path/'run/attempts/001/result.json')['failure']=='contact_lost'


def test_resume_without_required_reflection_never_silently_retries(tmp_path):
    t=strict_task()
    run_task(t,P(),Backend(),tmp_path/'run')
    (tmp_path/'run/attempts/001/reflection.json').unlink()
    b=Backend()
    with pytest.raises(ValueError,match='reflection'):
        run_task(t,P(),b,tmp_path/'run',resume=True)
    assert b.resets==0


def test_strict_mode_rejects_hidden_reference_action_before_reset(tmp_path):
    class Mixed(P):
        def plan(self,q):
            return {'analysis':{'evidence':['state'],'hypothesis':'candidate'},
                    'actions':[{'name':'reach','body_keyframes':[[0],[1]]}]}
    t=strict_task();t['policy_checks']['sonic_mode']=1;b=Backend()
    with pytest.raises(ValueError,match='mode1'):
        run_task(t,Mixed(),b,tmp_path/'run')
    assert b.resets==0


def test_retry_must_acknowledge_exact_reflection(tmp_path):
    class Unlinked(P):
        def plan(self,q):
            d=super().plan(q);d['analysis'].pop('reflection_sha256',None);return d
    b=Backend()
    with pytest.raises(ValueError,match='reflection'):
        run_task(strict_task(),Unlinked(),b,tmp_path/'run')
    assert b.resets==1


def test_measured_mode_mismatch_stops_before_next_stage(tmp_path):
    class B(Backend):
        def execute(self,a,remaining):
            r=super().execute(a,remaining);r['controller_diagnostics']={'actual_modes':[0]};return r
    class Q:
        def plan(self,q):return {'analysis':{'evidence':['state'],'hypothesis':'test'},'actions':[{'name':'reach'},{'name':'win'}]}
    t=task();t['policy_checks']={'sonic_mode':1};b=B()
    with pytest.raises(RuntimeError,match='Measured controller mode'):
        run_task(t,Q(),b,tmp_path/'run')
    assert b.played==[['reach']]
