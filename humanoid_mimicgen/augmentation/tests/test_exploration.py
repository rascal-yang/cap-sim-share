import json
import numpy as np
import pytest
from humanoid_mimicgen.augmentation.exploration import query_case, explore_decision


def test_query_bounds_and_alignment(tmp_path):
    p=tmp_path/'execution';p.mkdir()
    np.savez(p/'controller.npz',records=np.arange(24).reshape(4,6))
    q={'case':'replay','artifact':'controller','start':1,'stop':3,'stride':1}
    r=query_case({'replay':str(tmp_path)},q)
    assert r['arrays']['records']==[[6,7,8,9,10,11],[12,13,14,15,16,17]]
    assert r['physics_steps']==0 and r['source_sha256']
    with pytest.raises(ValueError):query_case({'replay':str(tmp_path)},q|{'case':'../../secret'})
    with pytest.raises(ValueError):query_case({'replay':str(tmp_path)},q|{'stop':10000})


def test_needs_tool_feedback_and_continue(tmp_path):
    calls=[]
    responses=iter([{'action':'proceed','rationale':'plan','query':None}, {'action':'needs_tool'},
                    {'action':'inspect','rationale':'read','query':{'case':'replay','artifact':'metrics','start':0,'stop':2,'stride':1}},
                    {'action':'proceed','rationale':'now plan','query':None},{'action':'run'}])
    p=tmp_path/'replay/execution';p.mkdir(parents=True);(p/'metrics.json').write_text('[{"frame":0}]')
    def decide(provider,root,context,episodes,**kw):
        calls.append(context);return next(responses),root/'decision.json'
    result,_=explore_decision(None,tmp_path/'api_root',{},[0],decide=decide,decision_tools=[],validator=lambda d,e:d,skill_path='unused',cases={'replay':str(p.parent)},budget={'remaining':8})
    assert result['action']=='run'
    assert any(c.get('tool_results') for c in calls)
    assert len(list((tmp_path/'api_root').glob('exploration_*/tool_result.json')))==1


def test_exhaustion_is_explicit(tmp_path):
    with pytest.raises(TimeoutError,match='exploration'):
        explore_decision(None,tmp_path,{},[],decide=None,decision_tools=[],validator=None,skill_path='',cases={},budget={'remaining':0})


def test_missing_data_returns_to_api(tmp_path):
    responses=iter([{'action':'inspect','rationale':'query missing force','query':{'case':'replay','artifact':'contact_control','start':0,'stop':1,'stride':1}},
                    {'action':'proceed','rationale':'use available evidence','query':None},{'action':'run'}])
    contexts=[]
    def decide(provider,root,context,episodes,**kw):
        contexts.append(context);return next(responses),root/'decision.json'
    d,_=explore_decision(None,tmp_path/'run',{},[],decide=decide,decision_tools=[],validator=None,skill_path='',cases={'replay':str(tmp_path)},budget={'remaining':3})
    assert d['action']=='run'
    assert contexts[1]['tool_results'][0]['result']['missing_data_is_unknown']


def test_stop_preserves_api_reason(tmp_path):
    def decide(*args,**kw):return {'action':'stop','rationale':'required sensor not recorded','query':None},tmp_path/'api.json'
    with pytest.raises(RuntimeError,match='required sensor not recorded'):
        explore_decision(None,tmp_path,{},[],decide=decide,decision_tools=[],validator=None,skill_path='',cases={'replay':'unused'},budget={'remaining':1})
    assert json.loads((tmp_path/'exploration_blocked.json').read_text())['rationale']=='required sensor not recorded'


def test_persisted_api_decisions_keep_original_provenance(tmp_path):
    from humanoid_mimicgen.augmentation.quality_suite import api_decide,verify_origin
    from humanoid_mimicgen.augmentation.contact_review import SKILL
    responses=iter([{'action':'inspect','rationale':'read recorded metrics','query':{'case':'replay','artifact':'metrics','start':0,'stop':1,'stride':1}},
                    {'action':'proceed','rationale':'enough evidence','query':None},{'action':'run'}])
    class Provider:
        config={'model':'test-model'}
        def respond(self,items,tools,instructions):
            return {'status':'completed','model':'test-model','output':[{'type':'function_call','name':'decide','call_id':'test-call','arguments':json.dumps(next(responses))}],'usage':{'total_tokens':1}}
    p=tmp_path/'replay/execution';p.mkdir(parents=True);(p/'metrics.json').write_text('[{"time_s":0.02}]')
    root=tmp_path/'run'
    d,path=explore_decision(Provider(),root,{'stage':'contact_plan'},[0],decide=api_decide,decision_tools=[],validator=lambda d,e:d,skill_path=SKILL,cases={'replay':str(p.parent)},budget={'remaining':4})
    assert verify_origin(path)['decision']==d=={'action':'run'}
    assert len(list(root.glob('**/api/*/decision.json')))==3
