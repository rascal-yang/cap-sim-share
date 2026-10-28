import copy
import pytest
from cap_sim.io import read
from cap_sim.task_skills import publish_skill, load_skill


def draft():
    return {'task_id':'apple_pick','revision_reason':'Extract observed contact experience',
            'skill_markdown':'# Apple\n\n```python\ndef policy(scene):\n    observe_then_choose_stance()\n```',
            'supported_controls':['velocity_xy','right_position'],
            'evidence':[{'id':'run_1','status':'observed','scope':'one original-scene run','sha256':'a'*64}],
            'strategies':[{'id':'local','status':'observed','evidence_ids':['run_1'],'required_controls':['right_position']},
                          {'id':'reposition','status':'candidate','evidence_ids':[],'required_controls':['velocity_xy']}],
            'limitations':['Single demonstration; new stance not physically validated']}


def test_versions_preserve_prior_content_and_require_current_parent(tmp_path):
    a=publish_skill(draft(),tmp_path);old=(tmp_path/'apple_pick/v0001/skill.json').read_bytes()
    d=draft();d['revision_reason']='Keep initial experience, add one failure observation'
    b=publish_skill(d,tmp_path,parent=a['revision_sha256'])
    assert b['version']==2 and b['parent_sha256']==a['revision_sha256']
    assert (tmp_path/'apple_pick/v0001/skill.json').read_bytes()==old
    with pytest.raises(ValueError,match='parent'):publish_skill(d,tmp_path,parent=a['revision_sha256'])
    assert not (tmp_path/'apple_pick/v0003').exists()
    assert load_skill(tmp_path/'apple_pick/v0002/skill.json')['experience']['skill_markdown']==d['skill_markdown']


def test_new_strategy_cannot_be_claimed_observed_without_evidence(tmp_path):
    d=draft();d['strategies'][1]['status']='observed'
    with pytest.raises(ValueError,match='evidence'):publish_skill(d,tmp_path)
    assert not (tmp_path/'apple_pick').exists()


def test_task_skill_cannot_invent_controller_channels(tmp_path):
    d=draft();d['strategies'][1]['required_controls']=['bind_object']
    with pytest.raises(ValueError,match='controls'):publish_skill(d,tmp_path)


def test_tampered_skill_is_rejected(tmp_path):
    publish_skill(draft(),tmp_path);p=tmp_path/'apple_pick/v0001/skill.json'
    import json
    a=read(p);a['experience']['skill_markdown']='tampered';p.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='hash'):load_skill(p)


def test_bad_task_id_cannot_escape_registry(tmp_path):
    d=draft();d['task_id']='../escape'
    with pytest.raises(ValueError):publish_skill(d,tmp_path)


def test_openai_planner_receives_frozen_skill_and_strategy_choices(tmp_path,monkeypatch):
    from cap_sim.planners import OpenAIPlanner
    skill=publish_skill(draft(),tmp_path)
    planner=OpenAIPlanner('test-only',tmp_path/'logs');captured={}
    def complete(prompt,payload,images=()):
        captured.update(payload);return {'analysis':{'strategy':{'strategy_id':'reposition'}},'actions':[]}
    monkeypatch.setattr(planner,'complete',complete)
    request={'skills':[skill],'observation':{'objects':{'target':[1,0,0]}}}
    planner.plan(request)
    context=captured['task_skill_context']
    assert context['frozen_versions'][0]['sha256']==skill['revision_sha256']
    assert context['strategy_options'][1]['status']=='candidate'
    assert 'stance' in context['required_decision_fields']
    assert 'task_skill_context' not in request


def test_summary_uses_contact_evidence_and_actual_control_interface(tmp_path):
    from cap_sim.experience import summarize
    from cap_sim.io import write
    source=tmp_path/'ref.json';write(source,{'acceptance':{'success':True}})
    class P:
        def complete(self,prompt,payload,images=()):
            assert payload['contact_prior']['contacts']==['fixture']
            assert payload['interface']['action']=={'velocity_xy':'fixture'}
            assert 'stance' in prompt and 'candidate' in prompt
            return {'skill_markdown':'# fixture','supported_controls':['velocity_xy'],'limitations':['test']}
    summarize(source,tmp_path/'summary',P(),interface={'action':{'velocity_xy':'fixture'}},contact_prior={'contacts':['fixture']})


def test_skill_updates_between_attempts_not_between_feedback_steps(tmp_path):
    from cap_sim.task_skills import EvolvingSkillPlanner
    registry=tmp_path/'registry';v1=publish_skill(draft(),registry)
    class P:
        def __init__(self):self.seen=[]
        def plan(self,q):
            self.seen.append(q['skills'][0]['version'])
            return {'analysis':{'evidence':['state'],'hypothesis':'candidate'},'actions':[{'stage':'hold'}]}
    p=P();wrapped=EvolvingSkillPlanner(p,[registry/'apple_pick'],tmp_path/'audit')
    req={'attempt':1,'skills':[],'observation':{}}
    wrapped.plan(req)
    d=draft();d['revision_reason']='Diagnosed missing stance margin';v2=publish_skill(d,registry,parent=v1['revision_sha256'])
    feedback=copy.deepcopy(req);feedback['feedback_index']=1;wrapped.plan(feedback)
    retry=copy.deepcopy(req);retry['attempt']=2;retry['history']=[{'failure':'reach_error'}]
    result=wrapped.plan(retry)
    assert p.seen==[1,1,2]
    assert result['skill_usage'][0]['version']==2
    assert read(tmp_path/'audit/a001_skills.json')['skills'][0]['version']==1
    assert read(tmp_path/'audit/a002_skills.json')['skills'][0]['version']==2
    assert read(tmp_path/'audit/a002_f00_request.json')['skills'][0]['version']==2
    # Reconstructing the planner cannot silently replace an active attempt's version.
    resumed=EvolvingSkillPlanner(p,[registry/'apple_pick'],tmp_path/'audit')
    resumed.plan(feedback);assert p.seen[-1]==1


def test_existing_run_budget_and_reset_loop_accept_inter_attempt_skill_revision(tmp_path):
    from cap_sim.task_skills import EvolvingSkillPlanner
    from cap_sim.core import run_task
    from test_core import Backend,task
    registry=tmp_path/'registry';v1=publish_skill(draft(),registry)
    class B(Backend):
        def execute(self,a,remaining):
            result=super().execute(a,remaining)
            if a['name']=='bad':
                d=draft();d['revision_reason']='Observed contact_lost in attempt1; revise candidate'
                publish_skill(d,registry,parent=v1['revision_sha256'])
            return result
    class P:
        def plan(self,q):
            version=q['skills'][0]['version']
            return {'analysis':{'evidence':['observed prior failure' if version==2 else 'initial'],
                                'hypothesis':'test revision handoff'},
                    'actions':[{'name':'bad' if version==1 else 'win'}]}
    t=task();t['budget'].update(attempts=2,optimization_attempts=0);b=B()
    planner=EvolvingSkillPlanner(P(),[registry/'apple_pick'],tmp_path/'run/skill_versions')
    result=run_task(t,planner,b,tmp_path/'run')
    assert result['success'] and result['attempts']==2 and b.resets==2
    assert read(tmp_path/'run/attempts/001/decision.json')['skill_usage'][0]['version']==1
    assert read(tmp_path/'run/attempts/002/decision.json')['skill_usage'][0]['version']==2
    assert read(tmp_path/'run/task.json')==t


def test_reflection_uses_completed_attempt_frozen_skill(tmp_path):
    from cap_sim.task_skills import EvolvingSkillPlanner
    registry=tmp_path/'registry';a=publish_skill(draft(),registry)
    class P:
        def plan(self,q):return {'analysis':{'evidence':['state'],'hypothesis':'test'},'actions':[{'stage':'hold'}]}
        def reflect(self,q):return {'seen_version':q['skills'][0]['version']}
    wrapped=EvolvingSkillPlanner(P(),[registry/'apple_pick'],tmp_path/'audit')
    wrapped.plan({'attempt':1})
    d=draft();d['revision_reason']='next attempt candidate';publish_skill(d,registry,parent=a['revision_sha256'])
    assert wrapped.reflect({'attempt':1})['seen_version']==1


def test_reflection_keeps_static_skills_used_by_the_attempt(tmp_path):
    from cap_sim.task_skills import EvolvingSkillPlanner
    registry=tmp_path/'registry';publish_skill(draft(),registry)
    seed={'artifact_kind':'seed_experience','skill_markdown':'Keep measured contact gates.'}
    class P:
        def plan(self,q):return {'actions':[]}
        def reflect(self,q):return {'skills':q['skills']}
    wrapped=EvolvingSkillPlanner(P(),[registry/'apple_pick'],tmp_path/'audit')
    wrapped.plan({'attempt':1,'skills':[seed]})
    assert seed in wrapped.reflect({'attempt':1})['skills']
