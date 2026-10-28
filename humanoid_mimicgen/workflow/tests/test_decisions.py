import json
import pytest
from humanoid_mimicgen.workflow.decisions import ResponsesDecision, validate_proposal, replay_decision
from humanoid_mimicgen.workflow.contracts import WorkflowError


def proposal():
    return {'action':'propose_stances','stances':[{'id':'face','xyyaw':[1.,2.,0.]}],
            'evidence_ids':['scene'],'reason':'reach target'}


def test_invalid_evidence_and_nonfinite_rejected():
    p=proposal();p['evidence_ids']=['invented']
    with pytest.raises(WorkflowError):validate_proposal(p,{'scene':{}})
    p=proposal();p['stances'][0]['xyyaw'][0]=float('nan')
    with pytest.raises(WorkflowError):validate_proposal(p,{'scene':{}})


def test_missing_key_is_audited(tmp_path,monkeypatch):
    monkeypatch.delenv('WORKFLOW_TEST_API_KEY',raising=False)
    provider=ResponsesDecision('gpt-6-astra','https://api.openai.com/v1','WORKFLOW_TEST_API_KEY',1.)
    with pytest.raises(WorkflowError,match='api_unavailable'):provider.propose({'scene':{}},tmp_path/'call')
    a=json.loads((tmp_path/'call/provenance.json').read_text());assert a['status']=='failed'
    assert 'Authorization' not in (tmp_path/'call/request.json').read_text()


def test_saved_decision_replay_validates_evidence_without_network(tmp_path):
    (tmp_path/'response.json').write_text(json.dumps(proposal()))
    from humanoid_mimicgen.workflow.contracts import digest
    (tmp_path/'provenance.json').write_text(json.dumps({'status':'completed','evidence_sha256':digest({'scene':{}})}))
    p=replay_decision(tmp_path/'response.json',{'scene':{}},provider_identity='test_fixture')
    assert p['action']=='propose_stances'
    with pytest.raises(WorkflowError):replay_decision(tmp_path/'response.json',{},provider_identity='test_fixture')


@pytest.mark.parametrize('status,model',[('incomplete','gpt-6-astra'),('completed','wrong-model')])
def test_remote_model_mismatch_and_incomplete_are_not_decisions(tmp_path,monkeypatch,status,model):
    import io
    import humanoid_mimicgen.workflow.decisions as module
    monkeypatch.setenv('WORKFLOW_TEST_API_KEY','test-secret')
    raw={'status':status,'model':model,'output':[{'content':[{'type':'output_text','text':json.dumps(proposal())}]}]}
    def transport(request,timeout):return io.BytesIO(json.dumps(raw).encode())
    monkeypatch.setattr(module.urllib.request,'urlopen',transport)
    provider=ResponsesDecision('gpt-6-astra','https://api.openai.com/v1','WORKFLOW_TEST_API_KEY',1.)
    with pytest.raises(WorkflowError,match='invalid_api_proposal'):provider.propose({'scene':{}},tmp_path/'call')
    assert not (tmp_path/'call/response.json').exists()
    assert all('test-secret' not in p.read_text() for p in (tmp_path/'call').iterdir())


def test_valid_api_response_is_saved_and_replay_is_offline(tmp_path,monkeypatch):
    import io
    import humanoid_mimicgen.workflow.decisions as module
    monkeypatch.setenv('WORKFLOW_TEST_API_KEY','test-secret')
    raw={'status':'completed','model':'gpt-6-astra','id':'test','usage':{'total_tokens':10},'output':[{'content':[{'type':'output_text','text':json.dumps(proposal())}]}]}
    calls=[]
    def transport(request,timeout):
        calls.append(json.loads(request.data));return io.BytesIO(json.dumps(raw).encode())
    monkeypatch.setattr(module.urllib.request,'urlopen',transport)
    provider=ResponsesDecision('gpt-6-astra','https://api.openai.com/v1','WORKFLOW_TEST_API_KEY',1.)
    assert provider.propose({'scene':{}},tmp_path/'call')==proposal()
    assert calls[0]['text']['format']['strict'] is True
    def forbidden(*args,**kwargs):raise AssertionError('Replay must not use network')
    monkeypatch.setattr(module.urllib.request,'urlopen',forbidden)
    assert replay_decision(tmp_path/'call/response.json',{'scene':{}},provider_identity='test_transport')==proposal()
    assert json.loads((tmp_path/'call/provenance.json').read_text())['status']=='completed'
