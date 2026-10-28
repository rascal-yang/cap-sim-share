import json
import pytest
from repro.robust_session import RobustSession, verify_response
from repro.session import StepResult, CorrectionExhausted

def make():
 return RobustSession({'state':{'x':1},'images':{'head':'fixture'}},convert=lambda r,s:r,state_keys=('x',),cameras=('head',),horizon=2,maximum=10)

def good(cid='a'):
 return {'type':'function_call','name':'act','call_id':cid,'arguments':json.dumps({'actions':[1],'execution_note':'fixture'})}

def test_bad_envelope_can_be_repaired_without_motion_and_enters_next_live():
 s=make();called=[]
 s.submit({'name':'wrong'},lambda c:called.append(c))
 assert not called and s.step==0 and not s.terminal
 assert 'invalid_call_envelope' in json.dumps(s.context())
 s.submit(good(),lambda c:StepResult(s.archive[0],True,.1))
 assert s.step==1 and 'invalid_call_envelope' in json.dumps(s.context())

def test_invalid_json_uses_bounded_zero_step_repair():
 s=make()
 for i in range(2):s.submit_json('{broken',lambda c:pytest.fail('motion'))
 with pytest.raises(CorrectionExhausted):s.submit_json('{broken',lambda c:pytest.fail('motion'))
 assert s.step==0 and s.terminal

def test_duplicate_call_is_recoverable_but_terminal_session_cannot_repair():
 s=make();s.submit(good(),lambda c:StepResult(s.archive[0],True,.1))
 s.submit(good(),lambda c:pytest.fail('duplicate executed'))
 assert s.step==1 and 'invalid_call_envelope' in json.dumps(s.context())
 s.terminal=True
 with pytest.raises(RuntimeError):s.submit_json('{broken',lambda c:None)

def test_response_binding_rejects_stale_or_inherited_planning(tmp_path):
 import hashlib
 req=tmp_path/'request.json';resp=tmp_path/'response.json';prov=tmp_path/'provenance.json'
 req.write_text('request1');resp.write_text(json.dumps(good()))
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 row={'request_sha256':h(req),'response_sha256':h(resp),'fork_turns':'none','agent':'isolated_child'}
 prov.write_text(json.dumps(row));assert verify_response(req,resp,prov)==row
 req.write_text('request2')
 with pytest.raises(ValueError):verify_response(req,resp,prov)
 row['request_sha256']=h(req);row['fork_turns']='all';prov.write_text(json.dumps(row))
 with pytest.raises(ValueError):verify_response(req,resp,prov)

def test_response_provenance_rejects_reused_action_child(tmp_path):
 import hashlib
 req=tmp_path/'request';resp=tmp_path/'response';prov=tmp_path/'provenance'
 req.write_text('x');resp.write_text('y');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 prov.write_text(json.dumps({'request_sha256':h(req),'response_sha256':h(resp),'fork_turns':'none','agent':'already_used'}))
 with pytest.raises(ValueError,match='provenance'):
  verify_response(req,resp,prov,seen_agents={'already_used'})
