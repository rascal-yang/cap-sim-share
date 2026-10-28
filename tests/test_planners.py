import io
import json
from cap_sim.planners import OpenAIPlanner


def test_responses_text_vision_transport(monkeypatch,tmp_path):
 calls=[]
 def send(req,timeout):
  calls.append(json.loads(req.data))
  return io.BytesIO(json.dumps({'id':'mock','status':'completed','usage':{'input_tokens':12,'output_tokens':5},'output':[{'content':[{'type':'output_text','text':'{"ok": true}'}]}]}).encode())
 monkeypatch.setenv('OPENAI_API_KEY','test-secret')
 monkeypatch.setattr('urllib.request.urlopen',send)
 img=tmp_path/'frame.png';img.write_bytes(b'fixture')
 p=OpenAIPlanner('explicit-model',tmp_path/'calls')
 assert p.complete('Return JSON',{'task':'test'},[img])=={'ok':True}
 assert calls[0]['input'][0]['content'][1]['type']=='input_image'
 assert calls[0]['store'] is False
 log=next((tmp_path/'calls').glob('*.json')).read_text()
 assert 'test-secret' not in log
 assert 'input_tokens' in log
