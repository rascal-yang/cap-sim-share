"""Bounded protocol repair and response provenance for isolated model transport."""
import hashlib
import json
from .session import Session

class RobustSession(Session):
    def _invalid_envelope(self,detail):
        if self.terminal:raise RuntimeError('terminal session; create a new episode explicitly')
        self.tail.append({'role':'user','content':[{'type':'input_text','text':json.dumps({
            'source':'LIVE','error':'invalid_call_envelope','detail':detail,'executed_steps':0,
            'observation_step':self.step,'recovery':'Return one act function_call with fresh call_id and JSON-string arguments; SAME observation, no motion executed.'})}]})
        self._correction()

    def submit_json(self,raw,execute):
        if self.terminal:raise RuntimeError('terminal session; create a new episode explicitly')
        try:call=json.loads(raw)
        except (ValueError,TypeError) as error:
            return self._invalid_envelope(str(error))
        return self.submit(call,execute)

    def submit(self,call,execute):
        if self.terminal:raise RuntimeError('terminal session; create a new episode explicitly')
        if (not isinstance(call,dict) or call.get('type')!='function_call' or call.get('name')!='act'
            or not isinstance(call.get('call_id'),str) or not call['call_id'].strip()
            or call['call_id'] in self.seen):
            return self._invalid_envelope('Malformed or duplicate act call')
        return super().submit(call,execute)

def verify_response(request,response,provenance,seen_agents=()):
    """Fail closed on stale response or inherited context; not a filesystem sandbox."""
    row=json.loads(provenance.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if (row.get('request_sha256')!=sha(request) or row.get('response_sha256')!=sha(response)
        or row.get('agent') in seen_agents or row.get('fork_turns')!='none' or not isinstance(row.get('agent'),str) or not row['agent']):
        raise ValueError('Planner response provenance mismatch')
    return row
