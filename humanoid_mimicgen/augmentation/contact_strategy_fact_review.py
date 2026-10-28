"""API factual consistency check for a stale-rationale regression; no physics."""
import json,time
from pathlib import Path
from .agent import save
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .quality_suite import api_decide
from .contact_review import tools
from ..workflow.contracts import now
PROPS={'verdict':{'type':'string','enum':['consistent','inconsistent']},'contradictions':{'type':'array','items':{'type':'string'}},'corrected_rationale':{'type':'string'},'next_action':{'type':'string','enum':['run','needs_tool','reject']}}
def run(config,provider_config):
 root=Path(config['output_root']);start=now();tick=time.monotonic();provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+600)
 def validate(d,eps):
  if set(d)!=set(PROPS) or d['verdict'] not in ['consistent','inconsistent'] or d['next_action'] not in ['run','needs_tool','reject']:raise ValueError('Invalid review')
  return d
 error=None;decision=None
 try:decision,path=api_decide(provider,root,json.loads(Path(config['context']).read_text()),[],tools=tools(PROPS),validator=validate,skill_path=Path(config['skill_path']))
 except Exception as e:error=str(e)
 calls=[json.loads(p.read_text()) for p in root.glob('api/*/attempt_*.json')];s={'experiment_id':root.name,'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'decision':decision,'error':error,'physics_steps':0,'api_requests':len(calls),'known_tokens':sum((x.get('usage') or {}).get('total_tokens',0) for x in calls),'unknown_usage_calls':sum(x.get('usage') is None for x in calls)};save(root/'summary.json',s);return s
