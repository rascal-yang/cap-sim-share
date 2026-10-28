"""Automatic diagnostic completion for a finished transfer study; no manual physical parameters."""
import json
from pathlib import Path
from .agent import save
from .contact_strategy_diagnostics import diagnose
from . import contact_strategy_transfer as transfer

def run(config,provider_config):
 root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);parent=Path(config['parent_study']);old=json.loads((parent/'config.json').read_text());summary=json.loads((parent/'summary.json').read_text());specs={s['task']:s for s in old['tasks']};tasks=[];evidence={}
 for row in summary['results']:
  if row['status']!='needs_tool':continue
  spec=specs[row['task']];source=row['attempts'][-1]['result'] if row['attempts'] else spec['source'];diag=diagnose(spec|{'source':source},root/'diagnostics'/spec['task']);plans=sorted((parent/spec['task']).glob('attempt_*/plan.json'));request=json.loads(plans[-1].read_text())
  evidence[spec['task']]={'requested_diagnostic':{k:request[k] for k in ['action','rationale','hypothesis','strategy']},'source_case_diagnosed':source['experiment_path'],'diagnostic':diag,'limits':'Tool computes saved-state geometry, actual forces and tracking; it does not provide a dynamic pelvis/foot sensitivity model. No new control channel is introduced. Choose supported physical experiment if justified, or state the concrete remaining capability blocker.'};save(root/'diagnostics'/spec['task']/'context.json',evidence[spec['task']]);tasks.append(spec)
 derived=old|{'output_root':str(root),'config_path':str(root/'execution_config.json'),'tasks':tasks,'parent_study':str(parent),'diagnostic_followup':True,'entrypoint':'contact_strategy_followup.run','additional_budget':'at most2 candidates per needs_tool task; no retry of budget-exhausted tasks'};save(root/'execution_config.json',derived)
 original=transfer.api_decide
 def decide(provider,ar,context,episodes,**kwargs):
  task=context['task'];d=evidence[task];compact=dict(d['diagnostic']);samples=compact.pop('phase_samples');compact['phase_samples']=samples[:1]+samples[-6:];context=context|{'requested_tool_results':d|{'diagnostic':compact},'instructions_after_tool':'The requested reusable read-only tool has now run. Do not request the same supplied geometry/force evidence again. A missing dynamics predictor is a capability limit, not hidden evidence. If no supported repair is justified, explain that limit. Physical decisions must still originate in your decide call.'};return original(provider,ar,context,episodes,**kwargs)
 transfer.api_decide=decide
 try:return transfer.run(derived,provider_config)
 finally:transfer.api_decide=original
