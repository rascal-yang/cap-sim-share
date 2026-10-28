#!/usr/bin/env python3
"""Standalone search-throughput curves; no simulation or API calls."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--summary',default='summary.json');a=p.parse_args();root=Path(a.root);s=json.loads((root/a.summary).read_text());cfg=json.loads((root/'config.json').read_text());limit=cfg['benchmark']['wall_budget_s'];xs=np.arange(limit+1)
fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
for col,task in enumerate(s['results']):
 for arm,color in [('restart','#d07843'),('rollback','#277ca8')]:
  curves=[[],[]]
  for condition in task['conditions']:
   if arm not in condition['arms']:continue
   rows=[r for r in condition['arms'][arm]['rows'] if not r.get('shared_seed')]
   for metric in (0,1):curves[metric].append([sum(r['elapsed_s']+task['shared_cost_s']<=x and (metric==0 or r['success']) for r in rows) for x in xs])
  for metric in (0,1):
   if curves[metric]:axes[metric,col].step(xs,np.mean(curves[metric],axis=0),where='post',label=arm,color=color)
 for metric in (0,1):axes[metric,col].grid(alpha=.2);axes[metric,col].set_xlim(0,limit);axes[metric,col].set_ylim(bottom=0)
 axes[0,col].set_title(task['profile']);axes[1,col].set_xlabel('Charged wall-clock budget (s)')
axes[0,0].set_ylabel('Mean newly completed attempts');axes[1,0].set_ylabel('Mean new acceptance successes');axes[0,2].legend()
fig.suptitle(root.name+'\nThree candidate-order permutations per task; shared seed excluded from counts',fontsize=10)
fig.tight_layout();fig.savefig(root/'budget_curves.png',dpi=170);plt.close(fig)
