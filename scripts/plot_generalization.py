#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from generalization_records import roots as study_roots, conditions as study_conditions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);rows=study_conditions(root)
fig,axes=plt.subplots(1,4,figsize=(15,4),sharex=True,sharey=True)
for ax,label in zip(axes,['plate','chair','button','apple']):
    rs=[r for r in rows if r['task_label']==label]
    for r in rs:
        passed=[r.get('arms',{}).get(k,{}).get('verified_success',False) for k in ('restart','rollback')]
        color='#29965e' if all(passed) else '#3877ce' if passed[0] else '#ed9a26' if passed[1] else '#d14747'
        if r['status']=='error':color='#888888'
        sp=json.loads((root/(r['condition_id']+'_spec.json')).read_text());x,y=[v*100 for v in sp['offset']]
        ax.scatter([x],[y],s=200,c=color,edgecolor='white',linewidth=1,zorder=3);ax.annotate(r['condition_id'][-2:],(x,y),ha='center',va='center',fontsize=9,color='white')
    ax.set_title(label+f' ({len(rs)}/10 complete)');ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xticks([-2,0,2,4]);ax.set_yticks([-2,0,2]);ax.set_xlim(-3,5);ax.set_ylim(-3,3);ax.set_xlabel('Object offset X (cm)')
axes[0].set_ylabel('Object offset Y (cm)')
fig.suptitle(root.name+'\nVerified success within120s (full API planning + seed included)',fontsize=12)
labels=[('Both pass','#29965e'),('Restart only','#3877ce'),('Rollback only','#ed9a26'),('Both fail','#d14747'),('Infrastructure error','#888888')]
fig.legend([Line2D([],[],marker='o',linestyle='',color=c) for _,c in labels],[n for n,_ in labels],loc='lower center',ncol=5)
fig.tight_layout(rect=[0,.08,1,.87]);fig.savefig(root/'position_results.png',dpi=180)
