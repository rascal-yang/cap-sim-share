"""Export a diagnostic plot; references are never presented as physics results."""

from pathlib import Path

import numpy as np


def plot_summary(demo, annotation, batch, output, experiment_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .dataset import load_demo

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    colors = ['#587792', '#c68824', '#358b73', '#9861a5']
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), constrained_layout=True)
    t = demo['time_s']
    ax = axes[0, 0]
    ax.plot(t, demo['hand_force_N'], color='#237c9b', lw=1.4, label='Recorded hand-object force')
    ax.plot(t, demo['support_force_N'], color='#888888', lw=.8, alpha=.75, label='Recorded support force')
    for i, s in enumerate(annotation['stages']):
        start, end = s['start_frame'], s['end_frame']
        ax.axvspan(t[start], t[end-1], color=colors[i % len(colors)], alpha=.12)
        label = str(i+1) if t[end-1] - t[start] < 2 else f"{i+1}. {s['kind']}"
        ax.text((t[start]+t[end-1])/2, .97, label,
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9)
    ax.set(xlabel='Source time (s)', ylabel='Force (N)', title='GPT-6 phases aligned to measured evidence', ylim=(0, 32))
    ax.legend(fontsize=8, loc='upper left')
    ax = axes[0, 1]
    ax.plot(t, demo['object_pose'][:, 2], label='Measured apple z', color='#358b73')
    ax.plot(t, demo['right_pose'][:, 2], label='Measured right wrist z', color='#587792')
    ax.axvline(t[annotation['stages'][1]['start_frame']], c='#c68824', ls='--', lw=1, label='Contact starts')
    ax.set(xlabel='Source time (s)', ylabel='World height (m)', title='Source demonstration: contact then lift')
    ax.legend(fontsize=8)
    ax_xy, ax_z = axes[1]
    for entry in batch['results']:
        name = entry['case']['id']
        generated = load_demo(entry['trajectory'])
        if entry['case'].get('table_delta', 0) == 0:
            ax_xy.plot(generated['right_pose'][:, 0], generated['right_pose'][:, 1], lw=1, label=name)
        else:
            ax_z.plot(t, generated['right_pose'][:, 2], lw=1, label=name)
    ax_xy.plot(demo['right_pose'][:, 0], demo['right_pose'][:, 1], c='black', lw=1.8, label='source')
    ax_z.plot(t, demo['right_pose'][:, 2], c='black', lw=1.8, label='source')
    ax_xy.set(xlabel='World x (m)', ylabel='World y (m)', title='Six lateral variants: generated wrist references')
    ax_z.set(xlabel='Source time (s)', ylabel='World wrist height (m)', title='Four table-height variants: generated references')
    for ax in (ax_xy, ax_z):
        ax.legend(fontsize=7, loc='best', ncols=2)
    for ax in axes.flat:
        ax.grid(alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(experiment_id + '\nGPT-6 contact reuse | offline geometry; not simulated success', fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
