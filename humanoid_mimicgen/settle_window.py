"""Stationarity uses recent measured motion, excluding earlier braking travel."""
import numpy as np

def recent_root_drift(observations, stage_start_s, window_s=.3):
    if not observations:return float('inf')
    end=observations[-1]['time_s']; start=end-window_s
    if start < stage_start_s-1e-8:return float('inf')
    rows=[o for o in observations if o['time_s']>=start-1e-8]
    if not rows or rows[0]['time_s']>start+.006:return float('inf')
    xy=np.asarray([o['root_pose'][:2] for o in rows])
    return float(np.max(np.linalg.norm(xy-xy[0],axis=1)))
