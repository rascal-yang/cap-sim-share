"""Cruise to a stance neighborhood; stopping does not certify reachability."""
import numpy as np

def navigation_command(current_xyyaw, goal_xyyaw, speed=.5, stop_radius=.10):
    current=np.asarray(current_xyyaw,float); goal=np.asarray(goal_xyyaw,float)
    if current.shape!=(3,) or goal.shape!=(3,) or not np.isfinite([*current,*goal,speed,stop_radius]).all() or speed<=0 or stop_radius<=0:
        raise ValueError("Invalid navigation input")
    delta=goal[:2]-current[:2]; distance=float(np.linalg.norm(delta))
    stop=distance<=stop_radius
    return (np.zeros(2) if stop else delta*(speed/distance)),float(goal[2]),stop
