"""Explicit live full-body planner reference for SONIC encoder0 navigation.

This consumes newly planned references, never original source motion tokens.
After navigation, manipulation can use encoder1 via an explicit handoff.
"""
import numpy as np


def body_future_from_planner(sonic,state,velocity,height,yaw):
    if sonic.ref is None or sonic.tick%25==0:sonic.plan_legs(state,velocity,height,yaw)
    frames=(sonic.tick-sonic.ref_tick)*.6+np.arange(10)*3
    return sonic.sample(frames),sonic.sample(frames+.3)
