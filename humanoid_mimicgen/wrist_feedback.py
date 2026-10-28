"""Bounded world-frame integral correction; desired trajectory remains unchanged."""
import numpy as np

def position_bias(previous, desired, measured, dt):
    bias=np.asarray(previous)+.5*dt*(np.asarray(desired)-np.asarray(measured))
    norm=np.linalg.norm(bias)
    return bias*min(1.,.02/max(norm,1e-12))
