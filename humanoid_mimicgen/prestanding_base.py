"""Physical settling before replanning from measured wrists."""
from .standing_handoff import stable_sample

def stage_stable(stage, drift, speed, yaw_speed, wrist_error, wrist_angle, feet):
    if stage == 'prestanding':
        wrist_error = wrist_angle = 0.
    return stable_sample(drift, speed, yaw_speed, wrist_error, wrist_angle, feet)
