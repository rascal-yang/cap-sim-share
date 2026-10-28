"""Opt-in lower-body standing-idle adapter, independent of encoder mode."""
import hashlib
import numpy as np

class StandingIdlePlanner:
    def __init__(self, session, enabled=True, clock=lambda: None):
        self.session, self.enabled, self.clock = session, enabled, clock
        self.records = []

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, output_names, feed, *args, **kwargs):
        # GEAR ROS2 standing-stop tuple; enum 0 is NOT SONIC encoder mode 0.
        changed = bool(self.enabled and feed['mode'].item() == 4
                       and feed['target_vel'].item() == -1
                       and feed['height'].item() >= np.float32(.72))
        actual = dict(feed)
        if changed:
            actual['mode'] = np.zeros_like(feed['mode'])
            actual['movement_direction'] = np.zeros_like(feed['movement_direction'])
            actual['height'] = np.full_like(feed['height'], -1)
        keys = ('mode','target_vel','movement_direction','height','random_seed','facing_direction')
        record = {'time':self.clock(), 'transformed':changed,
                  'before':{k:feed[k].tolist() for k in keys},
                  'after':{k:actual[k].tolist() for k in keys},
                  'context_sha256':hashlib.sha256(feed['context_mujoco_qpos'].tobytes()).hexdigest()}
        result = self.session.run(output_names, actual, *args, **kwargs)
        self.records.append(record)
        return result
