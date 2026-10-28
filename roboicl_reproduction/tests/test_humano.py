import copy
import math
import unittest

from repro.humano import encode_actions


class HumanoContractTests(unittest.TestCase):
    def setUp(self):
        self.state = {'root_pose': [0, 0, .75, 1, 0, 0, 0],
                      'torso_quaternion_wxyz': [1, 0, 0, 0]}
        for side in ('left', 'right'):
            self.state[side + '_position'] = [0, 0, 1]
            self.state[side + '_quaternion_wxyz'] = [1, 0, 0, 0]
        self.limits = {side: [[-1, 1]] * 7 for side in ('left', 'right')}
        self.row = {'stage': 'approach', 'left_delta': [.02, 0, 0, 0, 0, .1],
                    'right_delta': [0] * 6, 'left_fingers': [.2] * 7,
                    'right_fingers': [.3] * 7}

    def test_accumulates_from_observation_without_mutation(self):
        original = copy.deepcopy((self.row, self.state, self.limits))
        actions = encode_actions([self.row, self.row], self.state, self.limits)
        self.assertEqual(actions[1]['left_position'], [.04, 0, 1])
        self.assertAlmostEqual(actions[1]['left_quaternion_wxyz'][0], math.cos(.1))
        self.assertAlmostEqual(actions[1]['left_quaternion_wxyz'][3], math.sin(.1))
        self.assertEqual(actions[0]['duration_s'], .1)
        self.assertEqual(actions[0]['stage'], 'approach')
        self.assertEqual(actions[0]['right_fingers'], [.3] * 7)
        self.assertEqual((self.row, self.state, self.limits), original)

    def test_world_rotation_multiplies_on_left_and_normalizes(self):
        self.state['left_quaternion_wxyz'] = [math.sqrt(.5)*1.001, math.sqrt(.5)*1.001, 0, 0]
        q = encode_actions([self.row], self.state, self.limits)[0]['left_quaternion_wxyz']
        self.assertAlmostEqual(q[2], math.sqrt(.5)*math.sin(.05))
        self.assertAlmostEqual(sum(x*x for x in q), 1)

    def test_rejects_invalid_rows_including_final_row(self):
        bad = [dict(self.row, body_keyframes=[]), dict(self.row, token=[]),
               dict(self.row, replayprefix=[]), dict(self.row, duration_s=.04),
               dict(self.row, left_gripper=.5), dict(self.row, stage='bogus'),
               dict(self.row, left_delta=[.025, .025, 0, 0, 0, 0]),
               dict(self.row, right_delta=[0, 0, 0, .12, .12, 0]),
               dict(self.row, velocity_xy=[.4, .4]),
               dict(self.row, facing_yaw=4), dict(self.row, pelvis_height_m=.9),
               dict(self.row, left_fingers=[1.01]*7),
               dict(self.row, right_fingers=.5), dict(self.row, left_delta=[True]*6),
               dict(self.row, facing_yaw=True), dict(self.row, facing_yaw=float('nan')),
               dict(self.row, torso_quaternion_wxyz=[0]*4),
               dict(self.row, left_delta=['0']*6)]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                encode_actions([self.row, row], self.state, self.limits)

    def test_requires_limits_observation_and_bounded_chunk(self):
        for rows, state, limits, horizon in [([], self.state, self.limits, 32),
                ([self.row]*2, self.state, self.limits, 1),
                ([self.row], {}, self.limits, 32),
                ([self.row], self.state, {}, 32),
                ([self.row], self.state, self.limits, True)]:
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                encode_actions(rows, state, limits, horizon=horizon)

    def test_checks_accumulated_workspace(self):
        self.row['left_delta'] = [.03, 0, 0, 0, 0, 0]
        self.state['left_position'] = [1.4, 0, 1]
        with self.assertRaises(ValueError):
            encode_actions([self.row]*16, self.state, self.limits)

    def test_optional_channels_preserved(self):
        self.row.update(velocity_xy=[.3,.4], facing_yaw=math.pi,
                        pelvis_height_m=.4, torso_quaternion_wxyz=[1.001,0,0,0])
        result = encode_actions([self.row], self.state, self.limits)[0]
        self.assertEqual(result['velocity_xy'], [.3,.4])
        self.assertEqual(result['facing_yaw'], math.pi)
        self.assertEqual(result['pelvis_height_m'], .4)
        self.assertEqual(result['torso_quaternion_wxyz'], [1,0,0,0])

    def test_invalid_limits_and_observation_fail_closed(self):
        for pair in ([0, 0], [1, -1], [False, 1], [0, float('inf')]):
            limits = copy.deepcopy(self.limits)
            limits['left'][0] = pair
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                encode_actions([self.row], self.state, limits)
        for key, value in [('left_position', [False, 0, 1]),
                           ('right_quaternion_wxyz', [2, 0, 0, 0]),
                           ('root_pose', [0, 0, float('nan'), 1, 0, 0, 0])]:
            state = dict(self.state, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                encode_actions([self.row], state, self.limits)


if __name__ == '__main__':
    unittest.main()
