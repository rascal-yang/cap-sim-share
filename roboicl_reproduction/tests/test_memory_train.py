import json
from copy import deepcopy
import pytest
from repro.offline import create_policy, observation, response
from roboicl.policy.trajectory_memory import LiveTrajectoryMemory
from roboicl.policy.train_tool_reference import tool_reference_messages
from roboicl.policy.astra_policy import image_url


def test_fixed_anchors_latest_and_explicit_gap():
    m = LiveTrajectoryMemory(3, retain_latest=True); m.configure(12)
    obs = lambda s: {'role': 'user', 'content': [{'type':'input_text', 'text':f'OBS {s}'}]}
    for i in range(6):
        m.capture(i, i+1, [{'role':'assistant', 'content':f'ACTION {i}'}], summary={'executed_steps':1})
    text = json.dumps(m.render(obs(0), 6, obs))
    assert 'ACTION 0' in text and 'ACTION 3' in text and 'ACTION 5' in text
    assert 'ACTION 2' not in text and 'TRAJECTORY_GAP' in text
    assert m.stats['captured_chunks'] == 6
    assert not m.capture(6, 6, [])


def test_original_train_tool_history_contains_commands_and_terminal(tmp_path):
    p = create_policy(tmp_path, [])
    state = observation(0)['state']; result = observation(2)['state']
    rows = json.loads(response('x')['output'][0]['arguments'])['actions']
    image = image_url(observation(0)['vision']['cam_head']['color'])
    example = {'frame':0, 'result_frame':2, 'action_tensor':rows,
               'action_step_interval_inclusive':[0,1], 'state':state, 'result_state':result,
               'head_image':image, 'prompt_images':{'cam_head':image},
               'result_prompt_images':{'cam_head':image}}
    demo = {'sha256':'0'*64, 'examples':[example], 'total_frames':3,
            'terminal_observation':{'frame':2,'state':result,'prompt_images':{'cam_head':image}}}
    messages = tool_reference_messages(demo, [], p._convert_tensor, include_result_observations=False)
    call = next(m for m in messages if m.get('type') == 'function_call')
    assert json.loads(call['arguments'])['actions'] == rows
    receipt = json.loads(next(m for m in messages if m.get('type') == 'function_call_output')['output'])
    assert receipt['executed_steps'] == 2 and receipt['success'] is None
    assert 'end_of_example' in json.dumps(messages)
    assert p.pending is None and p.step is None and not p.ledger
    bad = deepcopy(demo); bad['examples'][0]['result_frame'] = 3
    with pytest.raises(ValueError, match='interval'):
        tool_reference_messages(bad, [], p._convert_tensor)


def test_original_endpoint_dependencies_are_available():
    pytest.importorskip('h5py', reason='Original TRAIN raw-source endpoints require h5py; not installed in existing HUMANO env')
    pytest.importorskip('XPolicyLab', reason='Original source RGB decoder requires uninitialized upstream submodule')
