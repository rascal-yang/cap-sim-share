import json
import hashlib
import pytest
from repro.train import build_train


def test_train_rechecks_commands_not_measured_endpoint_deltas(tmp_path):
    source = {'schema':'humano.command_episode.v1', 'profile':'test-profile', 'observation_times_s':[0,.1,.2,.3],
      'observations':[{'state':{'x':i},'images':{'head':'synthetic'}} for i in range(4)],
      'transitions':[{'rows':[1], 'commands':[1], 'elapsed_s':.1, 'control_modes':[1], 'event':None} for _ in range(3)]}
    path=tmp_path/'source.json'; path.write_text(json.dumps(source))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    convert=lambda rows, state: rows
    messages=build_train(path,sha,'test-profile',convert,starts=[0,2],horizon=1, state_keys=('x',), cameras=('head',))
    assert sum(m.get('type')=='function_call' for m in messages)==2
    assert 'TRAJECTORY_GAP' in json.dumps(messages)
    assert 'end_of_example' in json.dumps(messages)
    source['transitions'][0]['commands']=[999];path.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='hash'): build_train(path,sha,'test-profile',convert,starts=[0],horizon=1, state_keys=('x',), cameras=('head',))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='command'): build_train(path,sha,'test-profile',convert,starts=[0],horizon=1, state_keys=('x',), cameras=('head',))


def test_train_refuses_profile_and_mode_mismatch(tmp_path):
    path=tmp_path/'source.json'
    source={'schema':'humano.command_episode.v1','profile':'wrong','observations':[{},{}],
            'transitions':[{'rows':[1],'commands':[1],'elapsed_s':.1,'control_modes':[0],'event':None}]}
    path.write_text(json.dumps(source));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='profile'): build_train(path,sha,'correct',lambda r,s:r,starts=[0],horizon=1, state_keys=('x',), cameras=('head',))


def test_train_rejects_privileged_state(tmp_path):
    source={'schema':'humano.command_episode.v1','profile':'p', 'observation_times_s':[0,.1],
            'observations':[{'state':{'x':0,'object_pose':'SECRET'},'images':{'head':'synthetic'}}]*2,
            'transitions':[{'rows':[1],'commands':[1],'elapsed_s':.1,'control_modes':[1],'event':None}]}
    path=tmp_path/'s.json';path.write_text(json.dumps(source))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='whitelist'):
        build_train(path,sha,'p',lambda r,s:r,starts=[0],horizon=1,state_keys=('x',),cameras=('head',))


@pytest.mark.parametrize('times,mode', [([0,100],[1]), ([0,-.1],[1]), ([0,.1],[True])])
def test_train_rejects_time_and_boolean_mode(tmp_path,times,mode):
    source={'schema':'humano.command_episode.v1','profile':'p', 'observation_times_s':times,
            'observations':[{'state':{'x':0},'images':{'head':'synthetic'}}]*2,
            'transitions':[{'rows':[1],'commands':[1],'elapsed_s':.1,'control_modes':mode,'event':None}]}
    path=tmp_path/'s.json';path.write_text(json.dumps(source))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        build_train(path,sha,'p',lambda r,s:r,starts=[0],horizon=1,state_keys=('x',),cameras=('head',))
