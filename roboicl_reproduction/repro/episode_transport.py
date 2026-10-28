"""One isolated planner identity per episode; chained raw-response provenance."""
import hashlib
import json
from native_run import train_messages


def verify_episode_response(request, response, provenance, *, agent, episode, turn, previous):
    row = json.loads(provenance.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    expected = {'request_sha256':sha(request), 'response_sha256':sha(response),
                'agent':agent, 'fork_turns':'none', 'episode_id':episode,
                'turn_index':turn, 'previous_response_sha256':previous,
                'history_scope':'current_episode_only'}
    if not isinstance(agent,str) or not agent or any(row.get(k)!=v for k,v in expected.items()):
        raise ValueError('Episode planner provenance mismatch')
    return row


def observation_train(demo):
    # Whitelist metadata; historical validation, paths and reflections are not
    # model inputs. The measured demonstration itself remains available.
    return train_messages({'kind':'sparse measured TRAIN observations',
        'interface_note':'Source demonstration used historical prefix/mode0; these are observations, not mode1 action labels.',
        'observations':demo['observations']})
