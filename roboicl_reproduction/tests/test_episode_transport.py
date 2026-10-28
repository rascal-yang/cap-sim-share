import hashlib
import json
import pytest
from repro.episode_transport import verify_episode_response, observation_train

def test_chain_and_same_episode_identity(tmp_path):
    request, response, proof = [tmp_path/n for n in ('request','response','proof')]
    request.write_text('{}'); response.write_text('{}')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    row={'request_sha256':digest(request),'response_sha256':digest(response),
         'agent':'/root/episode','fork_turns':'none','episode_id':'trial',
         'turn_index':1,'previous_response_sha256':'previous','history_scope':'current_episode_only'}
    proof.write_text(json.dumps(row))
    assert verify_episode_response(request,response,proof,agent='/root/episode',episode='trial',turn=1,previous='previous')==row
    for key,value in [('agent','/root/old'),('episode_id','old'),('turn_index',0),
                      ('previous_response_sha256',None),('fork_turns','all'),('history_scope','all')]:
        bad=dict(row);bad[key]=value;proof.write_text(json.dumps(bad))
        with pytest.raises(ValueError):verify_episode_response(request,response,proof,agent='/root/episode',episode='trial',turn=1,previous='previous')

def test_train_omits_past_outcomes_and_paths():
    source={'source_accepted':{'success':True},'source_sha256':'secret','source':'/old/result.json',
            'observations':[{'source_frame':0,'time':0,'state':{},'images':['train.png']}]}
    messages=observation_train(source)
    text=json.dumps(messages)
    assert 'source_accepted' not in text and '/old/result.json' not in text and 'secret' not in text
    assert 'train.png' in text and 'source_frame' in text
