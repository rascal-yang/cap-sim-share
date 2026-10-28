import pytest
from humanoid_mimicgen.augmentation import generalization_numeric as n

def test_feedback_does_not_clip_decision(monkeypatch):
    def reject(*args):raise ValueError('Invalid wrist offset')
    monkeypatch.setattr(n,'ORIGINAL_VALIDATE',reject)
    d={'candidates':[{'parameters':{'wrist_delta_m':[.003,0,.05],'tail_s':1.25}}]}
    with pytest.raises(ValueError,match='z=0.05 permits ONLY x=y=0') as e:n.validate_candidates(d,{})
    assert '62.5 ticks' in str(e.value)
    assert d['candidates'][0]['parameters']=={'wrist_delta_m':[.003,0,.05],'tail_s':1.25}

def test_valid_decision_returned_verbatim(monkeypatch):
    monkeypatch.setattr(n,'ORIGINAL_VALIDATE',lambda d,s:d)
    d={'candidates':[{'parameters':{'wrist_delta_m':[.003,0,.049],'tail_s':1.24}}]}
    assert n.validate_candidates(d,{}) is d
