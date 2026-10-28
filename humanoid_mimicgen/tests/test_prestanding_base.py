import pytest
from humanoid_mimicgen.prestanding_base import stage_stable

def test_measured_replan_allows_wrist_offset_only_before_approach():
    assert stage_stable('prestanding', .001, .001, .001, .028, .3, [208,147])
    assert not stage_stable('standing', .001, .001, .001, .028, .3, [208,147])

@pytest.mark.parametrize('drift,speed,yaw,feet',[(.031,0,0,[208,147]),(0,.031,0,[208,147]),(0,0,.151,[208,147]),(0,0,0,[0,147])])
def test_unstable_base_remains_rejected(drift,speed,yaw,feet):
    assert not stage_stable('prestanding',drift,speed,yaw,0,0,feet)
