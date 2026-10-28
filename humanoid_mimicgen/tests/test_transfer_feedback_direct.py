from humanoid_mimicgen.transfer_feedback_direct import direct_grasp_due

def test_direct_grasp_only_after_approach_and_not_during_prefix():
    assert not direct_grasp_due('prefix',1800)
    assert not direct_grasp_due('approach',1799)
    assert direct_grasp_due('approach',1800)
    assert not direct_grasp_due('grasp',1801)
