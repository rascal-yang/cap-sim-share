from humanoid_mimicgen.reachability_direct import approach_ready_for_contact

def test_contact_starts_after_reference_finishes_at_policy_boundary():
    assert not approach_ready_for_contact(2.98,3.,600)
    assert not approach_ready_for_contact(3.,3.,601)
    assert approach_ready_for_contact(3.,3.,600)
    assert approach_ready_for_contact(3.02,3.,604)
