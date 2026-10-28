from humanoid_mimicgen.transfer_feedback_wait import standing_timeout_reached

def test_wait_continues_past_old_deadline_and_stops_at_fourteen_seconds():
    assert not standing_timeout_reached(2200)
    assert not standing_timeout_reached(2799)
    assert standing_timeout_reached(2800)
    assert standing_timeout_reached(2801)
