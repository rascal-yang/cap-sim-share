import pytest
from repro.bridge import ContinuousContactGate, classify_result


def test_contact_gate_crosses_command_boundary_and_ignores_duplicate_reads():
    g=ContinuousContactGate()
    for i in range(1,10):
        assert not g.update(i*.005,6)
        assert not g.update(i*.005,6)
    assert g.update(.05,6)


def test_contact_gate_resets_only_on_measured_clearance():
    g=ContinuousContactGate()
    for i in range(1,8):g.update(i*.005,8)
    g.update(.04,0)
    assert not g.update(.045,8)


def test_zero_step_gate_can_replan_but_collision_is_terminal():
    assert classify_result({'failure':'hand_contact_gate','samples':[]},False)==(False,'hand_contact_gate')
    assert classify_result({'failure':'unexpected_table_contact','samples':[{}]},False)==(True,'unexpected_table_contact')
    assert classify_result({'failure':None,'samples':[{}]},True)==(True,'native_success')
