import numpy as np
import pytest
from humanoid_mimicgen.augmentation.suite_inspect import validate_trace


def test_inspection_rejects_missing_final_substep():
    trace={'state':np.zeros((8,3)),'next_state':np.zeros((7,3)),'commands':np.zeros((8,2))}
    with pytest.raises(ValueError):validate_trace(trace,8)


def test_inspection_preserves_full_coverage():
    trace={'state':np.zeros((8,3)),'next_state':np.zeros((8,3)),'commands':np.zeros((8,2))}
    assert validate_trace(trace,8)==8


def test_read_only_renderer_restores_physics_functions_even_on_error(monkeypatch,tmp_path):
    import mujoco
    from humanoid_mimicgen.augmentation import suite_inspect as s
    original=mujoco.mj_step
    def broken(*args,**kwargs):
        assert mujoco.mj_step is not original
        raise ValueError('render failure')
    monkeypatch.setattr(s,'_inspect_case',broken)
    with pytest.raises(ValueError,match='render failure'):s.inspect_case(tmp_path,tmp_path/'out')
    assert mujoco.mj_step is original
