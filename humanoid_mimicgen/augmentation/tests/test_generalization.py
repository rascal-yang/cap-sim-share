import numpy as np
import pytest
from humanoid_mimicgen.augmentation.generalization_scene import compare_arrays

def test_only_declared_target_moves():
    base={'body_pos':np.zeros((3,3)),'mass':np.ones(3)}
    new={k:v.copy() for k,v in base.items()};new['body_pos'][2]=[.02,0,0]
    compare_arrays(new,base,2,[.02,0,0])
    new['body_pos'][1,0]=.001
    with pytest.raises(ValueError):compare_arrays(new,base,2,[.02,0,0])

def test_physics_change_rejected():
    base={'body_pos':np.zeros((3,3)),'mass':np.ones(3)}
    new={k:v.copy() for k,v in base.items()};new['mass'][0]=2
    with pytest.raises(ValueError):compare_arrays(new,base,2,[0,0,0])

def test_positions_fixed_unique_and_small():
    from humanoid_mimicgen.augmentation.generalization_benchmark import offsets
    x=offsets()
    assert len(x)==10 and len({tuple(p) for p in x})==10
    assert x[0]==[0,0]
    assert max(np.linalg.norm(p) for p in x)<=.04

def test_options_gravity_and_solver_rejected():
    from types import SimpleNamespace
    from humanoid_mimicgen.augmentation.generalization_scene import numeric_fields
    a=SimpleNamespace(gravity=np.array([0.,0.,-9.81]),iterations=50)
    b=SimpleNamespace(gravity=np.array([0.,0.,-1.]),iterations=50)
    with pytest.raises(ValueError):compare_arrays(numeric_fields(a),numeric_fields(b),0,[0,0,0])
    b.gravity=a.gravity.copy();b.iterations=20
    with pytest.raises(ValueError):compare_arrays(numeric_fields(a),numeric_fields(b),0,[0,0,0])

def test_process_addresses_excluded():
    from types import SimpleNamespace
    from humanoid_mimicgen.augmentation.generalization_scene import numeric_fields
    a=SimpleNamespace(_address=123,iterations=50)
    b=SimpleNamespace(_address=456,iterations=50)
    compare_arrays(numeric_fields(a),numeric_fields(b),0,[0,0,0])
