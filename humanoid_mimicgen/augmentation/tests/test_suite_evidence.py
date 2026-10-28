import math
import pytest
from humanoid_mimicgen.augmentation.suite_evidence import encode_infinities


def test_metadata_infinity_preserves_sign_and_does_not_modify_source():
    value={'range':[-math.inf,math.inf],'distance':math.inf,'finite':2.}
    out=encode_infinities(value)
    assert out['range']==[{'ieee754':'-Infinity'},{'ieee754':'+Infinity'}]
    assert out['distance']=={'ieee754':'+Infinity'} and out['finite']==2.
    assert value['range'][0]==-math.inf


def test_nan_is_not_an_accepted_unbounded_sentinel():
    with pytest.raises(ValueError):encode_infinities({'state':float('nan')})
