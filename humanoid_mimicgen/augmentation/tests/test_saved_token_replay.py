import numpy as np
import pytest
from humanoid_mimicgen.augmentation.saved_token_replay import extract_tokens

def test_extract_keeps_all_token_values_and_excludes_encoder_targets():
    tokens=np.arange(24,dtype=np.float32).reshape(3,8)
    records=np.concatenate([np.zeros((3,2)),tokens,np.ones((3,1762+29))],axis=1)
    np.testing.assert_array_equal(extract_tokens(records,3),tokens)

@pytest.mark.parametrize('records,n',[(np.zeros((2,1801)),3),(np.zeros((3,1793)),3),(np.full((3,1801),np.nan),3)])
def test_reject_incomplete_invalid_or_nonfinite(records,n):
    with pytest.raises(ValueError):extract_tokens(records,n)
