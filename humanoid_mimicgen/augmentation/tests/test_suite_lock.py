import pytest
from humanoid_mimicgen.augmentation.suite_lock import exclusive


def test_same_study_cannot_have_two_owners(tmp_path):
    with exclusive(tmp_path,'study'):
        with pytest.raises(RuntimeError,match='already owned'):
            with exclusive(tmp_path,'study'):pass
    with exclusive(tmp_path,'study'):pass


def test_independent_tasks_can_run_under_one_study(tmp_path):
    with exclusive(tmp_path,'study'):
        with exclusive(tmp_path/'a','task'):
            with exclusive(tmp_path/'b','task'):pass
