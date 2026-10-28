import json

import pytest

from humanoid_mimicgen.pipeline import generate_batch
from test_dataset_annotation import demo_fixture, annotation_fixture


def test_failed_batch_preserves_completed_case_and_reports_failure(tmp_path):
    demo = demo_fixture()
    annotation = annotation_fixture(demo)
    cases = [
        {'id': 'valid', 'translation': [0, 0, 0]},
        {'id': 'invalid', 'translation': [float('nan'), 0, 0]},
        {'id': 'unreached', 'translation': [0, 0, 0]},
    ]
    with pytest.raises(ValueError):
        generate_batch(demo, annotation, cases, tmp_path / 'batch')
    assert (tmp_path / 'batch/valid/trajectory.npz').is_file()
    assert not (tmp_path / 'batch/unreached').exists()
    result = json.loads((tmp_path / 'batch/summary.json').read_text())
    assert result['status'] == 'error'
    assert result['failed_case'] == 'invalid'
    assert result['completed_case_count'] == 1
    assert result['error_type'] == 'ValueError'
