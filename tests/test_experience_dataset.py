import json
import pytest

from humanoid_mimicgen.experience_dataset.schema import new_record, validate_record, classify_response
from humanoid_mimicgen.experience_dataset.store import Dataset


def record(source='case/001'):
    return new_record(source, task='apple', producer='test')


def test_unknown_is_not_failure_and_bool_is_strict():
    r = record()
    assert r['outcome']['task_success'] is None
    validate_record(r)
    r['outcome']['task_success'] = 'false'
    with pytest.raises(ValueError):
        validate_record(r)


def test_response_requires_comparable_measurements():
    args = dict(desired=0.02, measured=-0.01, frame='world', unit='m', window_s=1., tolerance=.001)
    assert classify_response(**args) == 'opposite'
    assert classify_response(**dict(args, measured=.005)) == 'insufficient'
    assert classify_response(**dict(args, measured=.0205)) == 'tracked'
    assert classify_response(**dict(args, frame=None)) == 'unknown'
    assert classify_response(**dict(args, window_s=0)) == 'unknown'
    assert classify_response(**dict(args, measured=float('nan'))) == 'unknown'


def test_revisions_are_immutable_and_idempotent(tmp_path):
    ds = Dataset(tmp_path)
    r = record()
    first = ds.put(r)
    assert ds.put(r) == first
    r['outcome']['task_success'] = True
    second = ds.put(r)
    assert second['revision'] == 2
    assert ds.get(first['attempt_id'], 1)['outcome']['task_success'] is None
    assert ds.get(first['attempt_id'], 2)['outcome']['task_success'] is True
    assert record('case/002')['attempt_id'] != r['attempt_id']
    ds.rebuild()
    assert ds.stats()['attempts'] == 1


def test_snapshot_binds_old_revision_and_corruption_rejected(tmp_path):
    ds = Dataset(tmp_path)
    first = ds.put(record())
    snap = ds.snapshot('before')
    r = record(); r['outcome']['task_success'] = False
    ds.put(r)
    assert ds.load_snapshot(snap)['records'][first['attempt_id']]['revision'] == 1
    path = ds.record_path(first['attempt_id'], 1)
    value = json.loads(path.read_text()); value['task_id'] = 'changed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='integrity'):
        ds.get(first['attempt_id'], 1)


def test_record_id_cannot_escape_store(tmp_path):
    ds = Dataset(tmp_path)
    with pytest.raises(ValueError):
        ds.get('../../escape', 1)


def test_chinese_fts_index_is_rebuildable(tmp_path):
    ds=Dataset(tmp_path); r=record()
    r['orchestration']=[{'instruction':'苹果抬升不足'}]; ds.put(r)
    with ds.connect() as con:
        assert con.execute("SELECT count(*) FROM search WHERE search MATCH ?", ('抬升',)).fetchone()[0]==1
    ds.rebuild()
    with ds.connect() as con:
        assert con.execute("SELECT count(*) FROM search WHERE search MATCH ?", ('抬升',)).fetchone()[0]==1
