import multiprocessing as mp
import pytest
from humanoid_mimicgen.augmentation.jobs import JobStore
from humanoid_mimicgen.workflow.contracts import WorkflowError


def submit(root):
    return JobStore(root).submit({'request_id': 'same', 'value': 7}, {'a': 'hash'})


def claim(root, job_id, number, queue):
    queue.put(JobStore(root).claim(job_id, {'pid': number}))


def test_duplicate_submission_reuses_job(tmp_path):
    a = submit(tmp_path); b = submit(tmp_path)
    assert a['job_id'] == b['job_id']
    assert a['reused'] is False and b['reused'] is True
    assert JobStore(tmp_path).get(a['job_id'])['job']['value'] == 7


def test_same_id_changed_content_conflicts(tmp_path):
    submit(tmp_path)
    with pytest.raises(WorkflowError, match='conflict'):
        JobStore(tmp_path).submit({'request_id': 'same', 'value': 8}, {'a': 'hash'})
    with pytest.raises(WorkflowError, match='conflict'):
        JobStore(tmp_path).submit({'request_id': 'same', 'value': 7}, {'a': 'changed'})


def test_concurrent_claim_has_one_owner(tmp_path):
    ident = submit(tmp_path)['job_id']; ctx = mp.get_context('spawn'); queue = ctx.Queue()
    processes = [ctx.Process(target=claim, args=(str(tmp_path), ident, i, queue)) for i in range(2)]
    for p in processes: p.start()
    for p in processes: p.join(10); assert p.exitcode == 0
    assert sorted([queue.get(timeout=1), queue.get(timeout=1)]) == [False, True]
    assert JobStore(tmp_path).get(ident)['status'] == 'running'


def test_call_id_bound_to_arguments(tmp_path):
    store = JobStore(tmp_path)
    assert store.lookup_call('session', 'call', 'args') is None
    store.record_call('session', 'call', 'args', {'job_id': 'j'})
    assert store.lookup_call('session', 'call', 'args') == {'job_id': 'j'}
    with pytest.raises(WorkflowError, match='conflict'):
        store.lookup_call('session', 'call', 'other')
    with pytest.raises(WorkflowError, match='conflict'):
        store.record_call('session', 'call', 'args', {'job_id': 'different'})


def test_transaction_survives_process_exit(tmp_path):
    ctx = mp.get_context('spawn'); p = ctx.Process(target=submit, args=(str(tmp_path),))
    p.start(); p.join(10); assert p.exitcode == 0
    assert submit(tmp_path)['reused'] is True


def test_transition_and_case_history_preserved(tmp_path):
    store = JobStore(tmp_path); ident = submit(tmp_path)['job_id']
    assert store.claim(ident, {'pid': 1})
    store.record_case(ident, 'case_a', 'prepared', {'experiment': 'path'})
    store.record_case(ident, 'case_a', 'executing', {'started_at': 'time'})
    actual = store.get(ident)
    assert actual['cases']['case_a']['stage'] == 'executing'
    assert actual['cases']['case_a']['details']['experiment'] == 'path'
    store.transition(ident, 'running', 'completed', {'successes': 0})
    with pytest.raises(WorkflowError): store.transition(ident, 'queued', 'running', {})
    assert not store.claim(ident, {'pid': 2})
