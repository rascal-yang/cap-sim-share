import json
import os
from pathlib import Path
import time
import pytest
from humanoid_mimicgen.augmentation.jobs import JobStore
from humanoid_mimicgen.augmentation import worker
from humanoid_mimicgen.workflow import service
from humanoid_mimicgen.workflow.contracts import file_hash


def fixture(tmp_path, monkeypatch, *, wall=5):
    # Fake the physical boundary; orchestration, process isolation, store, files and locks are real.
    from humanoid_mimicgen.augmentation.tests.test_conditions import job
    j = job(); j['conditions'] = {'mode': 'explicit', 'cases': [
        {'case_id': 'a', 'robot_delta_xyyaw': [0, 0, 0], 'object_delta_xy': [0, 0]},
        {'case_id': 'b', 'robot_delta_xyyaw': [0, 0, 0], 'object_delta_xy': [0, 0]}]}
    j.update(output_root=str(tmp_path/'output'), export=False)
    j['budgets']['wall_s'] = wall
    source = tmp_path/'input'; source.write_text('original')
    store = JobStore(tmp_path/'store'); ident = store.submit(j, {str(source): file_hash(source)})['job_id']
    def materialize(job, condition, directory):
        directory.mkdir(parents=True)
        p = directory/'request.json'; p.write_text(json.dumps({'case': condition['case_id']})); return p
    monkeypatch.setattr(worker, 'materialize_case', materialize)
    monkeypatch.setattr(service, 'implementation_files', lambda: {})
    monkeypatch.setattr(service, 'plan', lambda request: {'strategy_id': request['case']})
    def prepare(strategy, request):
        root = tmp_path/strategy; root.mkdir()
        (root/'experiment.json').write_text('frozen')
        return {'experiment_path': str(root), 'experiment_id': strategy}
    monkeypatch.setattr(service, 'prepare', prepare)
    def execute(path):
        path = Path(path)
        assert (path/'experiment.json').read_text() == 'frozen'
        (path/'execution').mkdir()
        (path/'execution/execution_audit.json').write_text('{}')
        result = {'status': 'completed', 'physical_task_completed': True}
        (path/'result.json').write_text(json.dumps(result))
        with (tmp_path/'executed').open('a') as f: f.write(path.name+'\n')
        return result
    monkeypatch.setattr(service, 'execute', execute)
    return store, ident, source


def test_all_cases_frozen_before_execution(tmp_path, monkeypatch):
    store, ident, _ = fixture(tmp_path, monkeypatch)
    result = worker.run_job(store, ident)
    assert result['status'] == 'completed'
    assert result['details']['successes'] == 2
    assert (tmp_path/'executed').read_text() == 'a\nb\n'
    assert all(c['stage'] == 'verified' for c in result['cases'].values())


def test_duplicate_start_does_not_execute_twice(tmp_path, monkeypatch):
    store, ident, _ = fixture(tmp_path, monkeypatch)
    worker.run_job(store, ident); worker.run_job(store, ident)
    assert (tmp_path/'executed').read_text() == 'a\nb\n'


def test_dead_worker_not_replayed(tmp_path, monkeypatch):
    store, ident, _ = fixture(tmp_path, monkeypatch)
    store.claim(ident, {'pid': 99999999, 'boot_id': 'gone', 'start_ticks': '0'})
    store.record_case(ident, 'a', 'executing', {'experiment_path': str(tmp_path/'partial')})
    result = worker.reconcile_job(store, ident)
    assert result['status'] == 'interrupted'
    assert result['cases']['a']['stage'] == 'interrupted'
    assert not (tmp_path/'executed').exists()


def test_complete_artifacts_resume_verification(tmp_path, monkeypatch):
    store, ident, _ = fixture(tmp_path, monkeypatch)
    root = tmp_path/'finished'; root.mkdir()
    (root/'result.json').write_text(json.dumps({'status': 'completed', 'physical_task_completed': True}))
    store.claim(ident, {'pid': 99999999, 'boot_id': 'gone', 'start_ticks': '0'})
    store.record_case(ident, 'a', 'executing', {'experiment_path': str(root)})
    result = worker.reconcile_job(store, ident)
    assert result['cases']['a']['stage'] == 'awaiting_verification'
    assert result['cases']['a']['details'].get('success') is not True
    assert not (tmp_path/'executed').exists()


def test_budget_stops_remaining_cases(tmp_path, monkeypatch):
    store, ident, _ = fixture(tmp_path, monkeypatch, wall=.15)
    monkeypatch.setattr(service, 'execute', lambda p: time.sleep(10))
    start = time.monotonic(); result = worker.run_job(store, ident)
    assert time.monotonic()-start < 3
    assert result['status'] == 'interrupted'
    assert result['details']['reason_code'] == 'wall_budget_exceeded'
    assert not (tmp_path/'b').exists()
    assert not worker.process_alive(result['owner'])


def test_changed_queued_input_blocks_launch(tmp_path, monkeypatch):
    store, ident, source = fixture(tmp_path, monkeypatch); source.write_text('modified')
    result = worker.run_job(store, ident)
    assert result['status'] == 'failed'
    assert result['details']['reason_code'] == 'integrity_error'
    assert not (tmp_path/'executed').exists()


def test_process_identity_prevents_pid_reuse():
    owner = worker.process_identity()
    assert worker.process_alive(owner)
    assert not worker.process_alive(dict(owner, start_ticks='-1'))


@pytest.mark.parametrize('sig', [15, 9])
def test_signal_interruption_and_stale_lock_do_not_replay(tmp_path, monkeypatch, sig):
    store, ident, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(service, 'execute', lambda p: os.kill(os.getpid(), sig))
    result = worker.run_job(store, ident)
    assert result['status'] == 'interrupted'
    assert result['cases']['a']['stage'] == 'interrupted'
    assert not worker.process_alive(result['owner'])
    with worker.device_lock() as acquired:
        assert acquired
    worker.run_job(store, ident)
    assert not (tmp_path/'executed').exists()


def test_resume_runs_only_unstarted_cases(tmp_path, monkeypatch):
    store, ident, _ = fixture(tmp_path, monkeypatch)
    store.claim(ident, {'pid': 99999999, 'boot_id': 'gone', 'start_ticks': '0'})
    store.record_case(ident, 'a', 'executing', {'experiment_path': str(tmp_path/'partial')})
    worker.reconcile_job(store, ident)
    result = worker.resume_job(store, ident, background=False)
    assert result['status'] == 'completed'
    assert result['details']['successes'] == 1
    assert result['details']['case_count'] == 2
    assert (tmp_path/'executed').read_text() == 'b\n'


def _hold_lock(queue):
    with worker.device_lock() as acquired:
        queue.put(acquired); time.sleep(.3)


def test_busy_device_queues_then_executes(tmp_path, monkeypatch):
    import multiprocessing as mp
    store,ident,_=fixture(tmp_path,monkeypatch)
    ctx=mp.get_context('spawn'); queue=ctx.Queue(); holder=ctx.Process(target=_hold_lock,args=(queue,))
    holder.start(); assert queue.get(timeout=5)
    try:
        result=worker.run_job(store,ident)
        assert result['status']=='completed'
    finally: holder.join(5)


def test_supervisor_interrupt_stops_child_before_unlock(tmp_path, monkeypatch):
    import multiprocessing as mp
    import signal
    store,ident,_=fixture(tmp_path,monkeypatch)
    ready=tmp_path/'physical_ready'
    def execute(p): ready.write_text('ready'); time.sleep(20)
    monkeypatch.setattr(service,'execute',execute)
    supervisor=mp.get_context('fork').Process(target=worker.run_job,args=(store,ident))
    supervisor.start()
    owner=None
    try:
        deadline=time.monotonic()+4
        while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)
        assert ready.exists()
        owner=store.get(ident)['owner']
        os.kill(supervisor.pid,signal.SIGINT); supervisor.join(3)
        assert not supervisor.is_alive()
        assert not worker.process_alive(owner)
        with worker.device_lock() as acquired: assert acquired
    finally:
        if owner and worker.process_alive(owner):
            os.killpg(owner['pid'],signal.SIGKILL)
        if supervisor.is_alive(): supervisor.kill(); supervisor.join()


def test_recovery_does_not_trust_cached_success(tmp_path, monkeypatch):
    store,ident,_=fixture(tmp_path,monkeypatch)
    root=tmp_path/'cached'; root.mkdir()
    (root/'result.json').write_text(json.dumps({'physical_task_completed':True,'status':'completed'}))
    store.claim(ident,{'pid':99999999,'boot_id':'gone','start_ticks':'0'})
    store.record_case(ident,'a','executing',{'experiment_path':str(root)})
    monkeypatch.setattr(service,'check_contract',lambda path:{})
    worker.reconcile_job(store,ident)
    result=worker.resume_job(store,ident,background=False)
    assert result['cases']['a']['stage']=='failed'
    assert result['cases']['a']['details'].get('success') is not True
    assert (tmp_path/'executed').read_text()=='b\n'


def test_total_wall_time_includes_export(tmp_path, monkeypatch):
    from humanoid_mimicgen.augmentation import export
    store,ident,source=fixture(tmp_path,monkeypatch)
    j=store.get(ident)['job']; j.update(request_id='with-export',export=True)
    ident=store.submit(j,{str(source):file_hash(source)})['job_id']
    def slow_export(s,i):time.sleep(.15);return {'case_count':2}
    monkeypatch.setattr(export,'export_job',slow_export)
    result=worker.run_job(store,ident)
    assert result['details']['export_wall_s']>=.15
    assert result['details']['total_wall_s']>=sum(c['details']['total_wall_s'] for c in result['cases'].values())+.15
