"""Single-device durable execution with a separate wall-clock supervisor."""
from contextlib import contextmanager
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import time

from .contracts import require, verify_inputs
from .conditions import sample_conditions
from .jobs import JobStore
from .scene import materialize_case
from ..workflow import service
from ..workflow.contracts import WorkflowError, canonical, file_hash, now


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks': stat[19]}


def process_alive(owner):
    try:
        if not isinstance(owner, dict) or type(owner.get('pid')) is not int or owner['pid'] <= 0:
            return False
        actual = process_identity(owner['pid'])
        state = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')', 1)[1].split()[0]
        return actual == owner and state not in ('Z', 'X')
    except (OSError, ValueError, IndexError):
        return False


@contextmanager
def device_lock(wait_s=0.):
    # Shared across stores/worktrees for this Unix account. Descriptor inherited by child.
    path = Path('/tmp')/f'cap-sim-augmentation-{os.getuid()}.lock'
    with path.open('a') as stream:
        deadline = time.monotonic()+wait_s
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(min(.05, max(0., deadline-time.monotonic())))
        # Closing this descriptor releases exclusion only after inherited child descriptors close.
        yield True



def _reconcile(store, job_id, reason='worker_disappeared', consumed_s=None):
    row = store.get(job_id)
    if row['status'] != 'running' or process_alive(row['owner']):
        return row
    for name, case in row['cases'].items():
        if case['stage'] in ('verified', 'failed', 'exported', 'interrupted'):
            continue
        root = Path(case['details'].get('experiment_path', '/nonexistent'))
        has_evidence = (root/'result.json').is_file() or (root/'execution/execution_audit.json').is_file()
        stage = 'awaiting_verification' if has_evidence else 'interrupted'
        store.record_case(job_id, name, stage, {'reason_code': reason, 'ended_at': now()})
    if consumed_s is None:
        from datetime import datetime
        consumed_s = row['details'].get('budget_used_s', 0.) + max(0., (datetime.fromisoformat(now())-datetime.fromisoformat(row['updated_at'])).total_seconds())
    store.transition(job_id, 'running', 'interrupted', {'reason_code': reason, 'ended_at': now(), 'budget_used_s': consumed_s})
    return store.get(job_id)


def reconcile_job(store, job_id):
    with device_lock() as acquired:
        if not acquired:
            return store.get(job_id)
        return _reconcile(store, job_id)


def _execute_job(store_root, job_id, supervisor_pid):
    import ctypes
    # Kill the physical process if its supervisor is killed, including SIGKILL.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != supervisor_pid:
        os._exit(1)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.setsid()
    store = JobStore(store_root)
    if not store.claim(job_id, process_identity()):
        return
    started = time.monotonic()
    row = store.get(job_id); job = row['job']; source_hashes = row['input_hashes']
    directory = store.root/'artifacts'/job_id
    directory.mkdir(parents=True, exist_ok=True)
    try:
        verify_inputs(source_hashes)
        implementations = {str(p): file_hash(p) for p in service.implementation_files().values()}
        cases = sample_conditions(job)
        for case_number, condition in enumerate(cases):
            name = condition['case_id']
            previous = store.get(job_id)['cases'].get(name)
            if previous and previous['stage'] in ('verified', 'failed', 'exported', 'interrupted'):
                continue
            # Never replay an unconfirmed executing stage after an interruption.
            if previous and previous['stage'] == 'awaiting_verification':
                root = Path(previous['details']['experiment_path'])
                try:
                    if not (root/'result.json').is_file():
                        service.check_contract(root/'experiment.json')
                        from ..workflow.verification import verify
                        verify(root)
                    from .export import export_case
                    checked = export_case(root, directory/'recovery-verification'/name)
                    result = json.loads((root/'result.json').read_text())
                    store.record_case(job_id, name, 'verified', {'result': result,
                        'success': checked['physical_success'], 'ended_at': now()})
                except Exception as exc:
                    store.record_case(job_id, name, 'failed', {'success': False,
                        'reason_code': exc.code if isinstance(exc, WorkflowError) else 'recovery_verification_failed', 'ended_at': now()})
                continue
            require(previous is None, 'Unfinished case requires explicit new attempt', 'interrupted')
            verify_inputs(source_hashes); verify_inputs(implementations)
            case_start = time.monotonic(); stage_times = {}; experiment = None
            store.record_case(job_id, name, 'materializing', {'started_at': now(), 'condition': condition})
            try:
                path = materialize_case(job, condition, directory/name)
                request = json.loads(path.read_text())
                for stage in ('plan', 'prepare', 'execute'):
                    verify_inputs(source_hashes); verify_inputs(implementations)
                    stage_start = now(); clock = time.monotonic()
                    if stage == 'plan':
                        result = service.plan(request); strategy_id = result['strategy_id']
                    elif stage == 'prepare':
                        if 'motion_noise' in job:
                            from .noise import freeze_noise
                            noise = dict(job['motion_noise'], seed=job['motion_noise']['seed']+case_number)
                            with freeze_noise(noise): result = service.prepare(strategy_id, request)
                        else: result = service.prepare(strategy_id, request)
                        experiment = Path(result['experiment_path'])
                        store.record_case(job_id, name, 'prepared', {'experiment_path': str(experiment)})
                    else:
                        store.record_case(job_id, name, 'executing', {'execution_started_at': now()})
                        if 'motion_noise' in job:
                            from .noise import execute_case
                            result = execute_case(experiment)
                        else: result = service.execute(experiment)
                    stage_times[stage] = {'started_at': stage_start, 'ended_at': now(), 'wall_s': time.monotonic()-clock}
                    (directory/name/(stage+'.json')).write_bytes(canonical(result))
                details = {'result': result, 'success': result.get('physical_task_completed') is True,
                           'timing': stage_times, 'total_wall_s': time.monotonic()-case_start,
                           'ended_at': now(), 'rendering_wall_s': None}
                if job['render']:
                    rt = time.monotonic()
                    with (directory/name/'render.log').open('w') as log:
                        process = subprocess.run([str(service.REPO/'scripts/humano_python.sh'), '-m',
                            'humanoid_mimicgen.render_standing', str(experiment)], cwd=service.REPO,
                            stdout=log, stderr=subprocess.STDOUT)
                    details.update(rendering_wall_s=time.monotonic()-rt, render_returncode=process.returncode)
                store.record_case(job_id, name, 'verified', details)
            except Exception as exc:
                details = {'success': False, 'reason_code': exc.code if isinstance(exc, WorkflowError) else 'case_error',
                           'error_type': type(exc).__name__, 'timing': stage_times,
                           'ended_at': now(), 'total_wall_s': time.monotonic()-case_start}
                if experiment is not None: details['experiment_path'] = str(experiment)
                store.record_case(job_id, name, 'failed', details)
            (directory/(name+'-reflection.json')).write_bytes(canonical(store.get(job_id)['cases'][name]))
        current = store.get(job_id)
        summary = {'successes': sum(c['details'].get('success') is True for c in current['cases'].values()),
                   'case_count': len(cases), 'ended_at': now(), 'total_wall_s': time.monotonic()-started,
                   'artifact_path': str(directory), 'budget_used_s': row['details'].get('budget_used_s', 0.)+time.monotonic()-started, 'physics_success_is_not_batch_status': True}
        summary['success_rate'] = summary['successes']/len(cases)
        summary['export_wall_s'] = None
        if job['export']:
            from .export import export_job
            export_started = time.monotonic()
            summary['export'] = export_job(store, job_id)
            summary['export_wall_s'] = time.monotonic()-export_started
        summary.update(ended_at=now(), total_wall_s=time.monotonic()-started,
                       budget_used_s=row['details'].get('budget_used_s', 0.)+time.monotonic()-started)
        (directory/'summary.json').write_bytes(canonical(summary))
        store.transition(job_id, 'running', 'completed', summary)
    except Exception as exc:
        store.transition(job_id, 'running', 'failed', {'reason_code': exc.code if isinstance(exc, WorkflowError) else 'worker_error',
                          'error_type': type(exc).__name__, 'ended_at': now(), 'total_wall_s': time.monotonic()-started})


def _stop_child(child):
    if child.pid is None:
        return
    if child.is_alive():
        try:
            if os.getpgid(child.pid) == child.pid: os.killpg(child.pid, signal.SIGTERM)
            else: child.terminate()
        except ProcessLookupError:
            pass
        child.join(1)
        if child.is_alive():
            try:
                if os.getpgid(child.pid) == child.pid: os.killpg(child.pid, signal.SIGKILL)
                else: child.kill()
            except ProcessLookupError:
                pass
    child.join()


def run_job(store, job_id):
    """Supervise child, including device wait and interruption in its wall budget."""
    row = store.get(job_id)
    if row['status'] != 'queued': return reconcile_job(store, job_id)
    remaining = float(row['job']['budgets']['wall_s'])-row['details'].get('budget_used_s', 0.)
    clock = time.monotonic()
    with device_lock(wait_s=max(0., remaining)) as acquired:
        row = store.get(job_id)
        if row['status'] != 'queued': return row
        remaining -= time.monotonic()-clock
        if not acquired or remaining <= 0:
            store.transition(job_id, 'queued', 'failed', {'reason_code': 'wall_budget_exceeded'})
            return store.get(job_id)
        child = mp.get_context('fork').Process(target=_execute_job, args=(str(store.root), job_id, os.getpid()))
        def interrupted(signum, frame): raise InterruptedError('Supervisor terminated')
        old_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            child.start(); child.join(remaining)
            if child.is_alive():
                _stop_child(child)
                return _reconcile(store, job_id, 'wall_budget_exceeded', row['details'].get('budget_used_s', 0.)+time.monotonic()-clock)
            child.join()
            return _reconcile(store, job_id, consumed_s=row['details'].get('budget_used_s', 0.)+time.monotonic()-clock)
        except BaseException:
            _stop_child(child)
            _reconcile(store, job_id, 'supervisor_interrupted', row['details'].get('budget_used_s', 0.)+time.monotonic()-clock)
            raise
        finally:
            signal.signal(signal.SIGTERM, old_handler)


def resume_job(store, job_id, *, background=True):
    row = reconcile_job(store, job_id)
    if row['status'] == 'interrupted':
        store.transition(job_id, 'interrupted', 'queued', {'resumed_at': now()})
    return start_job(store, job_id) if background else run_job(store, job_id)


def start_job(store, job_id):
    row = store.get(job_id)
    if row['status'] != 'queued':
        return {'job_id': job_id, 'status': row['status'], 'started': False}
    directory = store.root/'artifacts'/job_id; directory.mkdir(parents=True, exist_ok=True)
    with (directory/'worker.log').open('a') as log:
        p = subprocess.Popen([str(service.REPO/'scripts/humano_python.sh'), '-m',
            'humanoid_mimicgen.augmentation.worker', '--store', str(store.root), '--job-id', job_id],
            cwd=service.REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    return {'job_id': job_id, 'status': 'queued', 'started': True, 'supervisor': process_identity(p.pid)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True); parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    result = run_job(JobStore(args.store), args.job_id)
    print(canonical({'job_id': args.job_id, 'status': result['status']}).decode())
