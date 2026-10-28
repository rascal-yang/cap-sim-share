"""Durable out-of-process collection; no hooks inside physical control loops."""
import fcntl
import os
import signal
import subprocess
import threading
import time
import uuid
from .ingest import ingest
from .schema import now
from .store import atomic


def collect_once(dataset, roots):
    folder = dataset.root/'collector'; folder.mkdir(exist_ok=True)
    with (folder/'scan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = now(); tick = time.monotonic()
        atomic(folder/'status.json', {'pid': os.getpid(), 'status': 'scanning', 'started_at': start})
        try:
            report = ingest(dataset, roots)
            brief = {k: v for k, v in report.items() if k != 'entries'}
            atomic(folder/'status.json', {'pid': os.getpid(), 'status': 'idle', 'started_at': start,
                                          'ended_at': now(), 'wall_s': time.monotonic()-tick, 'last_report': brief})
            return report
        except Exception as exc:
            atomic(folder/'status.json', {'pid': os.getpid(), 'status': 'error', 'ended_at': now(), 'error_type': type(exc).__name__})
            raise


def watch(dataset, roots, interval=60):
    if interval < 1: raise ValueError('interval must be at least one second')
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        try:
            report = collect_once(dataset, roots)
            print({'at': now(), 'counts': report['counts'], 'discovered': report['discovered']}, flush=True)
        except Exception as exc:
            print({'at': now(), 'error_type': type(exc).__name__}, flush=True)
        stop.wait(interval)
    atomic(dataset.root/'collector'/'status.json', {'pid': os.getpid(), 'status': 'stopped', 'ended_at': now()})


def run_command(dataset, roots, command):
    """Explicit user-selected process, no shell. Exit code is not task acceptance."""
    if not command: raise ValueError('Command required')
    ident = uuid.uuid4().hex; path = dataset.root/'collector/runs'/(ident+'.json')
    event = {'run_id': ident, 'started_at': now(), 'status': 'running', 'exit_code': None,
             'roots': [str(r) for r in roots], 'command_recorded': False}
    atomic(path, event)
    try:
        process = subprocess.Popen(command)
        event['pid'] = process.pid; atomic(path, event)
        while True:
            try:
                code = process.wait(timeout=30); break
            except subprocess.TimeoutExpired:
                try: collect_once(dataset, roots)
                except Exception as exc:
                    event['collection_error'] = type(exc).__name__; atomic(path, event)
        event.update(status='exited', ended_at=now(), exit_code=code); atomic(path, event)
        try: collect_once(dataset, roots)
        except Exception as exc:
            event['collection_error'] = type(exc).__name__; atomic(path, event)
        return code
    except BaseException as exc:
        event.update(status='interrupted', ended_at=now(), error_type=type(exc).__name__); atomic(path, event)
        raise
