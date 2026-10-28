"""Process-owned leases prevent overlapping writers; kernel releases locks on exit."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import fcntl,json,os
from ..workflow.contracts import now


@contextmanager
def exclusive(root,kind):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/f'.{kind}.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'{kind} directory already owned by another running process: {root}') from None
        f.seek(0);f.truncate();f.write(json.dumps({'pid':os.getpid(),'started_at':now()}));f.flush()
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def exclusive_task(fn):
    @wraps(fn)
    def wrapped(dataset,root,*args,**kwargs):
        with exclusive(root,'task'):return fn(dataset,root,*args,**kwargs)
    return wrapped
