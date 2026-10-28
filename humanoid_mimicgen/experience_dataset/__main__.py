"""python -m humanoid_mimicgen.experience_dataset --help"""
import argparse
import json
from pathlib import Path
from .store import Dataset
from .collector import collect_once, watch, run_command
from .retrieval import query_experience, get_experience, TOOL_DEFINITIONS


def main(argv=None):
    p = argparse.ArgumentParser(description='Local evidence-backed experience dataset; no implicit physics/API calls')
    p.add_argument('--dataset', default=str(Path(__file__).resolve().parents[3]/'experience-dataset'))
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('ingest', 'watch'):
        cmd = sub.add_parser(name); cmd.add_argument('roots', nargs='+')
        if name == 'watch': cmd.add_argument('--interval', type=float, default=60)
    cmd = sub.add_parser('snapshot'); cmd.add_argument('name'); cmd.add_argument('--exclude-family', action='append', default=[])
    cmd = sub.add_parser('query'); cmd.add_argument('--snapshot', required=True); cmd.add_argument('--text', default='')
    cmd.add_argument('--kind', choices=['orchestration', 'control', 'both'], default='orchestration')
    for key in ('task-id', 'robot', 'hand', 'interface', 'controller-version', 'stage', 'current-attempt-id'): cmd.add_argument('--'+key)
    cmd.add_argument('--exclude-family', action='append', default=[]); cmd.add_argument('--max-chars', type=int, default=12000)
    cmd.add_argument('--actual-modes', type=json.loads, help='JSON array of actual control modes, including original_token prefix if present')
    cmd = sub.add_parser('get'); cmd.add_argument('attempt_id'); cmd.add_argument('revision', type=int)
    cmd.add_argument('--snapshot', required=True)
    for name in ('stats', 'rebuild', 'tools'): sub.add_parser(name)
    cmd = sub.add_parser('run'); cmd.add_argument('--root', action='append', required=True); cmd.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args(argv); ds = Dataset(a.dataset)
    if a.command == 'ingest': result = {k: v for k, v in collect_once(ds, a.roots).items() if k != 'entries'}
    elif a.command == 'watch': watch(ds, a.roots, a.interval); return 0
    elif a.command == 'run': return run_command(ds, a.root, a.args[1:] if a.args[:1] == ['--'] else a.args)
    elif a.command == 'snapshot': result = {'snapshot': ds.snapshot(a.name, excluded_families=a.exclude_family)}
    elif a.command == 'query':
        q = {k: v for k, v in vars(a).items() if k not in ('command', 'dataset', 'exclude_family') and v is not None}
        q['excluded_families'] = a.exclude_family; result = query_experience(ds, q)
    elif a.command == 'get': result = get_experience(ds, a.attempt_id, a.revision, snapshot=a.snapshot)
    elif a.command == 'stats': result = ds.stats()
    elif a.command == 'rebuild': result = ds.rebuild()
    else: result = TOOL_DEFINITIONS
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == '__main__': raise SystemExit(main())
