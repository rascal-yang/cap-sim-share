"""JSON entrypoint. Run native operations using scripts/humano_python.sh."""
import argparse
import sys
from .contracts import load_job, validate_job
from .jobs import JobStore
from ..workflow.contracts import WorkflowError, canonical
from ..workflow.__main__ import diagnostics_to_stderr


def main(argv=None):
    parser = argparse.ArgumentParser(description='Skill/API-orchestrated local augmentation')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'submit', 'agent'):
        p = sub.add_parser(name); p.add_argument('--job', required=True)
        if name != 'validate': p.add_argument('--store', required=True)
        if name == 'agent': p.add_argument('--provider', required=True)
    for name in ('worker', 'inspect', 'resume', 'export'):
        p = sub.add_parser(name); p.add_argument('--store', required=True); p.add_argument('--job-id', required=True)
    p = sub.add_parser('init-experiment'); p.add_argument('--root', default='../experiments'); p.add_argument('--description', default='api_augmentation')
    p = sub.add_parser('index-history'); p.add_argument('--root', default='../experiments')
    p = sub.add_parser('execute-case'); p.add_argument('--experiment', required=True)
    args = parser.parse_args(argv)
    try:
        with diagnostics_to_stderr():
            if args.command == 'execute-case':
                from .noise import execute_case
                result = execute_case(args.experiment)
            elif args.command == 'init-experiment':
                from .records import create_record
                path = create_record(args.root, args.description)
                result = {'experiment_id': path.name, 'path': str(path.absolute()), 'physics_steps': 0}
            elif args.command == 'index-history':
                from .records import index_history
                result = index_history(args.root)
            elif args.command == 'validate':
                result = validate_job(load_job(args.job))
                result = {k: v for k, v in result.items() if k != 'input_hashes'} | {'bound_files': len(result.get('input_hashes', {}))}
            else:
                store = JobStore(args.store)
                from . import tools, worker
                if args.command == 'submit':
                    # CLI callers explicitly choose the fixed workflow; bind its skill just as API callers do.
                    context = {'job': load_job(args.job), 'store': store, 'skill_read': True}
                    result = tools.dispatch('submit_job', {}, context)
                elif args.command == 'agent':
                    from .agent import run_agent
                    result = run_agent(args.job, args.provider, store)
                elif args.command == 'worker': result = tools.summary(worker.run_job(store, args.job_id))
                elif args.command == 'inspect': result = tools.summary(worker.reconcile_job(store, args.job_id))
                elif args.command == 'resume': result = worker.resume_job(store, args.job_id)
                else:
                    from .export import export_job
                    result = export_job(store, args.job_id)
        print(canonical(result).decode())
        return 1 if result.get('status') in ('failed', 'interrupted') else 0
    except WorkflowError as exc:
        print(canonical({'status': 'failed', 'reason_code': exc.code, 'detail': exc.detail}).decode()); return 2
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(canonical({'status': 'failed', 'reason_code': 'invalid_input', 'error_type': type(exc).__name__}).decode()); return 2


if __name__ == '__main__': sys.exit(main())
