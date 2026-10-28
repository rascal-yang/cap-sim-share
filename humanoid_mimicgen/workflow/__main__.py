"""JSON CLI. Use scripts/humano_python.sh for all native operations."""
import argparse
import contextlib
import os
import json
from pathlib import Path
import sys
from .contracts import WorkflowError, canonical


@contextlib.contextmanager
def diagnostics_to_stderr():
    # Native C libraries may write directly to fd1; redirect both Python and fd1.
    sys.stdout.flush();saved=os.dup(1)
    try:
        os.dup2(2,1)
        with contextlib.redirect_stdout(sys.stderr):yield
    finally:
        sys.stderr.flush();os.dup2(saved,1);os.close(saved)


def main(argv=None):
    p=argparse.ArgumentParser(description='Independent reproducible manipulation workflow')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','prepare','resolve'):
        cmd=sub.add_parser(name);cmd.add_argument('--request',required=True)
        if name=='prepare':cmd.add_argument('--strategy-id',required=True)
    for name in ('execute','inspect','verify'):
        cmd=sub.add_parser(name);cmd.add_argument('--experiment',required=True)
    cmd=sub.add_parser('calibrate-stop');cmd.add_argument('--experiments',nargs='+',required=True);cmd.add_argument('--output',required=True)
    cmd=sub.add_parser('batch');cmd.add_argument('--manifest',required=True)
    args=p.parse_args(argv)
    try:
        with diagnostics_to_stderr():
            from . import service
            if args.command=='calibrate-stop':
                from .parking import calibrate_stopping
                result=calibrate_stopping(args.experiments,args.output)
            elif args.command=='batch':
                from .batch import run_batch
                result=run_batch(args.manifest)
            elif args.command in ('plan','prepare','resolve'):
                path=Path(args.request).resolve();request=json.loads(path.read_text());base=path.parent
                def paths(r):
                    r['files']={k:str((base/v).resolve()) for k,v in r['files'].items()}
                    for k in ('strategy_root','experiment_root'):r[k]=str((base/r[k]).resolve())
                    if r['decision']['mode']=='saved':r['decision']['path']=str((base/r['decision']['path']).resolve())
                if args.command=='resolve':
                    for option in request['options']:paths(option['request'])
                    request['experiment_root']=str((base/request['experiment_root']).resolve())
                    if request.get('recovery_from'):request['recovery_from']=str((base/request['recovery_from']).resolve())
                    if request['decision']['mode']=='saved':request['decision']['path']=str((base/request['decision']['path']).resolve())
                    from .selection import resolve
                    result=resolve(request)
                else:
                    paths(request)
                    result=service.plan(request) if args.command=='plan' else service.prepare(args.strategy_id,request)
            elif args.command=='verify':
                from .verification import verify
                result=verify(args.experiment)
            else:result=getattr(service,args.command)(args.experiment)
        print(canonical(result).decode());return 0 if result.get('status')!='failed' else 1
    except WorkflowError as exc:
        print(canonical({'status':'failed','reason_code':exc.code,'detail':exc.detail}).decode());return 2
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(canonical({'status':'failed','reason_code':'invalid_input','error_type':type(exc).__name__}).decode());return 2

if __name__=='__main__':sys.exit(main())
