"""Run from the cap-sim repository: python -m humanoid_mimicgen."""

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def ensure_output_family(primary, sidecars=()):
    for path in [Path(primary), *map(Path, sidecars)]:
        if path.exists():
            raise FileExistsError(path)


def create_experiment(root, description):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_]*', description):
        raise ValueError('Description must contain lowercase letters, digits and underscores')
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    name = now.strftime('%Y%m%d_%H%M_') + description
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sequence = 0
    while True:
        out = root / (name if sequence == 0 else f'{name}_{sequence:02d}')
        try:
            out.mkdir()
            break
        except FileExistsError:
            sequence += 1
    write_new(out / 'creation.json', {'experiment_id': out.name,
              'created_at': now.isoformat(timespec='seconds'), 'timezone': 'Asia/Shanghai'})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description='GPT-6 annotated contact trajectory retargeting')
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('init-experiment', help='Create a unique timestamped experiment directory')
    c.add_argument('--root', default='../experiments')
    c.add_argument('--description', default='humanoid_mimicgen')
    c = sub.add_parser('run', help='Import, call GPT-6, annotate and generate ten offline variants')
    c.add_argument('--source', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--model', default='gpt-6-astra')
    c.add_argument('--provider', choices=['codex', 'openai'], default='codex')
    c.add_argument('--cases')
    c = sub.add_parser('import', help='Extract measured poses with FK; no physics stepping')
    c.add_argument('--source', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--object-body', default='apple_001')
    c = sub.add_parser('request', help='Build a compact GPT annotation evidence request')
    c.add_argument('--demo', required=True)
    c.add_argument('--out', required=True)
    c = sub.add_parser('annotate', help='Obtain and validate GPT-6 phases')
    c.add_argument('--demo', required=True)
    c.add_argument('--provider', choices=['codex', 'openai', 'file'], default='codex')
    c.add_argument('--model', default='gpt-6-astra')
    c.add_argument('--response')
    c.add_argument('--identity', help='Required provenance for an imported annotation')
    c.add_argument('--base-url', default='https://api.openai.com/v1')
    c.add_argument('--out', required=True)
    c = sub.add_parser('generate', help='Retarget contacts and generate transit; geometry only')
    c.add_argument('--demo', required=True)
    c.add_argument('--annotation', required=True)
    c.add_argument('--translation', nargs=3, type=float, default=[0., 0., 0.])
    c.add_argument('--rotation-wxyz', nargs=4, type=float, default=[1., 0., 0., 0.])
    c.add_argument('--preserve-prefix-s', type=float, default=5.)
    c.add_argument('--out', required=True)
    c = sub.add_parser('batch', help='Generate the fixed ten conditions from a validated annotation')
    c.add_argument('--demo', required=True)
    c.add_argument('--annotation', required=True)
    c.add_argument('--cases', default=str(Path(__file__).with_name('ten_conditions.json')))
    c.add_argument('--out', required=True)
    c = sub.add_parser('plot', help='Plot recorded evidence and offline generated references')
    c.add_argument('--demo', required=True)
    c.add_argument('--annotation', required=True)
    c.add_argument('--batch', required=True)
    c.add_argument('--experiment-id', required=True)
    c.add_argument('--out', required=True)
    c = sub.add_parser('prepare-physics', help='Prepare manifests without stepping physics')
    c.add_argument('--task', required=True)
    c.add_argument('--out', required=True)
    c = sub.add_parser('smoke', help='One physical rollout; requires sealed experiment contract')
    for name in ['task', 'demo', 'generated', 'annotation', 'parquet', 'contract', 'out']:
        c.add_argument('--' + name, required=True)
    args = p.parse_args(argv)
    if args.command == 'init-experiment':
        print(create_experiment(args.root, args.description))
        return
    if args.command == 'run':
        from .pipeline import run_offline
        result = run_offline(args.source, args.out, args.model, args.provider, args.cases)
        print(json.dumps({'case_count': result['case_count'], 'physics_steps': 0}))
        return
    if args.command == 'import':
        from .dataset import import_native
        demo = import_native(args.source, args.out, object_body=args.object_body)
        print(json.dumps({'frames': len(demo['time_s']), 'out': str(args.out)}))
        return
    if args.command == 'prepare-physics':
        from .native import prepare_manifest
        print(json.dumps(prepare_manifest(read(args.task), args.out), indent=2))
        return
    from .dataset import load_demo, save_demo
    demo = load_demo(args.demo)
    if args.command == 'batch':
        from .pipeline import generate_batch
        result = generate_batch(demo, read(args.annotation), read(args.cases), args.out)
        print(json.dumps({'case_count': result['case_count'], 'physics_steps': 0}))
    elif args.command == 'plot':
        from .visualize import plot_summary
        plot_summary(demo, read(args.annotation), read(args.batch), args.out, args.experiment_id)
        print(args.out)
    elif args.command == 'request':
        from .annotation import build_request
        write_new(args.out, build_request(demo))
    elif args.command == 'annotate':
        from .annotation import build_request, validate_annotation, annotate_openai
        out = Path(args.out)
        ensure_output_family(out, [out.with_name(out.stem + '_provenance.json')])
        request = build_request(demo)
        if args.provider == 'file':
            if not args.response or not args.identity:
                p.error('file annotations require --response and --identity')
            annotation = read(args.response)
            provenance = {'provider': 'file', 'identity': args.identity,
                          'external_model_call_in_this_command': False,
                          'response_path': str(Path(args.response).resolve()),
                          'response_sha256': hashlib.sha256(Path(args.response).read_bytes()).hexdigest()}
        elif args.provider == 'codex':
            from .model import annotate_codex
            annotation = annotate_codex(request, args.model, out.parent / (out.stem + '_model_call'))
            provenance = read(out.parent / (out.stem + '_model_call') / 'provenance.json')
        else:
            call = out.parent / (out.stem + '_model_call')
            annotation = annotate_openai(request, args.model, call, base_url=args.base_url)
            provenance = {'provider': 'openai_responses', 'model_requested': args.model,
                          'external_model_call': True, 'artifacts': str(call.resolve())}
        validate_annotation(annotation, demo)
        write_new(out, annotation)
        write_new(out.with_name(out.stem + '_provenance.json'), provenance)
        print(json.dumps({'validated': True, 'stages': len(annotation['stages']), 'out': str(out)}))
    elif args.command == 'generate':
        from .retarget import retarget_demo, geometry_metrics
        annotation = read(args.annotation)
        out = Path(args.out)
        ensure_output_family(out, [out.with_suffix('.json'), out.with_name(out.stem + '_geometry.json')])
        generated = retarget_demo(demo, annotation, args.translation, args.rotation_wxyz,
                                  preserve_prefix_s=args.preserve_prefix_s)
        metrics = geometry_metrics(demo, generated, annotation)
        if not metrics['invariants_passed']:
            raise ValueError('Generated trajectory failed geometric invariants')
        save_demo(generated, out)
        write_new(out.with_name(out.stem + '_geometry.json'), metrics)
        print(json.dumps(metrics, indent=2))
    elif args.command == 'smoke':
        from .native import run_smoke
        result = run_smoke(read(args.task), demo, load_demo(args.generated), read(args.annotation),
                           args.parquet, args.out, args.contract)
        print(json.dumps(result, indent=2))
