"""Check the approved CPU runtime before a new HUMANO experiment.

No physics or model files are modified. --load-models verifies actual ORT sessions.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, default=repo / 'runtime/humano-cpu.json')
    parser.add_argument('--models', type=Path, default=Path(os.environ.get(
        'CAP_SIM_HUMANO_ROOT', '/home/kinetra-4090-01/HUMANO')) / 'third_party/gear_sonic_deploy')
    parser.add_argument('--load-models', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if f'{sys.version_info.major}.{sys.version_info.minor}' != profile['python_version']:
        raise RuntimeError('Wrong Python version for approved HUMANO profile')
    versions = {name: importlib.metadata.version(name) for name in profile['versions']}
    if versions != profile['versions']:
        raise RuntimeError(f'Runtime version mismatch: {versions}')
    if importlib.util.find_spec('onnxruntime_gpu') or any(
        d.metadata.get('Name', '').lower().replace('_', '-') == 'onnxruntime-gpu'
        for d in importlib.metadata.distributions()
    ):
        raise RuntimeError('Use the pinned CPU onnxruntime distribution')
    import tensorrt
    import onnxruntime as ort
    if tensorrt.__version__ != versions['tensorrt'] or ort.__version__ != versions['onnxruntime']:
        raise RuntimeError('Imported modules do not match installed distribution versions')
    provider = profile['sonic']['provider']
    if provider != 'CPUExecutionProvider' or profile['sonic']['tensorrt_used_for_sonic_inference']:
        raise RuntimeError('Approved SONIC profile requires CPU inference')
    if provider not in ort.get_available_providers():
        raise RuntimeError('ORT CPU provider unavailable')
    hashes = {}
    sessions = {}
    for relative, expected in profile['sonic']['models'].items():
        path = args.models / relative
        with path.open('rb') as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        hashes[relative] = h.hexdigest()
        if hashes[relative] != expected:
            raise RuntimeError(f'SONIC model changed: {relative}')
        if args.load_models:
            opt = ort.SessionOptions()
            opt.intra_op_num_threads = 2
            opt.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), sess_options=opt, providers=[provider])
            sessions[relative] = session.get_providers()
            if sessions[relative] != [provider]:
                raise RuntimeError(f'Unexpected active model provider: {sessions[relative]}')
    result = {
        'passed': True, 'profile_id': profile['profile_id'], 'python': sys.executable,
        'versions': versions, 'sonic_version': profile['sonic']['experiment_version'],
        'sonic_package_version': versions['gear_sonic'], 'model_sha256': hashes,
        'onnxruntime_available_providers': ort.get_available_providers(),
        'actual_session_providers': sessions, 'sessions_loaded': args.load_models,
        'tensorrt_module': tensorrt.__file__, 'tensorrt_used_for_sonic_inference': False,
        'physics_steps': 0,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
