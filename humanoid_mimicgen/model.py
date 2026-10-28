"""GPT-6 annotation through the user's existing Codex login; no key extraction."""

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def annotate_codex(request, model, outdir, timeout=300):
    if not model.startswith('gpt-6'):
        raise ValueError('This annotator requires an explicit GPT-6 model')
    from .annotation import ANNOTATION_SCHEMA, PROMPT

    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    schema = out / 'schema.json'
    schema.write_text(json.dumps(ANNOTATION_SCHEMA, indent=2) + '\n')
    (out / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    prompt = (
        PROMPT + '\n\nThis is a pure annotation request. Do not use tools, read '
        'workspace files, or execute commands. Use only the evidence below. '
        'Return the requested JSON. Stage names for physical lifting must '
        'contain "lift".\n\n' + json.dumps(request)
    )
    (out / 'prompt.txt').write_text(prompt)
    cmd = ['codex', 'exec', '--model', model, '--sandbox', 'read-only',
           '--ephemeral', '--skip-git-repo-check', '--color', 'never',
           '--output-schema', str(schema), '--output-last-message',
           str(out / 'response.json'), '--json', '-']
    started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    tick = time.perf_counter()
    status, code, error_type, launched = 'error', None, None, False
    try:
        with (out / 'events.jsonl').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            result = subprocess.run(cmd, input=prompt, text=True, cwd=out,
                                    stdout=stdout, stderr=stderr, timeout=timeout)
        code = result.returncode
        launched = True
        if code:
            raise RuntimeError(f'GPT-6 Codex annotation failed (exit {code}); see {out}')
        annotation = json.loads((out / 'response.json').read_text())
        status = 'completed'
        return annotation
    except Exception as exc:
        error_type = type(exc).__name__
        if isinstance(exc, subprocess.TimeoutExpired):
            launched = True
        raise
    finally:
        record = {
            'provider': 'codex_cli_existing_login', 'model_requested': model,
            'model_identity_source': 'explicit CLI --model argument; no implicit fallback',
            'external_model_call': (True if code == 0 and (out / 'response.json').is_file()
                                    else None if launched else False),
            'process_started': launched, 'error_type': error_type,
            'command': cmd, 'started_at': started,
            'ended_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'wall_s': time.perf_counter() - tick, 'status': status, 'exit_code': code,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'artifacts_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in out.iterdir() if p.is_file()},
        }
        (out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
