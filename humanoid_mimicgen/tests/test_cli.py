import json
import subprocess
import sys
from pathlib import Path

import pytest

from humanoid_mimicgen.cli import create_experiment, ensure_output_family
from humanoid_mimicgen.model import annotate_codex


def test_experiment_creation_never_overwrites_existing_directory(tmp_path):
    first = create_experiment(tmp_path, 'test')
    (first / 'keep').write_text('original')
    second = create_experiment(tmp_path, 'test')
    assert first != second
    assert (first / 'keep').read_text() == 'original'
    record = json.loads((second / 'creation.json').read_text())
    assert record['created_at'].endswith('+08:00')
    assert record['experiment_id'] == second.name


def test_codex_rejects_non_gpt6_model_before_invocation(tmp_path):
    with pytest.raises(ValueError, match='GPT-6'):
        annotate_codex({}, 'different-model', tmp_path / 'call')
    assert not (tmp_path / 'call').exists()


def test_help_does_not_initialize_simulator():
    result = subprocess.run(
        [sys.executable, '-m', 'humanoid_mimicgen', '--help'],
        cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'generate' in result.stdout
    assert 'annotate' in result.stdout
    assert 'smoke' in result.stdout


def test_stale_sidecar_blocks_before_creating_new_primary(tmp_path):
    primary = tmp_path / 'annotation.json'
    sidecar = tmp_path / 'annotation_provenance.json'
    sidecar.write_text('{"old":true}')
    with pytest.raises(FileExistsError):
        ensure_output_family(primary, [sidecar])
    assert not primary.exists()


def test_missing_codex_records_failed_launch_not_completed_call(tmp_path, monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError('test executable missing')
    monkeypatch.setattr(subprocess, 'Popen', absent)
    with pytest.raises(FileNotFoundError):
        annotate_codex({}, 'gpt-6-astra', tmp_path / 'call')
    provenance = json.loads((tmp_path / 'call' / 'provenance.json').read_text())
    assert provenance['external_model_call'] is False
    assert provenance['status'] == 'error'
    assert provenance['error_type'] == 'FileNotFoundError'
