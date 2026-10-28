"""Calendar directories for new batches; non-moving views of frozen history."""
from datetime import datetime
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo
from ..cli import write_new

ZONE = ZoneInfo('Asia/Shanghai')


def create_record(root, description, clock=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_]*', description):
        raise ValueError('Invalid experiment description')
    clock = clock or datetime.now(ZONE)
    if clock.tzinfo is None:
        raise ValueError('Creation time must include timezone')
    clock = clock.astimezone(ZONE)
    day = Path(root)/clock.strftime('%Y-%m')/clock.strftime('%d')
    day.mkdir(parents=True, exist_ok=True)
    name = clock.strftime('%Y%m%d_%H%M_')+description
    sequence = 0
    while True:
        path = day/(name if sequence == 0 else f'{name}_{sequence:02d}')
        try:
            path.mkdir()
            break
        except FileExistsError:
            sequence += 1
    write_new(path/'creation.json', {'experiment_id': path.name,
        'created_at': clock.isoformat(timespec='seconds'), 'timezone': 'Asia/Shanghai',
        'layout': 'YYYY-MM/DD/experiment_id'})
    return path


def index_history(root):
    root = Path(root).absolute()
    entries = []
    unclassified = []
    # Inspect the flat root only: never recurse into or rewrite frozen experiments.
    for source in sorted(root.iterdir()):
        if not source.is_dir() or source.is_symlink() or re.fullmatch(r'\d{4}-\d{2}', source.name):
            continue
        dates = re.findall(r'(?<!\d)(20\d{6})(?!\d)', source.name)
        if len(dates) != 1:
            unclassified.append(source.name)
            continue
        try:
            date = datetime.strptime(dates[0], '%Y%m%d')
        except ValueError:
            unclassified.append(source.name)
            continue
        alias = root/date.strftime('%Y-%m')/date.strftime('%d')/source.name
        # Validate the whole plan before creating links. No moves or overwrites.
        if os.path.lexists(alias) and not (alias.is_symlink() and alias.resolve() == source.resolve()):
            raise FileExistsError(alias)
        entries.append((source, alias, date.strftime('%Y-%m-%d')))
    created = 0
    for source, alias, _ in entries:
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(alias):
            alias.symlink_to(os.path.relpath(source, alias.parent), target_is_directory=True)
            created += 1
    return {'schema_version': 1, 'mode': 'nonmoving_calendar_view',
        'indexed_at': datetime.now(ZONE).isoformat(timespec='seconds'),
        'created_links': created, 'indexed_count': len(entries), 'unclassified': unclassified,
        'entries': [{'experiment_id': s.name, 'original_path': str(s), 'calendar_path': str(a),
                     'date': d, 'date_source': 'original_directory_name'} for s,a,d in entries]}
