"""Immutable, hash-bound JSON revisions with a rebuildable SQLite catalog."""
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from .schema import digest, encoded, now, summary, validate_record
from .search_text import grams, searchable


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.write-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): sha.update(chunk)
    return sha.hexdigest()


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root/'index.sqlite3'
        with self.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, revision INTEGER, task TEXT, status TEXT, family TEXT, text TEXT)')
            con.execute('CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED, revision UNINDEXED, terms)')

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=60)
        con.row_factory = sqlite3.Row
        try:
            with con: yield con
        finally: con.close()

    @contextmanager
    def lock(self):
        with (self.root/'.write.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(stream, fcntl.LOCK_UN)

    def record_path(self, ident, revision):
        if not re.fullmatch('[a-f0-9]{64}', ident) or type(revision) is not int or revision < 1:
            raise ValueError('Invalid record identity')
        return self.root/'records'/ident/str(revision)/'record.json'

    def get(self, ident, revision=None):
        if revision is None:
            with self.connect() as con:
                row = con.execute('SELECT revision FROM records WHERE id=?', (ident,)).fetchone()
            if row is None: raise ValueError('Unknown record')
            revision = row['revision']
        path = self.record_path(ident, revision)
        r = json.loads(path.read_text())
        sha = r.pop('record_sha256', None)
        if digest(r) != sha: raise ValueError('Record integrity failure')
        validate_record(r)
        if r['attempt_id'] != ident or r['revision'] != revision: raise ValueError('Record identity mismatch')
        r['record_sha256'] = sha
        return r

    def _index(self, con, r):
        con.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)',
                    (r['attempt_id'], r['revision'], r['task_id'], r['status'],
                     r['relations']['family'], encoded(r).decode()))
        con.execute('DELETE FROM search WHERE id=?', (r['attempt_id'],))
        con.execute('INSERT INTO search VALUES (?,?,?)',
                    (r['attempt_id'], r['revision'], ' '.join(sorted(grams(searchable(r))))))

    def put(self, record):
        r = json.loads(encoded(record))
        for key in ('revision', 'record_sha256', 'recorded_at', 'summary'): r.pop(key, None)
        validate_record(r)
        with self.lock():
            folder = self.root/'records'/r['attempt_id']
            revisions = sorted(int(p.name) for p in folder.iterdir() if p.name.isdigit() and (p/'record.json').exists()) if folder.exists() else []
            latest = revisions[-1] if revisions else 0
            if latest:
                old = self.get(r['attempt_id'], latest)
                comparison = {k: v for k, v in old.items() if k not in ('revision', 'record_sha256', 'recorded_at', 'summary')}
                if comparison == r:
                    with self.connect() as con: self._index(con, old)
                    return {'attempt_id': r['attempt_id'], 'revision': latest}
            r.update(revision=latest+1, recorded_at=now(), summary=summary(r))
            r['record_sha256'] = digest(r)
            path = self.record_path(r['attempt_id'], latest+1)
            atomic(path, r)
            (path.parent/'summary.md').write_text(r['summary'], encoding='utf-8')
            atomic(path.parent/'evidence.json', r['evidence'])
            with self.connect() as con: self._index(con, r)
            return {'attempt_id': r['attempt_id'], 'revision': latest+1}

    def rebuild(self):
        with self.lock(), self.connect() as con:
            records = []
            for path in sorted((self.root/'records').glob('*/*/record.json')):
                records.append(self.get(path.parent.parent.name, int(path.parent.name)))
            con.execute('DELETE FROM records')
            con.execute('DELETE FROM search')
            for r in sorted(records, key=lambda r: r['revision']): self._index(con, r)
        return self.stats()

    def stats(self):
        with self.connect() as con:
            rows = [dict(r) for r in con.execute('SELECT task,status,count(*) AS count FROM records GROUP BY task,status')]
        return {'attempts': sum(r['count'] for r in rows), 'by_task_status': rows}

    def snapshot(self, name, *, excluded_families=()):
        if not re.fullmatch('[a-zA-Z0-9_-]+', name): raise ValueError('Invalid snapshot name')
        path = self.root/'snapshots'/(name+'.json')
        with self.lock():
            if path.exists(): raise FileExistsError(path)
            with self.connect() as con:
                rows = list(con.execute('SELECT id,revision FROM records ORDER BY id'))
            bound = {}; records = []
            for row in rows:
                records.append(self.get(row['id'], row['revision']))
            parents = {}
            def find(x):
                parents.setdefault(x, x)
                if parents[x] != x: parents[x] = find(parents[x])
                return parents[x]
            def union(a, b):
                a, b = find(a), find(b)
                parents[max(a, b)] = min(a, b)
            for r in records:
                ident = 'id:'+r['attempt_id']; rel = r['relations']
                union(ident, 'family:'+rel['family'])
                if rel.get('parent'): union(ident, 'id:'+rel['parent'])
                for key in ('shared_plan', 'source_demonstration'):
                    if rel.get(key): union(ident, key+':'+str(rel[key]))
            groups = {r['attempt_id']: find('id:'+r['attempt_id']) for r in records}
            aliases = {r['relations']['family']: groups[r['attempt_id']] for r in records}
            exclusions = {aliases.get(f, f) for f in excluded_families}
            for r in records:
                if groups[r['attempt_id']] not in exclusions:
                    bound[r['attempt_id']] = {'revision': r['revision'], 'sha256': r['record_sha256']}
            data = {'schema_version': 1, 'created_at': now(), 'records': bound,
                    'excluded_families': list(excluded_families), 'retriever_version': 1,
                    'groups': groups, 'family_aliases': aliases}
            data['sha256'] = digest(data); atomic(path, data)
        return name

    def load_snapshot(self, name):
        if not re.fullmatch('[a-zA-Z0-9_-]+', name): raise ValueError('Invalid snapshot name')
        data = json.loads((self.root/'snapshots'/(name+'.json')).read_text())
        sha = data.pop('sha256')
        if digest(data) != sha: raise ValueError('Snapshot integrity failure')
        data['sha256'] = sha
        return data
