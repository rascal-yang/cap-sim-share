"""Transactional job and tool-call records, independent of simulation imports."""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3

from ..workflow.contracts import WorkflowError, canonical, digest, now
from .contracts import require

TRANSITIONS = {
    'queued': {'running', 'failed'},
    'running': {'completed', 'failed', 'interrupted'},
    'interrupted': {'queued', 'completed', 'failed'},
    'completed': set(), 'failed': set(),
}


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'jobs.sqlite3'
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
                    content_hash TEXT NOT NULL, job TEXT NOT NULL, input_hashes TEXT NOT NULL,
                    status TEXT NOT NULL, owner TEXT, details TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cases (
                    job_id TEXT NOT NULL REFERENCES jobs(id), case_id TEXT NOT NULL,
                    stage TEXT NOT NULL, details TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, case_id));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    case_id TEXT, event TEXT NOT NULL, details TEXT NOT NULL, at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                    session_id TEXT NOT NULL, call_id TEXT NOT NULL, arguments_hash TEXT NOT NULL,
                    result TEXT NOT NULL, at TEXT NOT NULL, PRIMARY KEY(session_id, call_id));
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def submit(self, job, input_hashes):
        key = job.get('request_id')
        require(isinstance(key, str) and re.fullmatch('[A-Za-z0-9_-]{1,128}', key), 'Request ID')
        signature = digest({'job': job, 'input_hashes': input_hashes})
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE request_id=?', (key,)).fetchone()
            if row:
                require(row['content_hash'] == signature, 'Request ID content changed', 'conflict')
                return {'job_id': row['id'], 'status': row['status'], 'reused': True}
            ident = signature
            stamp = now()
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)',
                (ident, key, signature, canonical(job).decode(), canonical(input_hashes).decode(),
                 'queued', None, '{}', stamp, stamp))
            db.execute('INSERT INTO events(job_id,event,details,at) VALUES (?,?,?,?)', (ident, 'queued', '{}', stamp))
            return {'job_id': ident, 'status': 'queued', 'reused': False}

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            require(row is not None, 'Unknown job', 'not_found')
            result = dict(row)
            for key in ('job', 'input_hashes', 'details', 'owner'):
                result[key] = json.loads(result[key]) if result[key] is not None else None
            result['job_id'] = result.pop('id')
            result['cases'] = {r['case_id']: {'stage': r['stage'], 'details': json.loads(r['details']),
                                            'updated_at': r['updated_at']}
                               for r in db.execute('SELECT * FROM cases WHERE job_id=? ORDER BY case_id', (job_id,))}
            return result

    def claim(self, job_id, owner):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            stamp = now()
            cursor = db.execute("UPDATE jobs SET status='running',owner=?,updated_at=? WHERE id=? AND status='queued'",
                                (canonical(owner).decode(), stamp, job_id))
            if cursor.rowcount:
                db.execute('INSERT INTO events(job_id,event,details,at) VALUES (?,?,?,?)',
                           (job_id, 'running', canonical(owner).decode(), stamp))
            return bool(cursor.rowcount)

    def transition(self, job_id, expected, next_status, details):
        require(next_status in TRANSITIONS.get(expected, set()), 'Illegal job transition', 'conflict')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status,details FROM jobs WHERE id=?', (job_id,)).fetchone()
            require(row is not None and row['status'] == expected, 'Job state changed', 'conflict')
            merged = json.loads(row['details']); merged.update(details)
            stamp = now()
            db.execute('UPDATE jobs SET status=?,details=?,updated_at=? WHERE id=?',
                       (next_status, canonical(merged).decode(), stamp, job_id))
            db.execute('INSERT INTO events(job_id,event,details,at) VALUES (?,?,?,?)',
                       (job_id, next_status, canonical(details).decode(), stamp))

    def record_case(self, job_id, case_id, stage, details):
        require(isinstance(case_id, str) and re.fullmatch('[A-Za-z0-9_-]{1,128}', case_id), 'Case ID')
        canonical(details)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT details FROM cases WHERE job_id=? AND case_id=?', (job_id, case_id)).fetchone()
            merged = json.loads(row['details']) if row else {}
            merged.update(details); stamp = now()
            db.execute('INSERT INTO cases VALUES (?,?,?,?,?) ON CONFLICT(job_id,case_id) DO UPDATE SET stage=excluded.stage,details=excluded.details,updated_at=excluded.updated_at',
                       (job_id, case_id, stage, canonical(merged).decode(), stamp))
            db.execute('INSERT INTO events(job_id,case_id,event,details,at) VALUES (?,?,?,?,?)',
                       (job_id, case_id, stage, canonical(details).decode(), stamp))

    def lookup_call(self, session_id, call_id, arguments_hash):
        with self.connect() as db:
            row = db.execute('SELECT * FROM calls WHERE session_id=? AND call_id=?', (session_id, call_id)).fetchone()
            if row is None:
                return None
            require(row['arguments_hash'] == arguments_hash, 'Tool call arguments changed', 'conflict')
            return json.loads(row['result'])

    def record_call(self, session_id, call_id, arguments_hash, result):
        encoded = canonical(result).decode()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM calls WHERE session_id=? AND call_id=?', (session_id, call_id)).fetchone()
            if row:
                require(row['arguments_hash'] == arguments_hash and row['result'] == encoded,
                        'Tool call result changed', 'conflict')
            else:
                db.execute('INSERT INTO calls VALUES (?,?,?,?,?)', (session_id, call_id, arguments_hash, encoded, now()))
