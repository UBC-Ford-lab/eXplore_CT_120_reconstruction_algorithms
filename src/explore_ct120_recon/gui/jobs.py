"""The job record and its store.

A job is one driver invocation. The record holds everything needed to say
afterwards what ran, where, from which settings, and what came out: the
command line, the machine and GPU, the log, the result path, the W&B run.
Stored in SQLite so the record survives a restart of the server and can be
read by other tools.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

STATUSES = ('queued', 'pending', 'running', 'finished', 'failed', 'cancelled')
#: pending = handed to a scheduler, not running yet
ACTIVE = ('queued', 'pending', 'running')


@dataclass
class Job:
    id: str
    driver: str
    algorithm: Optional[str]
    values: dict                      # the submitted form values
    command: list                     # the exact command line
    executor: str
    status: str = 'queued'
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    note: str = ''
    input: str = ''                   # the primary positional (scan or volume)
    cwd: str = ''
    hostname: Optional[str] = None
    gpu_index: Optional[int] = None
    gpu_name: Optional[str] = None
    pid: Optional[int] = None
    returncode: Optional[int] = None
    log_path: Optional[str] = None
    output_path: Optional[str] = None
    wandb_url: Optional[str] = None
    package_version: Optional[str] = None
    error: Optional[str] = None
    remote_id: Optional[str] = None      # remote pid or Slurm job id
    remote_state: Optional[str] = None   # the scheduler's word for it
    node: Optional[str] = None           # allocated node
    resources: str = ''                  # extra sbatch arguments for this job

    @staticmethod
    def new(driver, algorithm, values, command, executor, *, note='', input='', cwd='', resources='') -> 'Job':
        return Job(id=uuid.uuid4().hex[:12], driver=driver, algorithm=algorithm,
                   values=values, command=command, executor=executor,
                   note=note, input=input, cwd=cwd, resources=resources)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['duration'] = ((self.finished or time.time()) - self.started) if self.started else None
        return d


_JSON_COLUMNS = ('values', 'command')
_COLUMNS = [f.name for f in fields(Job)]
_REAL = ('created', 'started', 'finished')
_INT = ('gpu_index', 'pid', 'returncode')


def _sql_type(column: str) -> str:
    return 'REAL' if column in _REAL else 'INTEGER' if column in _INT else 'TEXT'


class JobStore:
    """SQLite-backed store. One table, one row per job, JSON in two columns."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            cols = ', '.join(f'"{c}" {_sql_type(c)}' for c in _COLUMNS)
            con.execute(f'CREATE TABLE IF NOT EXISTS jobs ({cols}, PRIMARY KEY ("id"))')
            con.execute('CREATE INDEX IF NOT EXISTS jobs_created ON jobs (created)')
            present = {r[1] for r in con.execute('PRAGMA table_info(jobs)')}
            for c in _COLUMNS:
                if c not in present:
                    con.execute(f'ALTER TABLE jobs ADD COLUMN "{c}" {_sql_type(c)}')
            # A server that died mid-run leaves 'running' rows behind; they are
            # not running any more.
            # Running rows are re-attached by their executor's tick(): local
            # jobs run in their own session and leave an exit file, remote ones
            # are polled by id.

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _row_to_job(row) -> Job:
        d = dict(row)
        for c in _JSON_COLUMNS:
            d[c] = json.loads(d[c]) if d[c] else ([] if c == 'command' else {})
        return Job(**d)

    def add(self, job: Job) -> Job:
        d = asdict(job)
        for c in _JSON_COLUMNS:
            d[c] = json.dumps(d[c])
        with self._lock, self._connect() as con:
            names = ', '.join('"%s"' % c for c in _COLUMNS)
            marks = ', '.join('?' for _ in _COLUMNS)
            con.execute(f'INSERT INTO jobs ({names}) VALUES ({marks})', [d[c] for c in _COLUMNS])
        return job

    def update(self, job_id: str, **changes: Any) -> None:
        if not changes:
            return
        for c in _JSON_COLUMNS:
            if c in changes:
                changes[c] = json.dumps(changes[c])
        with self._lock, self._connect() as con:
            assignments = ', '.join('"%s"=?' % k for k in changes)
            con.execute(f'UPDATE jobs SET {assignments} WHERE id=?', [*changes.values(), job_id])

    def get(self, job_id: str) -> Optional[Job]:
        with self._connect() as con:
            row = con.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list(self, limit: int = 200, status: Optional[str] = None) -> list[Job]:
        q = 'SELECT * FROM jobs' + (' WHERE status=?' if status else '') + ' ORDER BY created DESC LIMIT ?'
        params = ((status,) if status else ()) + (limit,)
        with self._connect() as con:
            rows = con.execute(q, params).fetchall()
        return [self._row_to_job(r) for r in rows]

    def next_queued(self, executor: str) -> Optional[Job]:
        with self._connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE status='queued' AND executor=? ORDER BY created LIMIT 1",
                              (executor,)).fetchone()
        return self._row_to_job(row) if row else None

    def delete(self, job_id: str) -> None:
        with self._lock, self._connect() as con:
            con.execute('DELETE FROM jobs WHERE id=?', (job_id,))

    def running(self, executor: Optional[str] = None) -> list[Job]:
        q = "SELECT * FROM jobs WHERE status='running'" + (' AND executor=?' if executor else '')
        with self._connect() as con:
            rows = con.execute(q, (executor,) if executor else ()).fetchall()
        return [self._row_to_job(r) for r in rows]
