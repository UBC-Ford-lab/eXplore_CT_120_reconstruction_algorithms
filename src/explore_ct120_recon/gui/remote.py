"""Machines reached over SSH: a plain remote shell, or a Slurm cluster.

Both executors talk to the machine through one multiplexed SSH connection
(``ControlMaster``). The master is opened once, interactively if the host
asks for a second factor, and every poll afterwards reuses it. Nothing is
installed on the remote side: a job is the package's own console script,
started with ``nohup`` (ssh) or ``sbatch`` (slurm), and the log is pulled
back in increments into the local log file so the page shows it live.

Remote state for a job: ``remote_id`` (the pid of the process group, or the
Slurm job id), ``remote_state`` (Slurm's word for it), ``node``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .executors import Executor
from .jobs import Job, JobStore
from .machines import Machine

_PROBE = r'''
import json, os, shutil, subprocess, sys, platform
out = {"hostname": platform.node(), "platform": platform.platform(), "cpus": os.cpu_count(), "python": sys.version.split()[0]}
try:
    with open("/proc/meminfo") as f:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in f if l.startswith(("MemTotal", "MemAvailable"))}
    out["memory"] = {"total_gib": mem.get("MemTotal", 0) / 2**20, "available_gib": mem.get("MemAvailable", 0) / 2**20}
except Exception:
    out["memory"] = {"total_gib": None, "available_gib": None}
gpus = []
if shutil.which("nvidia-smi"):
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        for line in q.stdout.strip().splitlines():
            i, n, t, u, g = [p.strip() for p in line.split(",")]
            gpus.append({"index": int(i), "name": n, "total_gib": float(t) / 1024, "free_gib": (float(t) - float(u)) / 1024, "used_gib": float(u) / 1024, "utilisation": float(g)})
    except Exception:
        pass
out["gpus"] = gpus
try:
    import explore_ct120_recon as r
    out["package"] = {"version": r.__version__, "commit": None, "python": out["python"], "path": os.path.dirname(r.__file__)}
except Exception as e:
    out["package"] = {"version": None, "commit": None, "python": out["python"], "error": "%s: %s" % (type(e).__name__, e)}
out["slurm"] = bool(shutil.which("sbatch"))
if out["slurm"]:
    try:
        # one line per job of this user: id|name|state|elapsed|limit|nodes|cpus|memory|gres|node or reason|partition
        q = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j|%T|%M|%l|%D|%C|%m|%b|%R|%P"], capture_output=True, text=True, timeout=15)
        keys = ["id", "name", "state", "elapsed", "limit", "nodes", "cpus", "memory", "gres", "where", "partition"]
        jobs = [dict(zip(keys, line.split("|", len(keys) - 1))) for line in q.stdout.splitlines() if line.count("|") >= len(keys) - 1]
        states = [j["state"] for j in jobs]
        out["queue"] = {"running": states.count("RUNNING"), "pending": states.count("PENDING"), "jobs": jobs}
    except Exception:
        out["queue"] = None
print(json.dumps(out))
'''


def control_dir() -> Path:
    """Where the multiplexing sockets live: short (a Unix socket path is limited
    to about 100 characters) and private to the user."""
    base = os.environ.get('XDG_RUNTIME_DIR') or f'/tmp/ct120-gui-{os.getuid()}'
    d = Path(base) / 'ct120-gui'
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def socket_for(machine_name: str) -> Path:
    tag = hashlib.sha1(machine_name.encode()).hexdigest()[:8]
    return control_dir() / f'{machine_name[:20]}-{tag}.sock'


class LoggedOut(RuntimeError):
    """Raised by a connection that was logged out on purpose."""


class SshConnection:
    """One multiplexed SSH connection to ``host``.

    ``status()`` is ``connected`` (the master is up), ``needs_login`` (no
    master and a non-interactive attempt failed, so the host wants a password
    or a second factor) or ``unreachable``.
    """

    def __init__(self, host: str, socket_path: Path, *, persist: str = '12h'):
        self.host = host
        self.socket = Path(socket_path)
        self.socket.parent.mkdir(parents=True, exist_ok=True)
        self.persist = persist
        self.last_error: Optional[str] = None
        self.closed = False      # logged out on purpose: no command may re-open the master

    def _base(self, batch: bool = True) -> list[str]:
        opts = ['-o', f'ControlPath={self.socket}', '-o', 'ControlMaster=auto',
                '-o', f'ControlPersist={self.persist}', '-o', 'ConnectTimeout=15',
                '-o', 'ServerAliveInterval=30']
        if batch:
            opts += ['-o', 'BatchMode=yes']
        return ['ssh', *opts, self.host]

    def login_command(self) -> str:
        """What to run in a terminal when the host asks for a second factor."""
        return shlex.join(['ssh', '-fNM', '-o', f'ControlPath={self.socket}',
                           '-o', f'ControlPersist={self.persist}', self.host])

    def master_alive(self) -> bool:
        if not self.socket.exists():
            return False
        r = subprocess.run(['ssh', '-O', 'check', '-o', f'ControlPath={self.socket}', self.host],
                           capture_output=True, text=True, timeout=15)
        return r.returncode == 0

    def close(self) -> None:
        """Log out: stop the multiplexed master, and stay logged out until
        ``reopen`` (the Connect button or a login) is called."""
        self.closed = True
        if self.socket.exists():
            subprocess.run(['ssh', '-O', 'exit', '-o', f'ControlPath={self.socket}', self.host],
                           capture_output=True, text=True, timeout=15)
        try:
            self.socket.unlink()
        except OSError:
            pass

    def reopen(self) -> None:
        self.closed = False

    def status(self) -> str:
        if self.master_alive():
            self.closed = False      # a login (ssh -fNM from the terminal) counts as reopening
            return 'connected'
        if self.closed:
            return 'logged_out'
        try:
            r = self.run('true', timeout=30)
        except subprocess.TimeoutExpired:
            self.last_error = 'timed out'
            return 'unreachable'
        if r.returncode != 255:
            self.last_error = None
            return 'connected'        # ssh itself only exits 255; anything else ran remotely
        self.last_error = (r.stderr or '').strip().splitlines()[-1][:300] if (r.stderr or '').strip() else f'ssh exit {r.returncode}'
        err = (r.stderr or '').lower()
        if any(k in err for k in ('permission denied', 'interactive', 'authentication', 'host key',
                                  'verification failed', 'passcode', 'password')):
            return 'needs_login'
        return 'unreachable'

    def run(self, command: str, *, timeout: float = 120, input: Optional[str] = None) -> subprocess.CompletedProcess:
        if self.closed:
            raise LoggedOut(f'{self.host}: logged out')
        return subprocess.run(self._base() + [command], capture_output=True, text=True,
                              timeout=timeout, input=input)

    def check(self, command: str, **kw) -> str:
        """``run`` that raises on a non-zero exit, returning stdout."""
        r = self.run(command, **kw)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or f'exit {r.returncode}').strip()[:800])
        return r.stdout

    def write_file(self, path: str, content: str) -> None:
        self.check(f'mkdir -p {shlex.quote(str(Path(path).parent))} && cat > {shlex.quote(path)}', input=content)

    def read_from(self, path: str, offset: int) -> bytes:
        if self.closed:
            raise LoggedOut(f'{self.host}: logged out')
        r = subprocess.run(self._base() + [f'tail -c +{offset + 1} {shlex.quote(path)} 2>/dev/null'],
                           capture_output=True, timeout=60)
        return r.stdout if r.returncode == 0 else b''

    def probe(self, python: str) -> dict:
        out = self.check(f'{shlex.quote(python)} - <<\'PY\'\n{_PROBE}\nPY', timeout=40)
        return json.loads(out.strip().splitlines()[-1])


def make_connection(machine: Machine, control_dir: Optional[Path] = None) -> SshConnection:
    """Factory, replaced in tests by a connection that runs commands locally.

    ``control_dir`` is accepted for the tests; real sockets always go to the
    short runtime directory (see ``control_dir()``)."""
    return SshConnection(machine.host, socket_for(machine.name))


class RemoteExecutor(Executor):
    """Runs drivers on a machine reached over SSH, with ``nohup``."""

    kind = 'ssh'

    def __init__(self, machine: Machine, control_dir: Path, log_dir: Path, env: Optional[dict] = None):
        self.machine = machine
        self.name = machine.name
        self.python = machine.python
        self.conn = make_connection(machine, control_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env = {**(env or {}), **machine.env}
        self.slots = machine.slots
        self._cache: tuple[float, dict] = (0.0, {})
        self._lock = threading.Lock()
        self._watching: set[str] = set()

    # -- what the machine is --------------------------------------------------
    def describe(self, fresh: bool = False) -> dict:
        base = {'name': self.name, 'kind': self.kind, 'host': self.machine.host,
                'cwd': self.machine.workdir, 'slots': self.slots, 'running': [],
                'hostname': self.machine.host, 'platform': '', 'cpus': None,
                'memory': {'total_gib': None, 'available_gib': None}, 'gpus': [],
                'package': {'version': None, 'commit': None, 'python': None},
                'login_command': self.conn.login_command()}
        try:
            status = self.conn.status()
        except Exception as e:
            return {**base, 'status': 'unreachable', 'error': str(e)}
        base['status'] = status
        if status != 'connected':
            if self.conn.last_error:
                base['error'] = self.conn.last_error
            return base
        now = time.time()
        with self._lock:
            age, cached = now - self._cache[0], self._cache[1]
        if cached and age < 60 and not fresh:
            return {**base, **cached}
        try:
            info = self.conn.probe(self.machine.python)
        except Exception as e:
            return {**base, 'status': 'connected', 'error': f'probe failed: {e}'}
        with self._lock:
            self._cache = (time.time(), info)
        return {**base, **info}

    def health(self) -> list[dict]:
        """Checks a user runs after adding the machine; each is ok/fail + text."""
        checks = []
        status = self.conn.status()
        checks.append({'name': 'ssh', 'ok': status == 'connected',
                       'detail': status if status == 'connected' else
                       f'{status}; run in a terminal: {self.conn.login_command()}'})
        if status != 'connected':
            return checks
        for label, cmd in (('workdir', f'test -d {shlex.quote(self.machine.workdir)}'),
                           ('scans_root', f'test -d {shlex.quote(self.machine.scans_root)}' if self.machine.scans_root else 'true'),
                           ('python', f'{shlex.quote(self.machine.python)} --version')):
            r = self.conn.run(f'cd ~ && {cmd}', timeout=60)
            checks.append({'name': label, 'ok': r.returncode == 0,
                           'detail': (r.stdout or r.stderr).strip() or ('ok' if r.returncode == 0 else 'not found')})
        try:
            info = self.conn.probe(self.machine.python)
            pkg = info.get('package') or {}
            checks.append({'name': 'package', 'ok': bool(pkg.get('version')),
                           'detail': f"explore_ct120_recon {pkg.get('version')} at {pkg.get('path')}" if pkg.get('version') else pkg.get('error', 'not importable')})
            checks.append({'name': 'gpus', 'ok': True, 'detail': ', '.join(g['name'] for g in info.get('gpus', [])) or 'none visible here'})
            if self.kind == 'slurm':
                checks.append({'name': 'slurm', 'ok': bool(info.get('slurm')), 'detail': 'sbatch found' if info.get('slurm') else 'sbatch not on PATH'})
        except Exception as e:
            checks.append({'name': 'package', 'ok': False, 'detail': str(e)})
        return checks

    def list_scans(self) -> list[dict]:
        if not self.machine.scans_root:
            return []
        root = self.machine.scans_root
        r = self.conn.run(f'cd {shlex.quote(root)} 2>/dev/null && ls -d */scan.xml 2>/dev/null | sed "s#/scan.xml##"', timeout=60)
        names = [n for n in r.stdout.split() if n]
        return [{'name': n, 'path': f'{root.rstrip("/")}/{n}'} for n in sorted(names)]

    # -- running ---------------------------------------------------------------
    def _remote_dir(self) -> str:
        return f'{self.machine.workdir.rstrip("/")}/.ct120-gui'

    def _env_exports(self, job: Job) -> str:
        env = {**self.env, 'PYTHONUNBUFFERED': '1'}
        return '\n'.join(f'export {k}={shlex.quote(v)}' for k, v in env.items())

    def start(self, job: Job, store: JobStore) -> None:
        remote_log = f'{self._remote_dir()}/logs/{job.id}.log'
        exit_file = f'{self._remote_dir()}/logs/{job.id}.exit'
        script = (f'#!/bin/bash\ncd {shlex.quote(self.machine.workdir)}\n{self._env_exports(job)}\n'
                  f'{shlex.join(job.command)}\necho $? > {shlex.quote(exit_file)}\n')
        script_path = f'{self._remote_dir()}/jobs/{job.id}.sh'
        local_log = self.log_dir / f'{job.id}.log'
        try:
            self.conn.check(f'mkdir -p {shlex.quote(self._remote_dir())}/logs {shlex.quote(self._remote_dir())}/jobs')
            self.conn.write_file(script_path, script)
            pid = self.conn.check(
                f'chmod +x {shlex.quote(script_path)} && nohup setsid {shlex.quote(script_path)} '
                f'> {shlex.quote(remote_log)} 2>&1 < /dev/null & echo $!').strip()
            int(pid)
        except Exception as e:
            local_log.write_text(f'could not start on {self.machine.host}: {e}\n')
            store.update(job.id, status='failed', error=str(e)[:500], finished=time.time(), log_path=str(local_log))
            return
        store.update(job.id, status='running', started=time.time(), remote_id=pid,
                     hostname=self.machine.host, log_path=str(local_log), cwd=self.machine.workdir)
        self._watch_in_thread(job.id, store, remote_log, exit_file)

    def _watch_in_thread(self, job_id, store, remote_log, exit_file):
        with self._lock:
            if job_id in self._watching:
                return
            self._watching.add(job_id)
        threading.Thread(target=self._watch, args=(job_id, store, remote_log, exit_file),
                         daemon=True, name=f'remote-{job_id}').start()

    def _pull_log(self, job: Job, remote_log: str, local_log: Path, state: dict) -> None:
        chunk = self.conn.read_from(remote_log, state['offset'])
        if chunk:
            with open(local_log, 'ab') as f:
                f.write(chunk)
            state['offset'] += len(chunk)
            self._scan_for_links(job, local_log, store=state['store'])

    @staticmethod
    def _scan_for_links(job: Job, local_log: Path, store: JobStore) -> None:
        from .executors import LocalExecutor
        seen = {'wandb_url': job.wandb_url, 'output_path': job.output_path,
                'output_logged': False}
        found = LocalExecutor._scan_log(local_log, seen)
        if found:
            store.update(job.id, **found)
            for k, v in found.items():
                setattr(job, k, v)

    def _poll(self, job: Job) -> tuple[bool, Optional[int], dict]:
        """(finished?, exit code, extra record fields)"""
        exit_file = f'{self._remote_dir()}/logs/{job.id}.exit'
        r = self.conn.run(f'cat {shlex.quote(exit_file)} 2>/dev/null || (kill -0 {job.remote_id} 2>/dev/null && echo RUNNING)', timeout=60)
        out = r.stdout.strip()
        if out == 'RUNNING':
            return False, None, {}
        if out.lstrip('-').isdigit():
            return True, int(out), {}
        if r.returncode != 0:
            return False, None, {}       # connection hiccup; try again later
        return True, None, {'error': 'process gone without an exit record'}

    def _preamble(self, job: Job) -> str:
        """What ``start`` wrote to the local log before any remote output."""
        return ''

    def _watch(self, job_id: str, store: JobStore, remote_log: str, exit_file: str) -> None:
        job = store.get(job_id)
        local_log = Path(job.log_path)
        # the local log = preamble + what was pulled so far; the remote offset excludes the preamble
        pulled = max(0, (local_log.stat().st_size if local_log.exists() else 0) - len(self._preamble(job).encode()))
        state = {'offset': pulled, 'store': store}
        try:
            while True:
                try:
                    self._pull_log(job, remote_log, local_log, state)
                    done, rc, extra = self._poll(job)
                except subprocess.TimeoutExpired:
                    done, rc, extra = False, None, {}
                except LoggedOut:
                    return      # ``tick`` re-attaches the watcher after the next login
                if extra:
                    store.update(job_id, **{k: v for k, v in extra.items() if k != 'error'})
                job = store.get(job_id)
                if job.status == 'cancelled' and done:
                    store.update(job_id, finished=time.time(), returncode=rc)
                    return
                if done:
                    try:
                        self._pull_log(job, remote_log, local_log, state)
                    except Exception:
                        pass
                    status = 'finished' if rc == 0 else 'failed'
                    from .executors import LocalExecutor
                    error = extra.get('error') or (None if rc == 0 else LocalExecutor._last_error(local_log))
                    store.update(job_id, status=status, returncode=rc, finished=time.time(), error=error)
                    return
                time.sleep(5.0)
        finally:
            with self._lock:
                self._watching.discard(job_id)

    def cancel(self, job: Job, store: JobStore) -> bool:
        if job.status == 'queued':
            store.update(job.id, status='cancelled', finished=time.time())
            return True
        if not job.remote_id:
            return False
        store.update(job.id, status='cancelled')
        self.conn.run(f'kill -TERM -- -{job.remote_id} 2>/dev/null; sleep 1; kill -TERM {job.remote_id} 2>/dev/null; true', timeout=60)
        return True

    def tick(self, store: JobStore) -> None:
        if self.conn.closed:
            return
        # Re-attach watchers after a server restart.
        for j in store.running(self.name):
            if j.remote_id:
                self._watch_in_thread(j.id, store, f'{self._remote_dir()}/logs/{j.id}.log',
                                      f'{self._remote_dir()}/logs/{j.id}.exit')
        active = [j for j in store.running(self.name)] + [j for j in store.list(status='pending') if j.executor == self.name]
        while len(active) < self.slots:
            job = store.next_queued(self.name)
            if job is None:
                return
            try:
                self.start(job, store)
            except Exception as e:
                store.update(job.id, status='failed', error=str(e)[:500], finished=time.time())
            active.append(job)


class SlurmExecutor(RemoteExecutor):
    """Submits drivers with ``sbatch`` from the machine's template."""

    kind = 'slurm'

    def __init__(self, machine: Machine, control_dir: Path, log_dir: Path, env: Optional[dict] = None):
        super().__init__(machine, control_dir, log_dir, env)
        self.slots = 50   # the scheduler queues; the GUI does not need to

    def render(self, job: Job) -> str:
        remote_log = f'{self._remote_dir()}/logs/{job.id}.log'
        fields = {'job_name': f'ct120_{job.driver}_{Path(job.input).name or job.id}'[:60],
                  'job_id': job.id, 'log': remote_log, 'workdir': self.machine.workdir,
                  'env': self._env_exports(job), 'command': shlex.join(job.command),
                  'python': self.machine.python, 'input': job.input}
        try:
            return self.machine.template_text().format_map(fields)
        except KeyError as e:
            raise ValueError(f'the Slurm template uses an unknown placeholder {{{e.args[0]}}}; '
                             f'known: {", ".join(sorted(fields))}')

    def directives(self, job: Job) -> list[dict]:
        """The ``#SBATCH`` options the job is submitted with: the template's,
        overridden by the per-job ``resources`` (sbatch's command line wins).
        Each: ``{'key', 'value', 'source': 'template' | 'override'}``."""
        found: list[dict] = []

        def put(key: str, value: Optional[str], source: str) -> None:
            key = key.lstrip('-')
            for d in found:
                if d['key'] == key:
                    d.update(value=value, source=source)
                    return
            found.append({'key': key, 'value': value, 'source': source})

        for line in self.render(job).splitlines():
            if not line.startswith('#SBATCH'):
                continue
            for tok in shlex.split(line[len('#SBATCH'):]):
                if tok.startswith('-'):
                    key, _, value = tok.partition('=')
                    put(key, value or None, 'template')
                elif found and found[-1]['value'] is None:
                    found[-1]['value'] = tok
        extra = shlex.split(job.resources or '')
        i = 0
        while i < len(extra):
            tok = extra[i]
            if tok.startswith('-'):
                key, has_eq, value = tok.partition('=')
                if not has_eq and i + 1 < len(extra) and not extra[i + 1].startswith('-'):
                    value = extra[i + 1]
                    i += 1
                put(key, value or None, 'override')
            i += 1
        return found

    def start(self, job: Job, store: JobStore) -> None:
        local_log = self.log_dir / f'{job.id}.log'
        script_path = f'{self._remote_dir()}/jobs/{job.id}.sbatch'
        try:
            script = self.render(job)
            self.conn.check(f'mkdir -p {shlex.quote(self._remote_dir())}/logs {shlex.quote(self._remote_dir())}/jobs')
            self.conn.write_file(script_path, script)
            extra = shlex.split(job.resources or '')
            out = self.conn.check(f'cd {shlex.quote(self.machine.workdir)} && sbatch --parsable {shlex.join(extra)} {shlex.quote(script_path)}', timeout=120)
            slurm_id = out.strip().split(';')[0].splitlines()[-1]
            int(slurm_id)
        except Exception as e:
            local_log.write_text(f'could not submit on {self.machine.host}: {e}\n')
            store.update(job.id, status='failed', error=str(e)[:500], finished=time.time(), log_path=str(local_log))
            return
        local_log.write_text(self._submitted_line(slurm_id))
        store.update(job.id, status='pending', started=time.time(), remote_id=slurm_id, remote_state='PENDING',
                     hostname=self.machine.host, log_path=str(local_log), cwd=self.machine.workdir)
        self._watch_in_thread(job.id, store, f'{self._remote_dir()}/logs/{job.id}.log', '')

    def _submitted_line(self, slurm_id: str) -> str:
        return f'submitted to Slurm on {self.machine.host} as job {slurm_id}\n'

    def _preamble(self, job: Job) -> str:
        return self._submitted_line(job.remote_id) if job.remote_id else ''

    def _poll(self, job: Job) -> tuple[bool, Optional[int], dict]:
        r = self.conn.run(f'squeue -h -j {job.remote_id} -o "%T|%N|%r" 2>/dev/null', timeout=60)
        line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ''
        if line:
            state, node, reason = (line.split('|') + ['', ''])[:3]
            extra = {'remote_state': state, 'node': node or None}
            if state == 'RUNNING' and job.status == 'pending':
                extra['status'] = 'running'
                extra['started'] = time.time()
            elif state == 'PENDING':
                extra['remote_state'] = f'PENDING ({reason})' if reason and reason != 'None' else 'PENDING'
            return False, None, extra
        if r.returncode != 0:
            return False, None, {}
        a = self.conn.run(f'sacct -X -n -P -j {job.remote_id} -o State,ExitCode,NodeList 2>/dev/null', timeout=60)
        row = a.stdout.strip().splitlines()[0] if a.stdout.strip() else ''
        if not row:
            return False, None, {}
        state, exitcode, node = (row.split('|') + ['', ''])[:3]
        rc = int(exitcode.split(':')[0]) if exitcode and exitcode.split(':')[0].isdigit() else None
        state = state.split()[0]
        extra = {'remote_state': state, 'node': node or None}
        if state in ('COMPLETED',):
            return True, rc if rc is not None else 0, extra
        if state in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'DEADLINE', 'BOOT_FAIL'):
            extra['error'] = f'Slurm: {state}'
            return True, rc if rc not in (None, 0) else 1, extra
        return False, None, extra

    def cancel(self, job: Job, store: JobStore) -> bool:
        if job.status == 'queued':
            store.update(job.id, status='cancelled', finished=time.time())
            return True
        if not job.remote_id:
            return False
        store.update(job.id, status='cancelled')
        self.conn.run(f'scancel {job.remote_id}', timeout=60)
        return True

    def scancel(self, slurm_id: str) -> str:
        """``scancel`` a job of the user's queue by its Slurm id (any job, not
        only one submitted from here). Returns scancel's message, if any."""
        if not slurm_id.isdigit():
            raise ValueError(f'not a Slurm job id: {slurm_id!r}')
        r = self.conn.run(f'scancel {slurm_id}', timeout=60)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip() or f'scancel exited {r.returncode}')
        with self._lock:
            self._cache = (0.0, {})      # the next describe re-reads the queue
        return (r.stderr or r.stdout).strip()

    def tick(self, store: JobStore) -> None:
        if self.conn.closed:
            return
        for j in store.running(self.name) + [j for j in store.list(status='pending') if j.executor == self.name]:
            if j.remote_id:
                self._watch_in_thread(j.id, store, f'{self._remote_dir()}/logs/{j.id}.log', '')
        while True:
            job = store.next_queued(self.name)
            if job is None:
                return
            self.start(job, store)
