"""Where a job runs.

An executor owns a machine: it reports what the machine is (hostname, GPUs,
memory, load), starts a queued job there, watches it and can cancel it. The
``LocalExecutor`` runs the driver as a subprocess on the machine that serves
the page, one job per GPU. A cluster executor (SSH + sbatch) will answer the
same interface and appear beside it in the machine list.

Nothing about a machine is written down here; everything is read at request
time, so the page shows the machine it is actually served from.
"""
from __future__ import annotations

import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .jobs import Job, JobStore

_WANDB_RUN = re.compile(r'View run(?: \S+)? at:? (https://\S+/runs/\S+)')
# Every driver announces where it writes with one of these lines; the first
# match replaces the path predicted at submit time.
_OUTPUT = re.compile(r'^\s*(?:Output path|Report written to|Filtered volume saved to|'
                     r'Synthetic scan written to):?\s*(.+?)\s*$', re.M)


def gpu_inventory() -> list[dict]:
    """Every CUDA device with its memory, and utilisation when nvidia-smi answers."""
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                gpus.append({'index': i, 'name': props.name, 'total_gib': total / 2**30,
                             'free_gib': free / 2**30, 'utilisation': None})
    except Exception:
        pass
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                idx, util, used = [p.strip() for p in line.split(',')]
                for g in gpus:
                    if g['index'] == int(idx):
                        g['utilisation'] = float(util)
                        g['used_gib'] = float(used) / 1024
    except Exception:
        pass
    return gpus


def host_memory() -> dict:
    try:
        page = os.sysconf('SC_PAGE_SIZE')
        return {'total_gib': os.sysconf('SC_PHYS_PAGES') * page / 2**30,
                'available_gib': os.sysconf('SC_AVPHYS_PAGES') * page / 2**30}
    except (ValueError, OSError, AttributeError):
        return {'total_gib': None, 'available_gib': None}


def package_provenance() -> dict:
    """Version, and the git commit when running from a checkout."""
    from .. import __version__
    info = {'version': __version__, 'commit': None, 'python': sys.version.split()[0]}
    root = Path(__file__).resolve().parents[3]
    if (root / '.git').exists():
        try:
            info['commit'] = subprocess.run(['git', '-C', str(root), 'describe', '--always', '--dirty', '--tags'],
                                            capture_output=True, text=True, timeout=5).stdout.strip() or None
        except Exception:
            pass
    return info


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie counts as gone
    try:
        with open(f'/proc/{pid}/stat') as f:
            return f.read().split(')')[-1].split()[0] != 'Z'
    except OSError:
        return True


class Executor:
    name: str = 'base'
    kind: str = 'base'
    python: str = sys.executable      # interpreter the job's command starts with
    slots: int = 1

    def describe(self, fresh: bool = False) -> dict:
        raise NotImplementedError

    def health(self) -> list[dict]:
        return [{'name': 'machine', 'ok': True, 'detail': 'this machine'}]

    def list_scans(self) -> list[dict]:
        return []

    def start(self, job: Job, store: JobStore) -> None:
        raise NotImplementedError

    def cancel(self, job: Job, store: JobStore) -> bool:
        raise NotImplementedError

    def tick(self, store: JobStore) -> None:
        """Called by the worker loop: start queued jobs when a slot is free."""
        raise NotImplementedError


class LocalExecutor(Executor):
    """Runs drivers as subprocesses on this machine, one at a time per GPU."""

    name = 'local'
    kind = 'local'

    def __init__(self, log_dir: Path, *, cwd: Optional[Path] = None,
                 slots: Optional[int] = None, env: Optional[dict] = None,
                 scans_root: Optional[Path] = None):
        self.scans_root = Path(scans_root) if scans_root else None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.env = dict(env or {})
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        gpus = gpu_inventory()
        self.slots = slots if slots is not None else max(1, len(gpus))

    def list_scans(self) -> list[dict]:
        root = self.scans_root
        if not root or not root.is_dir():
            return []
        return [{'name': p.name, 'path': str(p)} for p in sorted(root.iterdir())
                if p.is_dir() and (p / 'scan.xml').exists()]

    def describe(self, fresh: bool = False) -> dict:
        running = [j.id for j in self._running_ids()]
        return {
            'name': self.name,
            'kind': 'local',
            'status': 'connected',
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'cpus': os.cpu_count(),
            'memory': host_memory(),
            'gpus': gpu_inventory(),
            'slots': self.slots,
            'running': running,
            'cwd': str(self.cwd),
            'package': package_provenance(),
        }

    def _running_ids(self) -> list:
        with self._lock:
            return [type('J', (), {'id': k})() for k in self._procs]

    def _free_gpu(self, store: JobStore) -> int:
        busy = {j.gpu_index for j in store.running(self.name) if j.gpu_index is not None}
        n = max(1, len(gpu_inventory()))
        for i in range(n):
            if i not in busy:
                return i
        return 0

    def start(self, job: Job, store: JobStore) -> None:
        gpu = self._free_gpu(store)
        log_path = self.log_dir / f'{job.id}.log'
        env = {**os.environ, **self.env, 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': str(gpu)}
        gpus = gpu_inventory()
        gpu_name = gpus[gpu]['name'] if gpu < len(gpus) else None
        exit_file = self.log_dir / f'{job.id}.exit'
        if exit_file.exists():
            exit_file.unlink()
        log = open(log_path, 'wb')
        try:
            proc = subprocess.Popen([sys.executable, '-m', 'explore_ct120_recon.gui.runner', str(exit_file), '--', *job.command],
                                    cwd=job.cwd or self.cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            log.close()
            store.update(job.id, status='failed', error=str(e), finished=time.time(),
                         log_path=str(log_path))
            return
        with self._lock:
            self._procs[job.id] = proc
        store.update(job.id, status='running', started=time.time(), pid=proc.pid,
                     hostname=socket.gethostname(), gpu_index=gpu, gpu_name=gpu_name,
                     log_path=str(log_path), package_version=package_provenance()['version'])
        threading.Thread(target=self._watch, args=(job.id, proc, log, log_path, store),
                         daemon=True, name=f'job-{job.id}').start()

    def _watch(self, job_id: str, proc, log, log_path: Path, store: JobStore) -> None:
        """``proc`` is a Popen for a job this server started, or an int pid for
        one adopted after a restart (then the exit code comes from the exit file)."""
        seen = {'wandb_url': None, 'output_path': None}
        exit_file = self.log_dir / f'{job_id}.exit'
        while True:
            if isinstance(proc, int):
                rc = self._read_exit(exit_file)
                if rc is None and not _pid_alive(proc):
                    time.sleep(1.0)
                    rc = self._read_exit(exit_file)
                    if rc is None:
                        rc = -1
            else:
                rc = proc.poll()
                if rc is not None:
                    rc = self._read_exit(exit_file, default=rc)
            found = self._scan_log(log_path, seen)
            if found:
                store.update(job_id, **found)
            if rc is not None:
                break
            time.sleep(2.0)
        if log is not None:
            log.close()
        with self._lock:
            self._procs.pop(job_id, None)
        current = store.get(job_id)
        if current and current.status == 'cancelled':
            store.update(job_id, finished=time.time(), returncode=rc)
            return
        status = 'finished' if rc == 0 else 'failed'
        error = None if rc == 0 else self._last_error(log_path)
        store.update(job_id, status=status, returncode=rc, finished=time.time(), error=error)

    @staticmethod
    def _read_exit(exit_file: Path, default=None):
        try:
            return int(exit_file.read_text().strip())
        except (OSError, ValueError):
            return default

    def adopt(self, store: JobStore) -> None:
        """After a restart: jobs this executor recorded as running are still
        running in their own session, finished meanwhile, or gone."""
        for j in store.running(self.name):
            with self._lock:
                if j.id in self._procs:
                    continue
                self._procs[j.id] = j.pid or -1
            log_path = Path(j.log_path) if j.log_path else self.log_dir / f'{j.id}.log'
            if j.pid is None or (self._read_exit(self.log_dir / f'{j.id}.exit') is None and not _pid_alive(j.pid)):
                with self._lock:
                    self._procs.pop(j.id, None)
                store.update(j.id, status='failed', finished=time.time(),
                             error='lost: the process is gone and left no exit record')
                continue
            threading.Thread(target=self._watch, args=(j.id, j.pid, None, log_path, store),
                             daemon=True, name=f'job-{j.id}').start()

    @staticmethod
    def _scan_log(log_path: Path, seen: dict) -> dict:
        found = {}
        try:
            text = log_path.read_text(errors='replace')
        except OSError:
            return found
        if seen['wandb_url'] is None:
            m = _WANDB_RUN.search(text)
            if m:
                seen['wandb_url'] = found['wandb_url'] = m.group(1)
        if not seen.get('output_logged'):
            # The driver's own line wins over the path predicted at submit.
            m = _OUTPUT.search(text)
            if m:
                seen['output_logged'] = True
                path = m.group(1).strip()
                if path != seen.get('output_path'):
                    seen['output_path'] = found['output_path'] = path
        return found

    @staticmethod
    def _last_error(log_path: Path) -> Optional[str]:
        try:
            lines = log_path.read_text(errors='replace').rstrip().splitlines()
        except OSError:
            return None
        for line in reversed(lines[-40:]):
            if line.startswith('Error:') or 'Error' in line or 'error' in line:
                return line.strip()[:500]
        return lines[-1].strip()[:500] if lines else None

    def cancel(self, job: Job, store: JobStore) -> bool:
        if job.status == 'queued':
            store.update(job.id, status='cancelled', finished=time.time())
            return True
        with self._lock:
            proc = self._procs.get(job.id)
        pid = proc if isinstance(proc, int) else (proc.pid if proc is not None else job.pid)
        if not pid or pid < 0:
            return False
        store.update(job.id, status='cancelled')
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        threading.Thread(target=self._kill_after, args=(pid, 20.0), daemon=True).start()
        return True

    @staticmethod
    def _kill_after(pid: int, grace: float) -> None:
        deadline = time.time() + grace
        while time.time() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(0.5)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def tick(self, store: JobStore) -> None:
        self.adopt(store)
        while len(store.running(self.name)) < self.slots:
            job = store.next_queued(self.name)
            if job is None:
                return
            self.start(job, store)


class Worker(threading.Thread):
    """Polls the store and hands queued jobs to their executor."""

    def __init__(self, store: JobStore, executors: dict[str, Executor], interval: float = 1.0):
        super().__init__(daemon=True, name='ct120-gui-worker')
        self.store, self.executors, self.interval = store, executors, interval
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            for ex in self.executors.values():
                try:
                    ex.tick(self.store)
                except Exception as e:  # keep the loop alive; the job records the error
                    print(f'[ct120-gui] worker: {e!r}', file=sys.stderr)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
