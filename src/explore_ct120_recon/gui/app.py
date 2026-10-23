"""The web application: a JSON API and the built React page.

Routes (all under ``/api``):
  GET  /info                        package version, the machines and their state
  GET  /machines                    every machine with its live description
  PUT  /machines/{name}             add or edit a machine (body: the Machine fields)
  DELETE /machines/{name}
  POST /machines/{name}/check       health checks (ssh, paths, python, package, slurm)
  GET  /machines/{name}/scans       scan folders under the machine's scans root
  POST /roi/preview                 coarse FDK of a scan for the ROI picker: {executor, data_folder, ...}
  GET  /machines/template/default   the package's Slurm template
  WS   /machines/{name}/login       interactive login (terminal relay) to open the SSH master
  GET  /drivers                     the drivers and, for the learned one, its algorithms
  GET  /drivers/{name}/schema       the form (``?algorithm=`` for the learned driver)
  POST /drivers/{name}/output       the path the run will write, from {algorithm, values}
  POST /drivers/{name}/checks       are the inputs all there, on the chosen machine
  GET  /jobs                        recent jobs (``?status=`` filters)
  POST /jobs                        submit: {driver, algorithm, values, executor, note, resources}
  GET  /jobs/{id}                   one job
  GET  /jobs/{id}/log               the log from ``?offset=`` (bytes), for tailing
  GET  /jobs/{id}/files             the run's files (what delete-with-data removes)
  DELETE /jobs/{id}                 remove the record (``?data=true``: and the files + log)
  POST /jobs/{id}/cancel

Everything else is served from ``static/`` (the built page), with the SPA
fallback to ``index.html``.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from .executors import Executor, LocalExecutor, Worker
from .files import run_snippet
from .jobs import ACTIVE, Job, JobStore
from .machines import DEFAULT_TEMPLATE, Machine, MachineRegistry
from .remote import RemoteExecutor, SlurmExecutor
from .inputs import plan_for, run_checks
from .schema import DRIVERS, build_command, driver_schema, predict_output, shell_string

STATIC_DIR = Path(__file__).with_name('static')


@dataclass
class Settings:
    data_dir: Path                      # jobs.sqlite, logs/, machines.toml, ssh sockets
    scans_root: Optional[Path] = None   # local scans
    cwd: Optional[Path] = None          # where local jobs run
    slots: Optional[int] = None         # concurrent local jobs; default = GPU count
    env: dict = field(default_factory=dict)   # extra environment for every job (WANDB_PROJECT ...)

    @staticmethod
    def default_data_dir() -> Path:
        import platformdirs
        return Path(platformdirs.user_data_dir('explore_ct120_recon')) / 'gui'


class SubmitRequest(BaseModel):
    driver: str
    algorithm: Optional[str] = None
    values: dict[str, Any] = {}
    executor: str = 'local'
    note: str = ''
    resources: str = ''     # extra sbatch arguments (slurm machines)


class OutputRequest(BaseModel):
    algorithm: Optional[str] = None
    values: dict[str, Any] = {}


class ChecksRequest(OutputRequest):
    executor: str = 'local'


class BrowseRequest(BaseModel):
    path: Optional[str] = None     # directory (or a file / not-yet-existing path: its nearest folder)
    hidden: bool = False


class VolumeRequest(BaseModel):
    path: str
    executor: str = 'local'


class RoiPreviewRequest(BaseModel):
    executor: str = 'local'
    data_folder: str
    scan_folder: Optional[str] = None
    phase: str = '00'
    refresh: bool = False       # recompute even when a cached preview exists


class MachineRequest(BaseModel):
    type: str = 'ssh'
    host: str = ''
    python: str = 'python'
    workdir: str = '~'
    scans_root: str = ''
    slots: int = 1
    env: dict[str, str] = {}
    sbatch_template: Optional[str] = None
    connection: str = 'auto'


class Executors:
    """The executor per machine, rebuilt when the registry changes."""

    def __init__(self, settings: Settings, registry: MachineRegistry):
        self.settings, self.registry = settings, registry
        self._lock = threading.Lock()
        self.local = LocalExecutor(settings.data_dir / 'logs', cwd=settings.cwd, slots=settings.slots,
                                   env=settings.env, scans_root=settings.scans_root)
        self._by_name: dict[str, Executor] = {'local': self.local}
        self.reload()

    def reload(self) -> None:
        machines = self.registry.load()
        with self._lock:
            keep = {'local': self.local}
            for name, m in machines.items():
                if m.type == 'local':
                    continue
                old = self._by_name.get(name)
                if isinstance(old, RemoteExecutor) and old.machine == m:
                    keep[name] = old
                    continue
                cls = SlurmExecutor if m.type == 'slurm' else RemoteExecutor
                keep[name] = cls(m, self.settings.data_dir / 'ssh', self.settings.data_dir / 'logs',
                                 env=self.settings.env)
            self._by_name = keep

    def get(self, name: str) -> Optional[Executor]:
        with self._lock:
            return self._by_name.get(name)

    def all(self) -> dict[str, Executor]:
        with self._lock:
            return dict(self._by_name)

    # Worker expects a mapping; give it a live view.
    def values(self):
        return self.all().values()


def create_app(settings: Settings, *, start_worker: bool = True) -> FastAPI:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore(settings.data_dir / 'jobs.sqlite')
    registry = MachineRegistry(settings.data_dir)
    executors = Executors(settings, registry)
    worker = Worker(store, executors)   # type: ignore[arg-type]

    @asynccontextmanager
    async def lifespan(_app):
        if start_worker:
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title='eXplore CT 120 reconstruction', version=__version__, lifespan=lifespan)
    app.state.store, app.state.executors, app.state.worker, app.state.settings = store, executors, worker, settings
    app.state.registry = registry
    app.state.stop_server = None       # set by ``ct120-gui serve``: ends the server loop
    app.state.server_info = {'pid': os.getpid(), 'url': None, 'started': time.time()}

    def _machine_view(name: str, ex: Executor, fresh: bool = False) -> dict:
        m = registry.load().get(name)
        d = ex.describe(fresh=fresh)
        d['config'] = m.to_dict() if m else {'name': name, 'type': 'local'}
        d['running'] = [j.id for j in store.running(name)]
        d['pending'] = [j.id for j in store.list(status='pending') if j.executor == name]
        if d.get('queue') and d['queue'].get('jobs'):
            # rows of the user's scheduler queue that were submitted from here link to their job page
            ours = {j.remote_id: j.id for j in store.running(name) + store.list(status='pending') if j.executor == name and j.remote_id}
            for row in d['queue']['jobs']:
                row['job'] = ours.get(row['id'])
        return d

    @app.get('/api/info')
    def info():
        return {'version': __version__,
                'executors': [_machine_view(n, ex) for n, ex in executors.all().items()],
                'scans_root': str(settings.scans_root) if settings.scans_root else None,
                'cwd': str(settings.cwd or Path.cwd()),
                'data_dir': str(settings.data_dir)}

    @app.get('/api/server')
    def server():
        """Who is serving: used by ``ct120-gui`` to find a running server."""
        active = [j for j in store.list() if j.status in ACTIVE]
        return {**app.state.server_info, 'version': __version__, 'data_dir': str(settings.data_dir),
                'active': len(active), 'stoppable': app.state.stop_server is not None}

    @app.post('/api/server/stop')
    def stop_server():
        """Stop the server. Local jobs run in their own sessions and remote
        ones on their machines, so they continue; the next server picks them
        up again. Refused when the server is embedded (tests) rather than
        run by ``ct120-gui``."""
        if app.state.stop_server is None:
            raise HTTPException(409, 'this server cannot stop itself; end the process that started it')
        active = [j for j in store.list() if j.status in ACTIVE]
        threading.Timer(0.5, app.state.stop_server).start()   # after this response has gone out
        return {'stopping': True, 'active': len(active), 'pid': app.state.server_info.get('pid')}

    @app.get('/api/machines')
    def machines(fresh: bool = False):
        return [_machine_view(n, ex, fresh) for n, ex in executors.all().items()]

    @app.get('/api/machines/template/default')
    def default_template():
        return PlainTextResponse(DEFAULT_TEMPLATE.read_text())

    @app.put('/api/machines/{name}')
    def put_machine(name: str, req: MachineRequest):
        m = Machine(name=name, **req.model_dump())
        if m.type == 'slurm' and m.sbatch_template is not None and m.sbatch_template.strip() == DEFAULT_TEMPLATE.read_text().strip():
            m.sbatch_template = None
        try:
            registry.put(m)
        except ValueError as e:
            raise HTTPException(400, str(e))
        executors.reload()
        return _machine_view(name, executors.get(name))

    @app.delete('/api/machines/{name}')
    def delete_machine(name: str):
        if name == 'local':
            raise HTTPException(400, 'the local machine cannot be removed')
        if any(j.executor == name for j in store.list(limit=2000) if j.status in ACTIVE):
            raise HTTPException(409, 'the machine has active jobs')
        registry.remove(name)
        executors.reload()
        return {'removed': name}

    @app.post('/api/machines/{name}/check')
    def check_machine(name: str):
        ex = executors.get(name)
        if ex is None:
            raise HTTPException(404)
        return {'checks': ex.health(), 'machine': _machine_view(name, ex, fresh=True)}

    @app.post('/api/machines/{name}/disconnect')
    def disconnect_machine(name: str):
        """Log out: drop the SSH master and stay logged out until Connect."""
        ex = executors.get(name)
        if not isinstance(ex, RemoteExecutor):
            raise HTTPException(400, f'{name} is not a remote machine')
        ex.conn.close()
        return _machine_view(name, ex, fresh=True)

    @app.post('/api/machines/{name}/connect')
    def connect_machine(name: str):
        """Allow connecting again; the status says whether a key sufficed
        (``connected``) or the login terminal is needed (``needs_login``)."""
        ex = executors.get(name)
        if not isinstance(ex, RemoteExecutor):
            raise HTTPException(400, f'{name} is not a remote machine')
        ex.conn.reopen()
        return _machine_view(name, ex, fresh=True)

    @app.post('/api/machines/{name}/queue/{slurm_id}/cancel')
    def cancel_queue_row(name: str, slurm_id: str):
        """scancel one job of the user's queue on a Slurm machine. A job that
        was submitted from here goes through its own record, so it is marked
        cancelled at once."""
        ex = executors.get(name)
        if ex is None:
            raise HTTPException(404)
        if getattr(ex, 'kind', '') != 'slurm':
            raise HTTPException(400, f'{name} is not a Slurm machine')
        ours = next((j for j in store.running(name) + store.list(status='pending')
                     if j.executor == name and j.remote_id == slurm_id), None)
        if ours is not None:
            ex.cancel(ours, store)
        else:
            try:
                ex.scancel(slurm_id)
            except (ValueError, RuntimeError) as e:
                raise HTTPException(400, str(e))
        return {'cancelled': slurm_id, 'job': ours.id if ours else None, 'machine': _machine_view(name, ex, fresh=True)}

    @app.post('/api/machines/{name}/browse')
    def browse(name: str, req: BrowseRequest):
        """One directory of the machine's file system, for the page's file picker."""
        from .browse import list_dir
        ex = executors.get(name)
        if ex is None:
            raise HTTPException(404)
        try:
            if isinstance(ex, RemoteExecutor):
                if ex.conn.status() != 'connected':
                    raise HTTPException(409, f'{name} is not connected')
                return list_dir(req.path, cwd=ex.workdir, scans_root=ex.scans_root or None,
                                conn=ex.conn, python=ex.python, hidden=req.hidden)
            return list_dir(req.path, cwd=str(settings.cwd or Path.cwd()),
                            scans_root=str(settings.scans_root) if settings.scans_root else None, hidden=req.hidden)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f'could not list on {name}: {e}')

    # -- volumes: slices for the viewer -----------------------------------------
    from .volumes import SliceCache, volume_info, volume_slice
    slices = SliceCache()

    def _volume_machine(name: str):
        ex = executors.get(name)
        if ex is None:
            raise HTTPException(404, f'unknown machine {name!r}')
        if isinstance(ex, RemoteExecutor):
            if ex.conn.status() != 'connected':
                raise HTTPException(409, f'{name} is not connected')
            return dict(cwd=ex.workdir, conn=ex.conn, python=ex.python)
        return dict(cwd=str(settings.cwd or Path.cwd()))

    @app.post('/api/volumes/info')
    def volume_header(req: VolumeRequest):
        """Dimensions, voxel size and a sample of the values of a .vff, read where it lives."""
        try:
            return volume_info(req.path, **_volume_machine(req.executor))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f'cannot read {req.path} on {req.executor}: {e}')

    @app.get('/api/volumes/slice')
    def volume_plane(path: str, axis: str, index: int, executor: str = 'local', mtime: float = 0):
        """One slice as little-endian int16 bytes; header X-Shape = rows,cols.
        ``mtime`` (from the info call) keys the cache so a rewritten volume is re-read."""
        key = (executor, path, axis, index, mtime)
        hit = slices.get(key)
        if hit is None:
            try:
                hit = volume_slice(path, axis, index, **_volume_machine(executor))
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(400, f'cannot read {path} on {executor}: {e}')
            slices.put(key, hit)
        (rows, cols), data = hit
        return Response(content=data, media_type='application/octet-stream',
                        headers={'X-Shape': f'{rows},{cols}', 'Cache-Control': 'private, max-age=3600'})

    # -- ROI preview: a coarse FDK for the new-job page's picker ---------------
    from .roi_preview import PreviewCache, run_preview, spec_for
    previews = PreviewCache(settings.data_dir / 'previews')
    preview_lock = threading.Lock()      # one coarse reconstruction at a time

    @app.post('/api/roi/preview')
    def roi_preview(req: RoiPreviewRequest):
        """The whole attenuating field of a scan at 0.4 mm, reconstructed
        where the scan lives, for drawing the ``--roi`` box on. Cached on
        disk per machine, scan and phase; ``refresh`` recomputes."""
        ex = executors.get(req.executor)
        if ex is None:
            raise HTTPException(404, f'unknown machine {req.executor!r}')
        if not req.data_folder.strip():
            raise HTTPException(400, 'give the scan folder first')
        where = _volume_machine(req.executor)
        spec = spec_for(req.data_folder.strip(), cwd=where.get('cwd'), scan_folder=req.scan_folder or None,
                        phase=req.phase or '00')
        key = previews.key(req.executor, spec)
        if req.refresh:
            previews.forget(key)
        else:
            hit = previews.get(key)
            if hit is not None:
                return {**hit, 'cached': True}
        with preview_lock:
            hit = previews.get(key)
            if hit is not None:
                return {**hit, 'cached': True}
            try:
                result = run_preview(spec, conn=where.get('conn'), python=where.get('python') or ex.python)
            except subprocess.TimeoutExpired:
                raise HTTPException(504, 'the preview reconstruction took too long')
            except Exception as e:
                raise HTTPException(400, f'preview on {req.executor} failed: {e}')
            previews.put(key, result)
        return {**result, 'cached': False}

    @app.get('/api/machines/{name}/scans')
    def machine_scans(name: str):
        ex = executors.get(name)
        if ex is None:
            raise HTTPException(404)
        try:
            return ex.list_scans()
        except Exception as e:
            raise HTTPException(502, str(e))

    @app.websocket('/api/machines/{name}/login')
    async def login(ws: WebSocket, name: str):
        """Terminal relay: text frames both ways; a JSON frame {"resize":[rows,cols]} from the page;
        a final JSON frame {"exit": code, "status": ...} from the server."""
        ex = executors.get(name)
        await ws.accept()
        if not isinstance(ex, RemoteExecutor):
            await ws.send_json({'exit': 1, 'status': 'error', 'message': 'not a remote machine'})
            await ws.close()
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        session = ex.conn.login_session(
            on_output=lambda b: loop.call_soon_threadsafe(queue.put_nowait, ('out', b)),
            on_exit=lambda rc: loop.call_soon_threadsafe(queue.put_nowait, ('exit', rc)))
        await ws.send_text(f'$ {session.command}\r\n')

        async def pump():
            while True:
                kind, payload = await queue.get()
                if kind == 'out':
                    await ws.send_text(payload.decode('utf-8', errors='replace'))
                else:
                    status = ex.conn.status()
                    await ws.send_json({'exit': payload, 'status': status})
                    return

        pump_task = asyncio.create_task(pump())
        try:
            while not pump_task.done():
                try:
                    msg = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if msg.startswith('{"resize"'):
                    import json
                    rows, cols = json.loads(msg)['resize']
                    session.resize(int(rows), int(cols))
                else:
                    session.write(msg.encode())
            await pump_task
        except WebSocketDisconnect:
            session.close()
            pump_task.cancel()
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    @app.get('/api/drivers')
    def drivers():
        return [{'name': d.name, 'title': d.title, 'script': d.script, 'kind': d.kind,
                 'primary': d.primary, 'algorithms': d.algorithms()} for d in DRIVERS.values()]

    @app.get('/api/drivers/{name}/schema')
    def schema(name: str, algorithm: Optional[str] = None):
        d = DRIVERS.get(name)
        if d is None:
            raise HTTPException(404, f'unknown driver {name!r}')
        try:
            return driver_schema(d, algorithm)
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.post('/api/drivers/{name}/output')
    def output(name: str, req: OutputRequest):
        """Where the run will write, by the driver's own naming rule."""
        d = DRIVERS.get(name)
        if d is None:
            raise HTTPException(404, f'unknown driver {name!r}')
        algorithm = req.algorithm if d.has_algorithms else None
        try:
            return predict_output(d, driver_schema(d, algorithm), req.values, algorithm) or {}
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.post('/api/drivers/{name}/checks')
    def checks(name: str, req: ChecksRequest):
        """Groups of {label, status, detail} items, one group per input path,
        checked on the machine the job would run on."""
        d = DRIVERS.get(name)
        if d is None:
            raise HTTPException(404, f'unknown driver {name!r}')
        ex = executors.get(req.executor)
        if ex is None:
            raise HTTPException(400, f'unknown machine {req.executor!r}')
        algorithm = req.algorithm if d.has_algorithms else None
        try:
            plan = plan_for(d, driver_schema(d, algorithm), req.values, algorithm)
        except Exception as e:
            raise HTTPException(400, str(e))
        if not plan:
            return {'groups': []}
        try:
            if isinstance(ex, RemoteExecutor):
                if ex.conn.status() != 'connected':
                    return {'groups': [{'title': ex.name, 'path': '', 'items': [
                        {'label': 'connection', 'status': 'info',
                         'detail': 'connect the machine on the Machines page to check its files'}]}]}
                groups = run_checks(plan, cwd=ex.workdir, conn=ex.conn, python=ex.python)
            else:
                groups = run_checks(plan, cwd=str(settings.cwd or Path.cwd()), python=ex.python)
        except Exception as e:
            return {'groups': [{'title': 'check', 'path': '', 'items': [
                {'label': 'checker', 'status': 'warn', 'detail': f'could not run: {e}'}]}]}
        return {'groups': groups}

    @app.get('/api/scans')
    def scans():
        return executors.local.list_scans()

    @app.get('/api/jobs')
    def jobs(limit: int = Query(200, le=2000), status: Optional[str] = None):
        return [j.to_dict() for j in store.list(limit=limit, status=status)]

    def _build_job(req: SubmitRequest, lenient: bool = False):
        """The job a submission would record. ``lenient`` (previews) fills a
        missing required value with a placeholder instead of refusing."""
        d = DRIVERS.get(req.driver)
        if d is None:
            raise HTTPException(400, f'unknown driver {req.driver!r}')
        ex = executors.get(req.executor)
        if ex is None:
            raise HTTPException(400, f'unknown machine {req.executor!r}')
        algorithm = req.algorithm if d.has_algorithms else None
        values = dict(req.values)
        try:
            sch = driver_schema(d, algorithm)
            if lenient:
                for f in sch['fields']:
                    if (f['positional'] or f['required']) and not values.get(f['dest']):
                        values[f['dest']] = f'<{f["dest"]}>'
            cmd = build_command(d, sch, values, algorithm, python=ex.python)
        except ValueError as e:
            raise HTTPException(400, str(e))
        primary = req.values.get(d.primary, '')
        cwd = getattr(getattr(ex, 'machine', None), 'workdir', None) or str(settings.cwd or Path.cwd())
        job = Job.new(d.name, algorithm, req.values, cmd, req.executor, note=req.note,
                      input=str(primary), cwd=cwd, resources=req.resources)
        try:
            pred = predict_output(d, sch, req.values, algorithm)
        except Exception:
            pred = None
        if pred and pred.get('path'):
            job.output_path = pred['path']
        return job, ex

    @app.post('/api/jobs', status_code=201)
    def submit(req: SubmitRequest):
        job, _ = _build_job(req)
        store.add(job)
        return job.to_dict()

    @app.post('/api/jobs/preview')
    def preview(req: SubmitRequest):
        """What submitting this form would run, without recording anything:
        the command, and on a Slurm machine the rendered sbatch script with
        its effective directives."""
        job, ex = _build_job(req, lenient=True)
        out = {'command': shlex.join(job.command), 'cwd': job.cwd, 'script': None, 'directives': []}
        if getattr(ex, 'kind', '') == 'slurm':
            try:
                out['script'] = ex.render(job)
                out['directives'] = ex.directives(job)
            except ValueError as e:
                out['error'] = str(e)
        return out

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        j = store.get(job_id)
        if j is None:
            raise HTTPException(404)
        d = j.to_dict()
        d['shell'] = shell_string(j.command)
        return d

    @app.get('/api/jobs/{job_id}/log')
    def job_log(job_id: str, offset: int = 0, limit: int = 2_000_000):
        j = store.get(job_id)
        if j is None:
            raise HTTPException(404)
        if not j.log_path or not os.path.exists(j.log_path):
            return {'offset': 0, 'text': '', 'size': 0}
        size = os.path.getsize(j.log_path)
        offset = max(0, min(offset, size))
        with open(j.log_path, 'rb') as f:
            f.seek(offset)
            chunk = f.read(limit)
        return {'offset': offset + len(chunk), 'size': size, 'text': chunk.decode('utf-8', errors='replace')}

    def _job_files(j: Job, action: str) -> list[dict]:
        if not j.output_path:
            return []
        ex = executors.get(j.executor)
        base = j.output_path
        if isinstance(ex, RemoteExecutor):
            if not os.path.isabs(base) and not base.startswith('~'):
                base = f'{ex.workdir.rstrip("/")}/{base}'
            return run_snippet(base, action, conn=ex.conn, python=ex.python)
        if not os.path.isabs(base):
            base = os.path.join(j.cwd or str(settings.cwd or Path.cwd()), base)
        return run_snippet(base, action)

    @app.get('/api/jobs/{job_id}/files')
    def job_files(job_id: str):
        """What `delete with data` would remove: the run's own files by name."""
        j = store.get(job_id)
        if j is None:
            raise HTTPException(404)
        try:
            return {'files': _job_files(j, 'list'), 'output_path': j.output_path}
        except Exception as e:
            raise HTTPException(502, str(e))

    @app.delete('/api/jobs/{job_id}')
    def delete_job(job_id: str, data: bool = False):
        """Remove the record; with ``?data=true`` also the run's files and log."""
        j = store.get(job_id)
        if j is None:
            raise HTTPException(404)
        if j.status in ACTIVE:
            raise HTTPException(409, 'cancel the job first')
        removed = []
        if data:
            try:
                removed = _job_files(j, 'delete')
            except Exception as e:
                raise HTTPException(502, str(e))
        for p in (j.log_path, (str(Path(j.log_path).with_suffix('.exit')) if j.log_path else None)):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        store.delete(job_id)
        return {'deleted': job_id, 'files': removed}

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        j = store.get(job_id)
        if j is None:
            raise HTTPException(404)
        if j.status not in ACTIVE:
            raise HTTPException(409, f'job is {j.status}')
        ex = executors.get(j.executor)
        if ex is None or not ex.cancel(j, store):
            raise HTTPException(409, 'job is not under this server\'s control')
        return store.get(job_id).to_dict()

    if STATIC_DIR.is_dir() and (STATIC_DIR / 'index.html').exists():
        app.mount('/assets', StaticFiles(directory=STATIC_DIR / 'assets'), name='assets')

        @app.get('/{path:path}', include_in_schema=False)
        def spa(path: str):
            candidate = STATIC_DIR / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / 'index.html')
    else:
        @app.get('/', include_in_schema=False)
        def no_page():
            return PlainTextResponse('The web page has not been built (gui/static is empty). '
                                     'The JSON API is up under /api; see /docs.')
    return app
