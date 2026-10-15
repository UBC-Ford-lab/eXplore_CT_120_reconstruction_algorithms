"""``ct120-gui``: the page as a desktop application.

``ct120-gui``                 start the server in the background if it is not
                              running, then open the page in the browser
``ct120-gui serve …``         run the server in the foreground (a lab server,
                              a terminal, a service)
``ct120-gui stop``            stop the background server (jobs keep running)
``ct120-gui status``          is a server up, and where
``ct120-gui shortcut``        put an icon in the application menu / Start menu /
                              ~/Applications that runs ``ct120-gui``
``ct120-gui shortcut --remove``
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from .app import Settings, create_app

APP_NAME = 'eXplore CT 120 reconstruction'
ICON_DIR = Path(__file__).with_name('icons')
SUBCOMMANDS = ('serve', 'stop', 'status', 'shortcut')


def _server_options(p: argparse.ArgumentParser) -> None:
    p.add_argument('--host', default='127.0.0.1',
                   help='Interface to bind. 127.0.0.1 (default) is this machine only; '
                        '0.0.0.0 serves the lab network.')
    p.add_argument('--port', type=int, default=8120)
    p.add_argument('--scans-root', type=Path, default=None,
                   help='Folder whose subfolders are scans; offered as choices in the form. '
                        'Default: $CT_SCANS_ROOT, else data/scans if it exists.')
    p.add_argument('--data-dir', type=Path, default=None,
                   help='Where the job table and logs live. Default: the per-user data '
                        'directory (~/.local/share/explore_ct120_recon/gui on Linux).')
    p.add_argument('--cwd', type=Path, default=None,
                   help='Working directory for jobs; relative paths in the form resolve here '
                        '(default: the current directory).')
    p.add_argument('--slots', type=int, default=None,
                   help='Concurrent local jobs (default: one per GPU).')
    p.add_argument('--wandb-project', default=None,
                   help='Passed to every job as WANDB_PROJECT (default: inherit the environment).')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='ct120-gui',
        description='The reconstruction page as a desktop application. With no '
                    'subcommand: start the server in the background if needed and '
                    'open the page in the browser.')
    sub = p.add_subparsers(dest='command')
    launch = p   # the default command's options live on the main parser
    _server_options(launch)
    launch.add_argument('--no-open', action='store_true', help='Start (or find) the server but do not open a browser.')

    serve = sub.add_parser('serve', help='Run the server in the foreground.')
    _server_options(serve)
    serve.add_argument('--reload', action='store_true', help='Developer auto-reload.')

    stop = sub.add_parser('stop', help='Stop the background server. Jobs keep running.')
    stop.add_argument('--port', type=int, default=8120)
    stop.add_argument('--data-dir', type=Path, default=None)

    status = sub.add_parser('status', help='Is a server up?')
    status.add_argument('--port', type=int, default=8120)

    sc = sub.add_parser('shortcut', help='Install (or remove) the desktop icon.')
    sc.add_argument('--remove', action='store_true')
    sc.add_argument('--port', type=int, default=None, help='Port the icon starts the server on (default 8120).')
    sc.add_argument('--scans-root', type=Path, default=None, help='Scans folder baked into the icon.')
    sc.add_argument('--wandb-project', default=None)
    sc.add_argument('--desktop', action='store_true', help='Also put the icon on the Desktop.')
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


# -- the server --------------------------------------------------------------
def _settings(args) -> Settings:
    scans_root = args.scans_root or (Path(os.environ['CT_SCANS_ROOT']) if os.environ.get('CT_SCANS_ROOT')
                                     else (Path('data/scans') if Path('data/scans').is_dir() else None))
    env = {'WANDB_PROJECT': args.wandb_project} if args.wandb_project else {}
    return Settings(data_dir=args.data_dir or Settings.default_data_dir(),
                    scans_root=scans_root.resolve() if scans_root else None,
                    cwd=(args.cwd or Path.cwd()).resolve(), slots=args.slots, env=env)


def _uvicorn():
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('ct120-gui needs the gui extra: pip install "explore_ct120_recon[gui]"')
    return uvicorn


def serve(args) -> None:
    uvicorn = _uvicorn()
    settings = _settings(args)
    app = create_app(settings)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level='warning', reload=getattr(args, 'reload', False))
    server = uvicorn.Server(config)
    # the page's "stop server" button and ``ct120-gui stop`` end the loop gracefully
    app.state.stop_server = lambda: setattr(server, 'should_exit', True)
    app.state.server_info = {'pid': os.getpid(), 'host': args.host, 'port': args.port, 'started': time.time(),
                             'url': _url(args.port)}
    pid_file = settings.data_dir / 'server.pid'
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(json.dumps({'pid': os.getpid(), 'port': args.port, 'host': args.host}))
    print(f'ct120-gui: {_url(args.port)}  (jobs and logs in {settings.data_dir})', flush=True)
    try:
        server.run()
    finally:
        try:
            if json.loads(pid_file.read_text()).get('pid') == os.getpid():
                pid_file.unlink()
        except (OSError, ValueError):
            pass


def _url(port: int) -> str:
    return f'http://127.0.0.1:{port}'


def server_info(port: int, timeout: float = 2.0) -> dict | None:
    """``/api/server`` of a running server on this port, or None."""
    try:
        with urllib.request.urlopen(_url(port) + '/api/server', timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _detach_kwargs() -> dict:
    if sys.platform == 'win32':
        flags = getattr(subprocess, 'DETACHED_PROCESS', 0x8) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200)
        return {'creationflags': flags}
    return {'start_new_session': True}


def launch(args) -> int:
    """Default command: make sure a server runs, then open the page."""
    info = server_info(args.port)
    if info is None:
        settings_dir = (args.data_dir or Settings.default_data_dir())
        settings_dir.mkdir(parents=True, exist_ok=True)
        log = settings_dir / 'server.log'
        cmd = [sys.executable, '-m', 'explore_ct120_recon.gui.cli', 'serve', '--host', args.host, '--port', str(args.port)]
        for flag, value in (('--scans-root', args.scans_root), ('--data-dir', args.data_dir), ('--cwd', args.cwd),
                            ('--slots', args.slots), ('--wandb-project', args.wandb_project)):
            if value is not None:
                cmd += [flag, str(value)]
        with open(log, 'ab') as f:
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **_detach_kwargs())
        for _ in range(100):
            time.sleep(0.2)
            info = server_info(args.port, timeout=1)
            if info:
                break
        if info is None:
            print(f'ct120-gui: the server did not come up; see {log}', file=sys.stderr)
            return 1
        print(f'ct120-gui: server started, pid {info["pid"]}, {info["url"]}')
    else:
        print(f'ct120-gui: server already running, pid {info["pid"]}, {info["url"]}')
    if not args.no_open:
        webbrowser.open(info['url'])
    return 0


def stop(args) -> int:
    info = server_info(args.port)
    if info:
        try:
            req = urllib.request.Request(_url(args.port) + '/api/server/stop', method='POST')
            with urllib.request.urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
            print(f'ct120-gui: stopping pid {info["pid"]}'
                  + (f' ({d["active"]} active job(s) keep running)' if d.get('active') else ''))
            return 0
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f'ct120-gui: stop request failed ({e}); trying the pid file', file=sys.stderr)
    pid_file = (args.data_dir or Settings.default_data_dir()) / 'server.pid'
    try:
        pid = json.loads(pid_file.read_text())['pid']
    except (OSError, ValueError, KeyError):
        print('ct120-gui: no server running')
        return 1
    try:
        if sys.platform == 'win32':
            subprocess.run(['taskkill', '/PID', str(pid), '/F'], capture_output=True)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
        print(f'ct120-gui: sent stop to pid {pid}')
        return 0
    except OSError as e:
        print(f'ct120-gui: {e}', file=sys.stderr)
        return 1


def status(args) -> int:
    info = server_info(args.port)
    if info:
        print(f'ct120-gui: running, pid {info["pid"]}, {info["url"]}, {info["active"]} active job(s), data in {info["data_dir"]}')
        return 0
    print(f'ct120-gui: no server on port {args.port}')
    return 1


# -- the desktop icon ------------------------------------------------------------
def _launch_command(args) -> list[str]:
    """What the icon runs: this interpreter, this module, the baked-in options."""
    exe = sys.executable
    if sys.platform == 'win32':
        pythonw = Path(exe).with_name('pythonw.exe')   # no console window
        if pythonw.exists():
            exe = str(pythonw)
    cmd = [exe, '-m', 'explore_ct120_recon.gui.cli']
    if args.port:
        cmd += ['--port', str(args.port)]
    if args.scans_root:
        cmd += ['--scans-root', str(Path(args.scans_root).resolve())]
    if args.wandb_project:
        cmd += ['--wandb-project', args.wandb_project]
    return cmd


def shortcut_paths() -> dict[str, Path]:
    home = Path.home()
    if sys.platform == 'win32':
        programs = Path(os.environ.get('APPDATA', home / 'AppData' / 'Roaming')) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'
        return {'menu': programs / f'{APP_NAME}.lnk', 'desktop': home / 'Desktop' / f'{APP_NAME}.lnk'}
    if sys.platform == 'darwin':
        return {'menu': home / 'Applications' / f'{APP_NAME}.app', 'desktop': home / 'Desktop' / f'{APP_NAME}.app'}
    data = Path(os.environ.get('XDG_DATA_HOME', home / '.local' / 'share'))
    return {'menu': data / 'applications' / 'ct120-gui.desktop',
            'icon': data / 'icons' / 'hicolor' / '256x256' / 'apps' / 'ct120-gui.png',
            'desktop': home / 'Desktop' / 'ct120-gui.desktop'}


def install_shortcut(args) -> int:
    paths = shortcut_paths()
    cmd = _launch_command(args)
    targets = [paths['menu']] + ([paths['desktop']] if args.desktop else [])
    if sys.platform == 'win32':
        for t in targets:
            _write_lnk(t, cmd)
    elif sys.platform == 'darwin':
        for t in targets:
            _write_app_bundle(t, cmd)
    else:
        paths['icon'].parent.mkdir(parents=True, exist_ok=True)
        paths['icon'].write_bytes((ICON_DIR / 'ct120-gui.png').read_bytes())
        entry = ('[Desktop Entry]\nType=Application\nVersion=1.0\n'
                 f'Name={APP_NAME}\nComment=Reconstruct eXplore CT 120 scans\n'
                 f'Exec={shlex.join(cmd)}\nIcon=ct120-gui\nTerminal=false\n'
                 'Categories=Science;Education;\nKeywords=CT;reconstruction;microCT;\n'
                 'StartupNotify=false\n')
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(entry)
            t.chmod(0o755)
        subprocess.run(['update-desktop-database', str(paths['menu'].parent)], capture_output=True)
    for t in targets:
        print(f'ct120-gui: installed {t}')
    print(f'  runs: {shlex.join(cmd)}')
    return 0


def remove_shortcut() -> int:
    import shutil
    paths = shortcut_paths()
    removed = 0
    for t in paths.values():
        if t.is_dir():
            shutil.rmtree(t)
            removed += 1
        elif t.exists():
            t.unlink()
            removed += 1
        else:
            continue
        print(f'ct120-gui: removed {t}')
    if not removed:
        print('ct120-gui: no shortcut installed')
    return 0


def _write_lnk(target: Path, cmd: list[str]) -> None:
    """A Start-menu / Desktop shortcut through the Windows Script Host (no extra package)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    icon = ICON_DIR / 'ct120-gui.ico'
    vbs = (f'Set s = CreateObject("WScript.Shell")\n'
           f'Set l = s.CreateShortcut("{target}")\n'
           f'l.TargetPath = "{cmd[0]}"\n'
           f'l.Arguments = "{subprocess.list2cmdline(cmd[1:])}"\n'
           f'l.WorkingDirectory = "{Path.home()}"\n'
           f'l.IconLocation = "{icon}"\n'
           f'l.Description = "{APP_NAME}"\n'
           'l.Save\n')
    tmp = target.with_suffix('.vbs')
    tmp.write_text(vbs)
    try:
        subprocess.run(['cscript', '//nologo', str(tmp)], check=True, capture_output=True)
    finally:
        tmp.unlink()


def _write_app_bundle(bundle: Path, cmd: list[str]) -> None:
    """A minimal .app whose executable is a shell stub."""
    (bundle / 'Contents' / 'MacOS').mkdir(parents=True, exist_ok=True)
    (bundle / 'Contents' / 'Resources').mkdir(parents=True, exist_ok=True)
    stub = bundle / 'Contents' / 'MacOS' / 'ct120-gui'
    stub.write_text('#!/bin/sh\nexec ' + shlex.join(cmd) + '\n')
    stub.chmod(0o755)
    icns = bundle / 'Contents' / 'Resources' / 'ct120-gui.icns'
    _png_to_icns(ICON_DIR / 'ct120-gui@2x.png', icns)
    plist = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
             '<plist version="1.0"><dict>\n'
             f'<key>CFBundleName</key><string>{APP_NAME}</string>\n'
             '<key>CFBundleIdentifier</key><string>org.explore-ct120-recon.gui</string>\n'
             '<key>CFBundleExecutable</key><string>ct120-gui</string>\n'
             + ('<key>CFBundleIconFile</key><string>ct120-gui</string>\n' if icns.exists() else '')
             + '<key>CFBundlePackageType</key><string>APPL</string>\n'
             '<key>LSUIElement</key><true/>\n'
             '</dict></plist>\n')
    (bundle / 'Contents' / 'Info.plist').write_text(plist)


def _png_to_icns(png: Path, icns: Path) -> None:
    """Uses the system's iconutil when available; the bundle works without an icon otherwise."""
    import shutil
    import tempfile
    if not shutil.which('iconutil') or not shutil.which('sips'):
        return
    with tempfile.TemporaryDirectory() as d:
        iconset = Path(d) / 'icon.iconset'
        iconset.mkdir()
        for n in (16, 32, 128, 256, 512):
            subprocess.run(['sips', '-z', str(n), str(n), str(png), '--out', str(iconset / f'icon_{n}x{n}.png')], capture_output=True)
            subprocess.run(['sips', '-z', str(2 * n), str(2 * n), str(png), '--out', str(iconset / f'icon_{n}x{n}@2x.png')], capture_output=True)
        subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o', str(icns)], capture_output=True)


def main_rc(argv=None) -> int:
    args = parse_args(argv)
    if args.command == 'serve':
        serve(args)
        return 0
    if args.command == 'stop':
        return stop(args)
    if args.command == 'status':
        return status(args)
    if args.command == 'shortcut':
        return remove_shortcut() if args.remove else install_shortcut(args)
    return launch(args)


def main(argv=None) -> None:
    raise SystemExit(main_rc(argv))


if __name__ == '__main__':
    main()
