"""SSH through a library (paramiko) instead of the OpenSSH client.

The default connection (``remote.SshConnection``) keeps one multiplexed
OpenSSH master per machine, which Windows' OpenSSH cannot do. This backend
keeps one library session per machine instead, with the same interface:
``status``, ``run``, ``check``, ``write_file``, ``read_from``, ``probe``,
``close``/``reopen`` and a login session the page's terminal drives.

Authentication is the user's own: ``~/.ssh/config`` is honoured for the host
name, user, port and identity, the agent and the usual ``~/.ssh/id_*`` keys
are tried first, and whatever the host then asks for (a key passphrase, a
password, a Duo passcode or push choice, a host-key confirmation) is relayed
to the page and answered there. Nothing typed is stored.

Unlike the OpenSSH master the session lives in the server process, so a
server restart means logging in again.
"""
from __future__ import annotations

import os
import queue
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .remote import LoggedOut, SshConnection

KEY_NAMES = ('id_ed25519', 'id_ecdsa', 'id_rsa', 'id_dsa')


class AuthAborted(Exception):
    """The user closed the login terminal before authenticating."""


def _parse_destination(dest: str) -> tuple[Optional[str], str, Optional[int]]:
    """``user@host:port`` → (user, host, port); each part optional but host."""
    user, _, rest = dest.rpartition('@')
    host, _, port = rest.partition(':')
    return (user or None), host, (int(port) if port.isdigit() else None)


def resolve_target(dest: str, config_path: Optional[Path] = None) -> dict:
    """What to connect to, after ``~/.ssh/config`` (HostName, User, Port, IdentityFile)."""
    import paramiko
    user, host, port = _parse_destination(dest)
    out = {'hostname': host, 'username': user, 'port': port, 'identity': []}
    path = config_path or Path.home() / '.ssh' / 'config'
    if path.exists():
        try:
            cfg = paramiko.SSHConfig.from_path(str(path)).lookup(host)
        except Exception:
            cfg = {}
        out['hostname'] = cfg.get('hostname', host)
        out['username'] = user or cfg.get('user')
        out['port'] = port or (int(cfg['port']) if cfg.get('port') else None)
        out['identity'] = [os.path.expanduser(p) for p in cfg.get('identityfile', [])]
    out['port'] = out['port'] or 22
    out['username'] = out['username'] or os.environ.get('USER') or os.environ.get('USERNAME') or os.getlogin()
    return out


class LibraryConnection(SshConnection):
    """One persistent paramiko transport to ``host``."""

    kind = 'library'

    def __init__(self, host: str, name: str = '', *, key_files: Optional[list[str]] = None,
                 known_hosts: Optional[Path] = None, config_path: Optional[Path] = None,
                 use_agent: bool = True):
        self.host = host
        self.name = name or host
        self.socket = None            # no multiplexing socket; kept for the interface
        self.persist = ''
        self.last_error: Optional[str] = None
        self.closed = False
        self.key_files = key_files        # None = the usual ~/.ssh/id_* keys
        self.known_hosts = Path(known_hosts) if known_hosts else Path.home() / '.ssh' / 'known_hosts'
        self.config_path = config_path
        self.use_agent = use_agent
        self.transport = None
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------------
    def master_alive(self) -> bool:
        t = self.transport
        return bool(t is not None and t.is_active() and t.is_authenticated())

    def login_command(self) -> str:
        return f'ssh {self.host}'

    def close(self) -> None:
        self.closed = True
        t, self.transport = self.transport, None
        if t is not None:
            try:
                t.close()
            except Exception:
                pass

    def reopen(self) -> None:
        self.closed = False

    def status(self) -> str:
        if self.master_alive():
            return 'connected'
        if self.closed:
            return 'logged_out'
        import paramiko
        try:
            self.connect(prompt=None)        # keys and agent only; nothing to ask
        except paramiko.BadHostKeyException as e:
            self.last_error = f'host key changed: {e}'
            return 'needs_login'
        except (paramiko.AuthenticationException, AuthAborted) as e:
            self.last_error = str(e) or 'authentication needed'
            return 'needs_login'
        except (OSError, socket.timeout, paramiko.SSHException) as e:
            self.last_error = str(e)[:300] or type(e).__name__
            return 'unreachable'
        self.last_error = None
        return 'connected'

    # -- connecting -------------------------------------------------------------
    def connect(self, prompt: Optional[Callable[[str, bool], str]], *, timeout: float = 20) -> None:
        """Open and authenticate the transport.

        ``prompt(text, echo) -> answer`` is asked whatever the host wants
        beyond keys; ``None`` means fail instead of asking (a status probe)."""
        import paramiko
        target = resolve_target(self.host, self.config_path)
        sock = socket.create_connection((target['hostname'], target['port']), timeout=timeout)
        t = paramiko.Transport(sock)
        t.set_keepalive(30)
        try:
            t.start_client(timeout=timeout)
            self._check_host_key(t, target, prompt)
            self._authenticate(t, target, prompt)
        except BaseException:
            t.close()
            raise
        with self._lock:
            old, self.transport = self.transport, t
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        self.closed = False

    def _check_host_key(self, t, target: dict, prompt) -> None:
        import paramiko
        key = t.get_remote_server_key()
        hostname, port = target['hostname'], target['port']
        entry = hostname if port == 22 else f'[{hostname}]:{port}'
        known = paramiko.HostKeys()
        if self.known_hosts.exists():
            try:
                known.load(str(self.known_hosts))
            except Exception:
                pass
        stored = known.lookup(entry)
        if stored is not None:
            if stored.get(key.get_name()) is None:
                pass                                   # a different key type; treat as unknown
            elif stored[key.get_name()] == key:
                return
            else:
                raise paramiko.BadHostKeyException(hostname, key, stored[key.get_name()])
        fp = key.get_fingerprint().hex(':')
        if prompt is None:
            raise paramiko.AuthenticationException(f'unknown host key for {entry}; log in to confirm it')
        answer = prompt(f"The authenticity of host '{entry}' can't be established.\r\n"
                        f"{key.get_name()} key fingerprint is MD5:{fp}.\r\n"
                        f"Are you sure you want to continue connecting (yes/no)? ", True)
        if answer.strip().lower() not in ('yes', 'y'):
            raise AuthAborted('host key not accepted')
        known.add(entry, key.get_name(), key)
        self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
        known.save(str(self.known_hosts))

    def _candidate_keys(self, target: dict, prompt):
        """Agent keys first, then key files (asking for a passphrase if needed)."""
        import paramiko
        keys = []
        if self.use_agent:
            try:
                keys.extend(paramiko.Agent().get_keys())
            except Exception:
                pass
        files = self.key_files if self.key_files is not None else (
            target['identity'] or [str(Path.home() / '.ssh' / n) for n in KEY_NAMES])
        for path in files:
            if not os.path.exists(path):
                continue
            for cls in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
                try:
                    keys.append(cls.from_private_key_file(path))
                    break
                except paramiko.PasswordRequiredException:
                    if prompt is None:
                        break
                    passphrase = prompt(f"Enter passphrase for key '{path}': ", False)
                    try:
                        keys.append(cls.from_private_key_file(path, password=passphrase))
                    except paramiko.SSHException:
                        pass
                    break
                except paramiko.SSHException:
                    continue
        return keys

    def _authenticate(self, t, target: dict, prompt) -> None:
        import paramiko
        user = target['username']
        allowed: Optional[list] = None
        try:
            allowed = t.auth_none(user)
        except paramiko.BadAuthenticationType as e:
            allowed = list(e.allowed_types)
        except paramiko.AuthenticationException:
            allowed = None
        if t.is_authenticated():
            return
        allowed = allowed or ['publickey', 'keyboard-interactive', 'password']
        if 'publickey' in allowed:
            for key in self._candidate_keys(target, prompt):
                try:
                    allowed = t.auth_publickey(user, key) or allowed
                except paramiko.PartialAuthentication as e:   # the key was accepted; a second factor follows
                    allowed = list(e.allowed_types)
                    break
                except paramiko.AuthenticationException:
                    continue
                if t.is_authenticated():
                    return
        if prompt is None:
            raise paramiko.AuthenticationException('keys were not enough; the host asks for more')
        for _ in range(3):
            if t.is_authenticated():
                return
            if 'keyboard-interactive' in allowed:
                def handler(title, instructions, prompts):
                    if title:
                        _emit(prompt, title)
                    if instructions:
                        _emit(prompt, instructions)
                    return [prompt(text, bool(echo)) for text, echo in prompts]
                try:
                    allowed = t.auth_interactive(user, handler) or allowed
                except paramiko.PartialAuthentication as e:
                    allowed = list(e.allowed_types)
                    continue
                except paramiko.AuthenticationException:
                    _emit(prompt, 'Permission denied, please try again.')
                    continue
            elif 'password' in allowed:
                password = prompt(f"{user}@{target['hostname']}'s password: ", False)
                try:
                    allowed = t.auth_password(user, password) or allowed
                except paramiko.PartialAuthentication as e:
                    allowed = list(e.allowed_types)
                    continue
                except paramiko.AuthenticationException:
                    _emit(prompt, 'Permission denied, please try again.')
                    continue
            else:
                break
        if not t.is_authenticated():
            raise paramiko.AuthenticationException(f'authentication failed ({", ".join(allowed)})')

    # -- commands ---------------------------------------------------------------
    def _exec(self, command: str, *, timeout: float, input: Optional[str] = None) -> tuple[int, bytes, bytes]:
        t = self.transport
        if t is None or not t.is_active():
            return 255, b'', b'not connected'
        with self._lock:
            chan = t.open_session(timeout=min(timeout, 30))
        chan.settimeout(timeout)
        out, err = bytearray(), bytearray()
        deadline = time.time() + timeout
        try:
            chan.exec_command(command)
            if input is not None:
                chan.sendall(input.encode())
            chan.shutdown_write()
            while True:
                got = False
                while chan.recv_ready():
                    out += chan.recv(65536); got = True
                while chan.recv_stderr_ready():
                    err += chan.recv_stderr(65536); got = True
                if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                    break
                if not got:
                    if time.time() > deadline:
                        chan.close()
                        raise subprocess.TimeoutExpired(command, timeout)
                    time.sleep(0.02)
            rc = chan.recv_exit_status()
        finally:
            chan.close()
        return rc, bytes(out), bytes(err)

    def run(self, command: str, *, timeout: float = 120, input: Optional[str] = None) -> subprocess.CompletedProcess:
        if self.closed:
            raise LoggedOut(f'{self.host}: logged out')
        rc, out, err = self._exec(command, timeout=timeout, input=input)
        return subprocess.CompletedProcess(command, rc, out.decode('utf-8', 'replace'), err.decode('utf-8', 'replace'))

    def read_from(self, path: str, offset: int) -> bytes:
        if self.closed:
            raise LoggedOut(f'{self.host}: logged out')
        try:
            rc, out, _ = self._exec(f'tail -c +{offset + 1} {shlex.quote(path)} 2>/dev/null', timeout=60)
        except subprocess.TimeoutExpired:
            return b''
        return out if rc == 0 else b''

    def login_session(self, on_output, on_exit):
        return LibraryLoginSession(self, on_output, on_exit)


def _emit(prompt, text: str) -> None:
    """Show a line without asking: a prompt call that returns immediately."""
    prompt(f'{text}\r\n', None)


class LibraryLoginSession:
    """The interactive authentication, driven from the page's terminal.

    Same interface as ``terminal.LoginSession``: ``command``, ``write``
    (keystrokes from the page), ``resize`` (ignored), ``close``, and the
    ``on_output``/``on_exit`` callbacks. Prompts are shown as the host sends
    them; the user's line is read with echo (or stars off for secrets), and
    backspace works.
    """

    def __init__(self, conn: LibraryConnection, on_output: Callable[[bytes], None], on_exit: Callable[[int], None]):
        self.conn = conn
        self.on_output, self.on_exit = on_output, on_exit
        self.command = f'ssh {conn.host}'
        self._keys: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name=f'login-{conn.host}')
        self._thread.start()

    def _prompt(self, text: str, echo) -> str:
        self.on_output(text.encode())
        if echo is None:                 # a message, no answer wanted
            return ''
        line = ''
        while True:
            ch = self._keys.get()
            if ch is None:
                raise AuthAborted('login closed')
            if ch in ('\r', '\n'):
                self.on_output(b'\r\n')
                return line
            if ch in ('\x7f', '\b'):
                if line:
                    line = line[:-1]
                    if echo:
                        self.on_output(b'\b \b')
                continue
            if ch == '\x03':
                raise AuthAborted('interrupted')
            if ch < ' ':
                continue
            line += ch
            if echo:
                self.on_output(ch.encode())

    def _run(self) -> None:
        rc = 0
        try:
            self.conn.reopen()
            self.conn.connect(self._prompt)
            self.on_output(b'\r\nConnected.\r\n')
        except AuthAborted as e:
            rc = 130
            self.on_output(f'\r\n{e}\r\n'.encode())
        except Exception as e:
            rc = 255
            self.conn.last_error = str(e)[:300]
            self.on_output(f'\r\n{type(e).__name__}: {e}\r\n'.encode())
        finally:
            self.on_exit(rc)

    def write(self, data: bytes) -> None:
        for ch in data.decode('utf-8', 'replace'):
            self._keys.put(ch)

    def resize(self, rows: int, cols: int) -> None:
        pass

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._keys.put(None)
