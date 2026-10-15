"""Logging in to a machine from the page.

The multiplexed SSH master is opened in a pseudo-terminal owned by the
server, and the terminal is relayed to the browser over a websocket. Whatever
the host asks for (password, Duo passcode or push, host-key confirmation) the
user answers in the page; the text goes straight to ssh and is never stored.
``ssh -fNM`` backgrounds itself once authenticated, so the master outlives
this session and the server, until ``ControlPersist`` expires.
"""
from __future__ import annotations

import fcntl
import os
import pty
import select
import shlex
import struct
import subprocess
import termios
import threading
from typing import Callable, Optional

from .remote import SshConnection


class LoginSession:
    """One interactive ``ssh -fNM`` in a pty; ``on_output`` gets bytes."""

    def __init__(self, conn: SshConnection, on_output: Callable[[bytes], None], on_exit: Callable[[int], None]):
        self.conn = conn
        self.on_output, self.on_exit = on_output, on_exit
        self.master_fd, slave_fd = pty.openpty()
        self._set_size(slave_fd, 24, 100)
        cmd = ['ssh', '-fNM', '-o', f'ControlPath={conn.socket}', '-o', f'ControlPersist={conn.persist}',
               '-o', 'ConnectTimeout=20', '-o', 'StrictHostKeyChecking=ask', conn.host]
        self.command = shlex.join(cmd)

        def _make_controlling():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        self.proc = subprocess.Popen(cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                                     preexec_fn=_make_controlling, close_fds=True)
        os.close(slave_fd)
        self._reader = threading.Thread(target=self._pump, daemon=True, name=f'login-{conn.host}')
        self._reader.start()

    def _set_size(self, fd: int, rows: int, cols: int) -> None:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))

    def resize(self, rows: int, cols: int) -> None:
        try:
            self._set_size(self.master_fd, rows, cols)
        except OSError:
            pass

    def _pump(self) -> None:
        try:
            while True:
                r, _, _ = select.select([self.master_fd], [], [], 0.5)
                if r:
                    try:
                        data = os.read(self.master_fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    self.on_output(data)
                elif self.proc.poll() is not None:
                    # drain what is left
                    try:
                        while True:
                            r, _, _ = select.select([self.master_fd], [], [], 0.1)
                            if not r:
                                break
                            data = os.read(self.master_fd, 4096)
                            if not data:
                                break
                            self.on_output(data)
                    except OSError:
                        pass
                    break
        finally:
            rc = self.proc.wait()
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.on_exit(rc)

    def write(self, data: bytes) -> None:
        try:
            os.write(self.master_fd, data)
        except OSError:
            pass

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
