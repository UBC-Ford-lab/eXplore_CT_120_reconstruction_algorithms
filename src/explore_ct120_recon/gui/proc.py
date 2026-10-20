"""subprocess.run without a console window on Windows.

The server started by ``ct120-gui`` (or the desktop icon) has no console, so
every console program it runs, nvidia-smi, git, taskkill, ssh, would open
and close a window of its own. This wrapper adds CREATE_NO_WINDOW there and
is a plain subprocess.run elsewhere.
"""
from __future__ import annotations

import subprocess
import sys

NO_WINDOW = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}


def run(*args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(*args, **{**NO_WINDOW, **kwargs})
