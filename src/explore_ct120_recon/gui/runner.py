"""Wrapper a local job is started through.

``python -m explore_ct120_recon.gui.runner EXIT_FILE -- COMMAND...`` runs the
command and writes its exit code to EXIT_FILE. The server can then be
restarted while a job runs: the job keeps going in its own session, and the
new server finds the exit code on disk instead of needing to be the parent.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[1] != '--':
        raise SystemExit('usage: runner EXIT_FILE -- COMMAND...')
    exit_file, command = argv[0], argv[2:]
    # explicit handles: on Windows a child started without redirection does
    # not inherit the wrapper's redirected stdout (the log file) but attaches
    # to the hidden console, and the log stays empty.
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                            stdout=sys.stdout, stderr=sys.stderr)

    def forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    # the child is in the same process group, so a group signal (or Ctrl-Break
    # on Windows) already reaches it; forwarding covers a signal sent to the
    # wrapper alone. SIGHUP and SIGBREAK each exist on one platform only.
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP', 'SIGBREAK'):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, forward)
            except (OSError, ValueError):
                pass
    rc = proc.wait()
    tmp = exit_file + '.tmp'
    with open(tmp, 'w') as f:
        f.write(str(rc))
    os.replace(tmp, exit_file)
    return rc


if __name__ == '__main__':
    sys.exit(main())
