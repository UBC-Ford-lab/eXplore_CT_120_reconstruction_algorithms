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
    proc = subprocess.Popen(command)

    def forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    rc = proc.wait()
    tmp = exit_file + '.tmp'
    with open(tmp, 'w') as f:
        f.write(str(rc))
    os.replace(tmp, exit_file)
    return rc


if __name__ == '__main__':
    sys.exit(main())
