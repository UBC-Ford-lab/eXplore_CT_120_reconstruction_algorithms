"""Small shared system utilities for the reconstruction backends."""

from __future__ import annotations

import os
import subprocess
from typing import Optional


def resolve_visible_gpu(gpu_index: int = 0) -> Optional[str]:
    """Map a logical CUDA device index to the id nvidia-smi expects.

    ``gpu_index`` is a *logical* index — the one torch uses, relative to
    ``CUDA_VISIBLE_DEVICES``. nvidia-smi's ``--id`` is a *physical* index, so
    on a shared node where the scheduler handed out a subset (e.g.
    ``CUDA_VISIBLE_DEVICES=2``), logical 0 is physical 2 and querying
    ``--id=0`` reports the wrong card. Under cgroup-isolated allocations (the
    usual SLURM case) only the allocated GPUs are visible at all and the
    mapping is the identity, so this is a no-op there.

    Returns the id string to pass to ``nvidia-smi --id=`` (an index, or a
    ``GPU-<uuid>`` / ``MIG-<uuid>`` string — nvidia-smi accepts both), or None
    when ``CUDA_VISIBLE_DEVICES`` is set but empty (no GPU is usable).
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return str(int(gpu_index))
    entries = [e.strip() for e in visible.split(',') if e.strip()]
    if not entries:
        return None                       # explicitly no visible devices
    if not 0 <= int(gpu_index) < len(entries):
        return str(int(gpu_index))        # out of range — let nvidia-smi fail
    return entries[int(gpu_index)]


def _torch_gpu_memory(gpu_index: int = 0) -> Optional[dict]:
    """The same record from CUDA itself (creates a context, ~300-500 MB);
    the fallback when nvidia-smi cannot answer for the visible device."""
    try:
        import torch
        if not torch.cuda.is_available() or int(gpu_index) >= torch.cuda.device_count():
            import sys
            print(f"  GPU probe: torch sees {torch.cuda.device_count()} CUDA device(s) "
                  f"(available={torch.cuda.is_available()}); CUDA_VISIBLE_DEVICES="
                  f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}", file=sys.stderr)
            return None
        free, total = torch.cuda.mem_get_info(int(gpu_index))
        return {'name': torch.cuda.get_device_name(int(gpu_index)),
                'total_bytes': int(total), 'free_bytes': int(free)}
    except Exception as exc:                  # say why, so a broken node is diagnosable
        import sys
        print(f"  GPU probe: CUDA query failed ({type(exc).__name__}: {exc}); "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", file=sys.stderr)
        return None


def query_gpu_memory(gpu_index: int = 0) -> Optional[dict]:
    """Query GPU name and memory via nvidia-smi.

    Uses nvidia-smi directly (not torch) so backends that avoid initializing
    a CUDA context (~300-500 MB overhead) can still report memory. Returns
    ``{'name': str, 'total_bytes': int, 'free_bytes': int}`` or None when
    nvidia-smi is unavailable, times out, or no device is visible.

    ``gpu_index`` is interpreted the way torch interprets it (relative to
    CUDA_VISIBLE_DEVICES) — see :func:`resolve_visible_gpu`.
    """
    device_id = resolve_visible_gpu(gpu_index)
    if device_id is None:
        return None
    if str(device_id).startswith('MIG-'):
        # A MIG slice: nvidia-smi's --query-gpu rejects MIG ids (and would
        # report the whole card's memory anyway), so ask CUDA, which sees
        # exactly the slice. Found on fir's h100 MIG partitions 2026-10-26.
        return _torch_gpu_memory(gpu_index)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
             '--format=csv,noheader,nounits', f'--id={device_id}'],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _torch_gpu_memory(gpu_index)
    if result.returncode != 0:
        return _torch_gpu_memory(gpu_index)
    parts = result.stdout.strip().split(', ')
    return {
        'name': parts[0],
        'total_bytes': int(parts[1]) * 2**20,  # MiB -> bytes
        'free_bytes': int(parts[2]) * 2**20,
    }


def gpu_holders(gpu_index: int = 0) -> list:
    """The OTHER processes holding memory on the visible GPU, from nvidia-smi:
    ``[{'pid': int, 'name': str, 'bytes': int | None}, ...]`` (``bytes`` is
    None where the driver does not report per-process use, e.g. Windows
    WDDM). Empty when nvidia-smi cannot answer. The preflight prints this
    beside a short free-VRAM figure so a card that is full because an
    earlier job is still alive reads as exactly that, not as a small card.
    """
    device_id = resolve_visible_gpu(gpu_index)
    if device_id is None or str(device_id).startswith('MIG-'):
        return []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader,nounits', f'--id={device_id}'],
            capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    out = []
    for line in result.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            nbytes = int(float(parts[2])) * 2**20
        except ValueError:
            nbytes = None                     # '[N/A]' under WDDM
        out.append({'pid': pid, 'name': parts[1], 'bytes': nbytes})
    return out
