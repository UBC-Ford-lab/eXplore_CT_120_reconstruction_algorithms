"""The machines a job can run on, as configuration.

``machines.toml`` in the GUI data directory holds one entry per machine. The
local machine is implicit and always present. Nothing here is specific to
one lab: a user adds a host alias, a Python interpreter and a working
directory, and for a cluster a Slurm template. No password or key is ever
stored; SSH authentication is the user's ``~/.ssh`` setup.

Example::

    [machines.gpu-box]
    type = "ssh"
    host = "gpu-box"                 # as you would type after `ssh`
    python = "~/venv/bin/python"
    workdir = "~/ct_recon"
    scans_root = "~/ct_recon/data/scans"
    slots = 1
    # connection = "library"      # paramiko instead of the OpenSSH client; the default on Windows

    [machines.cluster]
    type = "slurm"
    host = "me@cluster.example.org"
    python = "~/venv/bin/python"
    workdir = "~/scratch/ct_recon"
    scans_root = "~/scratch/scans"
    [machines.cluster.env]
    WANDB_PROJECT = "my-project"

The Slurm template of a machine lives next to the file as
``templates/<name>.sbatch``; the package default is used until one is saved.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

TYPES = ('local', 'ssh', 'slurm')
DEFAULT_TEMPLATE = Path(__file__).with_name('templates') / 'slurm_default.sbatch'
CONNECTIONS = ('auto', 'openssh', 'library')


@dataclass
class Machine:
    name: str
    type: str = 'ssh'
    host: str = ''                 # ssh destination: alias, host or user@host
    python: str = 'python'         # interpreter with the package installed, on the machine
    workdir: str = '~'             # jobs run here; relative paths resolve here
    scans_root: str = ''           # subfolders are scans, offered in the form
    slots: int = 1                 # concurrent jobs (ssh); ignored for slurm
    env: dict = field(default_factory=dict)   # exported into every job
    sbatch_template: Optional[str] = None     # slurm only; None = package default
    connection: str = 'auto'       # 'openssh' (multiplexed master), 'library' (paramiko), 'auto' = by platform

    def validate(self) -> None:
        if not self.name or not self.name.replace('-', '').replace('_', '').isalnum():
            raise ValueError('machine name: letters, digits, - and _ only')
        if self.type not in TYPES:
            raise ValueError(f'type must be one of {TYPES}')
        if self.type != 'local' and not self.host:
            raise ValueError('host is required for a remote machine')
        if self.type == 'local' and self.name != 'local':
            raise ValueError("only the implicit machine is of type 'local'")
        self.slots = max(1, int(self.slots))
        if self.connection not in CONNECTIONS:
            raise ValueError(f'connection must be one of {CONNECTIONS}')
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.env.items()):
            raise ValueError('env must map names to strings')

    def template_text(self) -> str:
        if self.sbatch_template is not None:
            return self.sbatch_template
        return DEFAULT_TEMPLATE.read_text()

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.type == 'slurm':
            d['sbatch_template'] = self.template_text()
            d['template_is_default'] = self.sbatch_template is None
        return d


def _toml_str(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'


class MachineRegistry:
    """Reads and writes ``machines.toml`` plus the per-machine templates."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / 'machines.toml'
        self.templates = Path(data_dir) / 'templates'
        self.templates.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Machine]:
        out: dict[str, Machine] = {'local': Machine('local', 'local')}
        if not self.path.exists():
            return out
        with open(self.path, 'rb') as f:
            doc = tomllib.load(f)
        for name, body in (doc.get('machines') or {}).items():
            m = Machine(name=name, **{k: v for k, v in body.items() if k in Machine.__dataclass_fields__ and k != 'name'})
            tpl = self.templates / f'{name}.sbatch'
            if tpl.exists():
                m.sbatch_template = tpl.read_text()
            m.validate()
            out[name] = m
        return out

    def save(self, machines: dict[str, Machine]) -> None:
        lines = ['# Machines known to ct120-gui. Edit here or on the Machines page.', '']
        for m in machines.values():
            if m.type == 'local':
                continue
            lines.append(f'[machines.{m.name}]')
            for k in ('type', 'host', 'python', 'workdir', 'scans_root'):
                lines.append(f'{k} = {_toml_str(getattr(m, k))}')
            lines.append(f'slots = {int(m.slots)}')
            if m.connection != 'auto':
                lines.append(f'connection = {_toml_str(m.connection)}')
            if m.env:
                lines.append(f'[machines.{m.name}.env]')
                for k, v in m.env.items():
                    lines.append(f'{k} = {_toml_str(v)}')
            lines.append('')
            tpl = self.templates / f'{m.name}.sbatch'
            if m.type == 'slurm' and m.sbatch_template is not None:
                tpl.write_text(m.sbatch_template)
            elif tpl.exists():
                tpl.unlink()
        self.path.write_text('\n'.join(lines))

    def put(self, m: Machine) -> dict[str, Machine]:
        m.validate()
        machines = self.load()
        machines[m.name] = m
        self.save(machines)
        return machines

    def remove(self, name: str) -> dict[str, Machine]:
        machines = self.load()
        if name == 'local':
            raise ValueError('the local machine cannot be removed')
        machines.pop(name, None)
        tpl = self.templates / f'{name}.sbatch'
        if tpl.exists():
            tpl.unlink()
        self.save(machines)
        return machines
