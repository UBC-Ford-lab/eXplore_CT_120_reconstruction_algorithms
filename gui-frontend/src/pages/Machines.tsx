import { Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api, Info, Machine, MachineConfig } from '../api'
import LoginTerminal from '../components/LoginTerminal'

function gib(x: number | null | undefined) { return x == null ? '?' : x.toFixed(1) }

const EMPTY: MachineConfig = { name: '', type: 'ssh', host: '', python: 'python', workdir: '~', scans_root: '', slots: 1, env: {}, sbatch_template: null, connection: 'auto' }

function MachineForm({ initial, onSaved, onCancel }: { initial: MachineConfig; onSaved: () => void; onCancel: () => void }) {
  const [m, setM] = useState<MachineConfig>(initial)
  const [envText, setEnvText] = useState(Object.entries(initial.env).map(([k, v]) => `${k}=${v}`).join('\n'))
  const [tpl, setTpl] = useState(initial.sbatch_template ?? '')
  const [error, setError] = useState<string | null>(null)
  const isNew = !initial.name
  useEffect(() => { if (m.type === 'slurm' && !tpl) api.defaultTemplate().then(setTpl) }, [m.type])
  async function save() {
    const env: Record<string, string> = {}
    for (const line of envText.split('\n')) { const i = line.indexOf('='); if (i > 0) env[line.slice(0, i).trim()] = line.slice(i + 1).trim() }
    try {
      await api.putMachine(m.name, { ...m, env, sbatch_template: m.type === 'slurm' ? tpl : null })
      onSaved()
    } catch (e: any) { setError(e.message) }
  }
  const f = (k: keyof MachineConfig) => ({ value: String(m[k] ?? ''), onChange: (e: any) => setM({ ...m, [k]: k === 'slots' ? Number(e.target.value) : e.target.value }) })
  return (
    <div className="panel">
      <h2>{isNew ? 'Add machine' : `Edit ${m.name}`}</h2>
      <div className="grid2">
        <label>Name <input type="text" {...f('name')} disabled={!isNew} placeholder="cedar" /></label>
        <label>Type
          <select {...f('type')}><option value="ssh">ssh (runs directly)</option><option value="slurm">slurm (sbatch)</option></select>
        </label>
        <label>SSH destination <input type="text" {...f('host')} placeholder="alias from ~/.ssh/config, or user@host" /></label>
        <label>Python on the machine <input type="text" {...f('python')} placeholder="~/venv/bin/python" /></label>
        <label>Working directory <input type="text" {...f('workdir')} placeholder="~/scratch/ct_recon" /></label>
        <label>Scans root <input type="text" {...f('scans_root')} placeholder="~/scratch/scans" /></label>
        {m.type === 'ssh' && <label>Concurrent jobs <input type="number" min={1} {...f('slots')} /></label>}
        <label>Connection
          <select value={m.connection ?? 'auto'} onChange={e => setM({ ...m, connection: e.target.value as MachineConfig['connection'] })}>
            <option value="auto">automatic (OpenSSH; the library on Windows)</option>
            <option value="openssh">OpenSSH client, multiplexed</option>
            <option value="library">library (paramiko), no ssh client needed</option>
          </select>
        </label>
        <label>Environment (NAME=value per line)<textarea rows={3} value={envText} onChange={e => setEnvText(e.target.value)} placeholder={'WANDB_PROJECT=my-project'} /></label>
      </div>
      {m.type === 'slurm' && (
        <label className="block">Slurm template <span className="muted small">placeholders: {'{job_name} {log} {workdir} {env} {command} {python} {input} {job_id}'}</span>
          <textarea className="mono" rows={16} value={tpl} onChange={e => setTpl(e.target.value)} />
        </label>
      )}
      <p className="note">Authentication is your own SSH setup: keys in ~/.ssh, or an interactive login from this page when the host asks for a password or a second factor. Nothing secret is stored.</p>
      <div className="toolbar">
        <button className="primary" onClick={save} disabled={!m.name || !m.host}>Save</button>
        <button onClick={onCancel}>Cancel</button>
        {error && <span className="error">{error}</span>}
      </div>
    </div>
  )
}

function MachineCard({ m, onChanged }: { m: Machine; onChanged: () => void }) {
  const [checks, setChecks] = useState<{ name: string; ok: boolean; detail: string }[] | null>(null)
  const [checking, setChecking] = useState(false)
  const [login, setLogin] = useState(false)
  const [editing, setEditing] = useState(false)
  const remote = m.kind !== 'local'
  async function check() { setChecking(true); try { setChecks((await api.checkMachine(m.name)).checks) } finally { setChecking(false) } }
  const [cancelling, setCancelling] = useState<string | null>(null)
  async function cancelRow(id: string, name: string) {
    if (!confirm(`scancel ${id} (${name}) on ${m.name}?`)) return
    setCancelling(id)
    try { await api.cancelQueueRow(m.name, id); onChanged() }
    catch (e: any) { alert(e.message) }
    finally { setCancelling(null) }
  }
  const [busy, setBusy] = useState(false)
  async function logout() {
    setBusy(true)
    try { await api.disconnect(m.name); onChanged() } catch (e: any) { alert(e.message) } finally { setBusy(false) }
  }
  async function connectAgain() {
    setBusy(true)
    try {
      const v = await api.connect(m.name)
      if (v.status === 'needs_login') setLogin(true)
      onChanged()
    } catch (e: any) { alert(e.message) } finally { setBusy(false) }
  }
  async function remove() { if (confirm(`Remove ${m.name}?`)) { await api.deleteMachine(m.name); onChanged() } }
  if (editing) return <MachineForm initial={m.config} onSaved={() => { setEditing(false); onChanged() }} onCancel={() => setEditing(false)} />
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{m.name}</h2><span className="tag">{m.kind}</span>
        <span className={`st ${m.status}`}><span className={`dot ${m.status}`} />{m.status.replace('_', ' ')}</span>
        {remote && <span className="faint small">{m.host}</span>}
        {remote && <span className="right">
          {m.status === 'connected' && <button onClick={check} disabled={checking}>{checking ? 'Checking…' : 'Check'}</button>}
          {m.status === 'connected' && <button onClick={logout} disabled={busy} title="drop the SSH connection; jobs on the machine keep running">Log out</button>}
          <button onClick={() => setEditing(true)}>Edit</button>
          <button className="quiet" onClick={remove} title="remove this machine">✕</button>
        </span>}
      </div>
      {m.error && <p className="error">{m.error}</p>}
      {m.status === 'connected' && (
        <div className="kv">
          <div>Host</div><div>{m.hostname}{m.platform ? ` · ${m.platform}` : ''}</div>
          <div>CPUs / memory</div><div>{m.cpus ?? '?'} · {gib(m.memory.available_gib)} of {gib(m.memory.total_gib)} GiB free</div>
          <div>Package</div><div>{m.package.version ?? <span className="error">not importable with {m.config.python}</span>}{m.package.commit ? ` (${m.package.commit})` : ''}{m.package.python ? ` · Python ${m.package.python}` : ''}</div>
          <div>Jobs run in</div><div className="mono">{m.cwd}</div>
          {m.kind === 'slurm' && m.queue && <><div>Your queue</div><div>{m.queue.running} running · {m.queue.pending} pending{m.queue.jobs.length > m.queue.running + m.queue.pending ? ` · ${m.queue.jobs.length - m.queue.running - m.queue.pending} other` : ''}</div></>}
          <div>Jobs from here</div><div>{m.running.length} running{m.pending.length ? `, ${m.pending.length} pending` : ''}{m.kind !== 'slurm' ? ` of ${m.slots} slots` : ''}</div>
        </div>
      )}
      {m.status === 'needs_login' && <p>The host asks for a password or a second factor. <button className="primary" onClick={() => setLogin(true)}>Connect</button></p>}
      {m.status === 'unreachable' && <p className="muted">Not reachable right now. <button onClick={() => setLogin(true)}>Try connecting</button></p>}
      {m.status === 'logged_out' && <p className="muted">Logged out. Jobs already on the machine keep running and are picked up again after connecting. <button className="primary" onClick={connectAgain} disabled={busy}>Connect</button></p>}
      {login && <LoginTerminal machine={m.name} onDone={() => { setLogin(false); onChanged() }} />}
      {m.status === 'connected' && m.gpus.length > 0 && (
        <table><thead><tr><th>GPU</th><th>Name</th><th>VRAM free</th><th>Utilisation</th></tr></thead>
          <tbody>{m.gpus.map(g => <tr key={g.index}><td>{g.index}</td><td>{g.name}</td><td>{gib(g.free_gib)} of {gib(g.total_gib)} GiB</td><td>{g.utilisation == null ? '' : `${g.utilisation.toFixed(0)} %`}</td></tr>)}</tbody></table>
      )}
      {m.status === 'connected' && m.kind === 'slurm' && m.queue && m.queue.jobs.length > 0 && (
        <div className="queue-wrap"><table className="queue">
          <thead><tr><th>Job</th><th>Name</th><th>State</th><th>Elapsed</th><th>Limit</th><th>Nodes</th><th>CPUs</th><th>Mem</th><th>GRES</th><th>Node / reason</th><th>Partition</th><th></th></tr></thead>
          <tbody>{m.queue.jobs.map(r => (
            <tr key={r.id} className={r.job ? 'ours' : ''}>
              <td className="mono">{r.job ? <Link to={`/jobs/${r.job}`} title="submitted from here; open the job">{r.id}</Link> : r.id}</td>
              <td className="clip" title={r.name}>{r.name}</td>
              <td><span className={`st ${r.state === 'RUNNING' ? 'running' : r.state === 'PENDING' ? 'pending' : 'queued'}`}><span className={`dot ${r.state === 'RUNNING' ? 'running' : r.state === 'PENDING' ? 'pending' : 'queued'}`} />{r.state.toLowerCase()}</span></td>
              <td className="num">{r.elapsed}</td><td className="num">{r.limit}</td><td className="num">{r.nodes}</td><td className="num">{r.cpus}</td><td className="num">{r.memory}</td>
              <td className="mono clip" title={r.gres}>{r.gres}</td><td className="clip" title={r.where}>{r.where}</td><td className="clip" title={r.partition}>{r.partition}</td>
              <td className="act"><button className="quiet" disabled={cancelling === r.id} title={`scancel ${r.id}`} onClick={() => cancelRow(r.id, r.name)}>{cancelling === r.id ? '…' : 'cancel'}</button></td>
            </tr>))}</tbody>
        </table></div>
      )}
      {m.status === 'connected' && m.kind === 'slurm' && m.queue && m.queue.jobs.length === 0 && <p className="muted small">Your scheduler queue is empty.</p>}
      {m.status === 'connected' && m.gpus.length === 0 && <p className="muted small">{m.kind === 'slurm' ? 'No GPU on the login node; jobs get theirs from the scheduler.' : 'No CUDA device visible. FDK runs on the CPU; the other families need a GPU.'}</p>}
      {checks && <ul className="checks">{checks.map(c => <li key={c.name} className={c.ok ? 'ok' : 'error'}>{c.ok ? '✓' : '✗'} <b>{c.name}</b> {c.detail}</li>)}</ul>}
    </section>
  )
}

export default function Machines({ info, reload }: { info: Info | null; reload: () => void }) {
  const [adding, setAdding] = useState(false)
  if (!info) return <p className="muted">Waiting for the server…</p>
  return (
    <>
      <div className="page-head">
        <h1>Machines</h1>
        <span className="note">read live from each machine · kept in <span className="mono">{info.data_dir}/machines.toml</span></span>
        {!adding && <button className="primary right" onClick={() => setAdding(true)}>Add machine</button>}
      </div>
      {adding && <MachineForm initial={EMPTY} onSaved={() => { setAdding(false); reload() }} onCancel={() => setAdding(false)} />}
      <div className="machines" style={{ marginTop: adding ? '0.8rem' : 0 }}>
        {info.executors.map(m => <MachineCard key={m.name} m={m} onChanged={reload} />)}
      </div>
    </>
  )
}
