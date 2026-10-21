import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, fmtDuration, fmtTime, Job } from '../api'

export function StatusTag({ s }: { s: Job['status'] }) { return <span className={`st ${s}`}><span className={`dot ${s}`} />{s}</span> }

const ACTIVE = ['queued', 'pending', 'running']

// Every job the server knows, past and present. The records live in the
// server's SQLite file and survive restarts; a running job is re-attached.
export default function Jobs() {
  const nav = useNavigate()
  const [jobs, setJobs] = useState<Job[]>([])
  const [status, setStatus] = useState('')
  const [machine, setMachine] = useState('')
  const [driver, setDriver] = useState('')
  const [text, setText] = useState('')
  const load = () => api.jobs().then(setJobs).catch(() => {})
  useEffect(() => { load(); const t = setInterval(load, 3000); return () => clearInterval(t) }, [])
  const machines = useMemo(() => [...new Set(jobs.map(j => j.executor))].sort(), [jobs])
  const drivers = useMemo(() => [...new Set(jobs.map(j => j.driver))].sort(), [jobs])
  const shown = jobs.filter(j =>
    (!status || (status === 'active' ? ACTIVE.includes(j.status) : j.status === status)) &&
    (!machine || j.executor === machine) && (!driver || j.driver === driver) &&
    (!text || `${j.input} ${j.note} ${j.algorithm ?? ''} ${j.output_path ?? ''}`.toLowerCase().includes(text.toLowerCase())))
  async function remove(e: React.MouseEvent, j: Job) {
    e.stopPropagation()
    if (!confirm(`Remove the record of ${j.driver} on ${j.input}? The run's files stay (use the job page to delete those too).`)) return
    await api.remove(j.id, false); load()
  }
  return (
    <>
      <div className="page-head">
        <h1>Jobs</h1>
        <span className="faint">{shown.length} of {jobs.length}</span>
        <select value={status} onChange={e => setStatus(e.target.value)}>
          <option value="">any status</option><option value="active">active</option><option value="finished">finished</option>
          <option value="failed">failed</option><option value="cancelled">cancelled</option>
        </select>
        <select value={machine} onChange={e => setMachine(e.target.value)}><option value="">any machine</option>{machines.map(m => <option key={m}>{m}</option>)}</select>
        <select value={driver} onChange={e => setDriver(e.target.value)}><option value="">any driver</option>{drivers.map(d => <option key={d}>{d}</option>)}</select>
        <input type="search" placeholder="scan, note, output…" value={text} onChange={e => setText(e.target.value)} />
      </div>
      {jobs.length === 0 && <p className="muted">Nothing yet. <Link to="/new">Start one.</Link></p>}
      {shown.length > 0 && (
        <table className="jobs">
          <thead><tr><th>Status</th><th>Driver</th><th>Input</th><th>Machine</th><th>Started</th><th className="num">Duration</th><th>Output</th><th>Note</th><th></th></tr></thead>
          <tbody>
            {shown.map(j => (
              <tr key={j.id} className="link" onClick={() => nav(`/jobs/${j.id}`)}>
                <td><StatusTag s={j.status} /></td>
                <td>{j.driver}{j.algorithm ? <span className="muted"> / {j.algorithm}</span> : ''}</td>
                <td className="mono clip" title={j.input}>{j.input.split('/').pop()}</td>
                <td>{j.executor}{j.node ? ` · ${j.node}` : ''}{j.gpu_index != null ? ` · GPU ${j.gpu_index}` : ''}{j.remote_state && j.status === 'pending' ? ` · ${j.remote_state}` : ''}</td>
                <td className="faint">{fmtTime(j.started ?? j.created)}</td>
                <td className="num">{fmtDuration(j.duration)}{j.status === 'running' && j.progress?.fraction != null && <span className="faint"> · {Math.round(j.progress.fraction * 100)} %</span>}</td>
                <td className="mono clip small" title={j.output_path ?? ''}>{j.output_path ? j.output_path.split('/').pop() : ''}</td>
                <td className="clip">{j.note}</td>
                <td>{!ACTIVE.includes(j.status) && <button className="quiet" onClick={e => remove(e, j)} title="remove the record; files stay">✕</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  )
}
