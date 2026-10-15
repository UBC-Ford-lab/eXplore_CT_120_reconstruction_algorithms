import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import DeleteJob from '../components/DeleteJob'
import { api, fmtDuration, fmtTime, Job } from '../api'
import { StatusTag } from './Jobs'

export default function JobDetail() {
  const { id } = useParams()
  const nav = useNavigate()
  const [job, setJob] = useState<Job | null>(null)
  const [log, setLog] = useState('')
  const [error, setError] = useState<string | null>(null)
  const offset = useRef(0)
  const pre = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (!id) return
    offset.current = 0; setLog('')
    let stop = false
    const tick = async () => {
      try {
        const j = await api.job(id); if (stop) return
        setJob(j)
        const chunk = await api.log(id, offset.current); if (stop) return
        if (chunk.size < offset.current) { offset.current = 0; setLog('') }
        if (chunk.text) { offset.current = chunk.offset; setLog(l => l + chunk.text) }
      } catch (e: any) { setError(e.message) }
    }
    tick()
    const t = setInterval(tick, 2000)
    return () => { stop = true; clearInterval(t) }
  }, [id])

  useEffect(() => { if (pre.current) pre.current.scrollTop = pre.current.scrollHeight }, [log])

  if (error) return <p className="error">{error}</p>
  if (!job) return <p className="muted">Loading…</p>
  const active = job.status === 'queued' || job.status === 'pending' || job.status === 'running'
  const changed = Object.entries(job.values).filter(([, v]) => v !== null && v !== '' && !(Array.isArray(v) && v.length === 0))
  return (
    <>
      <div className="page-head">
        <Link to="/jobs" className="faint">Jobs</Link><span className="faint">/</span>
        <h1>{job.driver}{job.algorithm ? ` / ${job.algorithm}` : ''}</h1>
        <StatusTag s={job.status} />
        {job.duration != null && <span className="faint">{fmtDuration(job.duration)}{active ? ' so far' : ''}</span>}
        <div className="right actions" style={{ margin: 0 }}>
          {active && <button onClick={() => api.cancel(job.id).then(setJob).catch(e => setError(e.message))}>Cancel</button>}
          {!active && <DeleteJob job={job} onDeleted={() => nav('/jobs')} />}
        </div>
      </div>
      <div className="kv2">
        <div>Input</div><div className="mono span">{job.input}</div>
        <div>Machine</div><div>{job.executor}{job.hostname ? ` · ${job.hostname}` : ''}{job.node ? ` · node ${job.node}` : ''}{job.gpu_name ? ` · GPU ${job.gpu_index}: ${job.gpu_name}` : ''}</div>
        <div>Created</div><div>{fmtTime(job.created)}</div>
        <div>Started</div><div>{fmtTime(job.started)}</div>
        {job.finished && <><div>Finished</div><div>{fmtTime(job.finished)}{job.returncode != null ? ` · exit ${job.returncode}` : ''}</div></>}
        {job.remote_id && <><div>Remote id</div><div>{job.remote_id}{job.remote_state ? ` · ${job.remote_state}` : ''}</div></>}
        {job.resources && <><div>sbatch</div><div className="mono">{job.resources}</div></>}
        {job.package_version && <><div>Package</div><div>{job.package_version}</div></>}
        {job.output_path && <><div>Output</div><div className="mono span">{job.output_path}</div></>}
        {job.wandb_url && <><div>W&amp;B</div><div className="span"><a href={job.wandb_url} target="_blank" rel="noreferrer">{job.wandb_url}</a></div></>}
        {job.note && <><div>Note</div><div className="span">{job.note}</div></>}
        {job.error && <><div>Error</div><div className="error span">{job.error}</div></>}
        <div>Command</div><div className="mono wrap span">{job.shell ?? job.command.join(' ')}</div>
        <div>Log file</div><div className="mono span">{job.log_path ?? ''}</div>
      </div>
      <details>
        <summary>Submitted values ({changed.length})</summary>
        <table><tbody>{changed.map(([k, v]) => <tr key={k}><td className="mono">{k}</td><td className="mono">{JSON.stringify(v)}</td></tr>)}</tbody></table>
      </details>
      <pre ref={pre} className="log">{log || (active ? 'waiting for output…' : '(empty)')}</pre>
    </>
  )
}
