import { useEffect, useState } from 'react'
import { api, fmtBytes, Job } from '../api'

// Delete a finished job's record, optionally with the files the run wrote.
// The file list comes from the server by the run's own naming rule and is
// shown before anything is removed.
export default function DeleteJob({ job, onDeleted }: { job: Job; onDeleted: () => void }) {
  const [open, setOpen] = useState(false)
  const [files, setFiles] = useState<{ path: string; dir: boolean; bytes: number }[] | null>(null)
  const [withData, setWithData] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (!open) return
    setFiles(null)
    api.files(job.id).then(r => setFiles(r.files)).catch(e => { setError(e.message); setFiles([]) })
  }, [open, job.id])
  const total = (files ?? []).reduce((n, f) => n + f.bytes, 0)
  async function go() {
    setBusy(true); setError(null)
    try { await api.remove(job.id, withData); onDeleted() } catch (e: any) { setError(e.message) } finally { setBusy(false) }
  }
  if (!open) return <button onClick={() => setOpen(true)}>Delete…</button>
  return (
    <div className="panel danger" style={{ minWidth: "28rem" }}>
      <h3>Delete this job</h3>
      <label className="inline"><input type="radio" checked={!withData} onChange={() => setWithData(false)} /> record and log only; the run's files stay</label>
      <label className="inline"><input type="radio" checked={withData} onChange={() => setWithData(true)} /> record, log and the run's files</label>
      {files === null && <p className="muted small">listing files…</p>}
      {files !== null && (
        <>
          {files.length === 0 && <p className="muted small">No files found for <span className="mono">{job.output_path ?? '(no output path recorded)'}</span>.</p>}
          {files.length > 0 && (
            <table><tbody>
              {files.map(f => <tr key={f.path} className={withData ? 'error' : 'muted'}><td className="mono">{f.path}{f.dir ? '/' : ''}</td><td>{fmtBytes(f.bytes)}</td></tr>)}
              <tr><td className="muted">total</td><td>{fmtBytes(total)}</td></tr>
            </tbody></table>
          )}
        </>
      )}
      <div className="toolbar">
        <button className="danger" disabled={busy || files === null} onClick={go}>{withData ? `Delete record and ${files?.length ?? 0} file${files?.length === 1 ? '' : 's'}` : 'Delete record'}</button>
        <button onClick={() => setOpen(false)}>Cancel</button>
        {error && <span className="error">{error}</span>}
      </div>
    </div>
  )
}
