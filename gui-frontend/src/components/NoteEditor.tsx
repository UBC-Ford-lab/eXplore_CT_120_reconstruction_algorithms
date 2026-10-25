import { useEffect, useRef, useState } from 'react'
import { api, Job } from '../api'

// A job's note, editable in place at any time: before the run starts, while
// it runs, or long after. Click the text (or the pencil that appears on
// hover) to edit; Enter saves, Escape cancels, blur saves too. The note is
// display only and never reaches the run, so editing a finished job is safe.
export default function NoteEditor({ job, onSaved, block = false }: { job: Job; onSaved: (j: Job) => void; block?: boolean }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(job.note)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => { if (editing) input.current?.select() }, [editing])

  function start(e: React.SyntheticEvent) { e.stopPropagation(); setDraft(job.note); setError(null); setEditing(true) }
  async function save() {
    const note = draft.trim()
    setEditing(false)
    if (note === job.note) return
    setBusy(true)
    try { onSaved(await api.setNote(job.id, note)) } catch (e: any) { setError(e.message) } finally { setBusy(false) }
  }
  if (editing) {
    return (
      <input ref={input} className="note-input" type="text" value={draft} placeholder="note" maxLength={500}
        onClick={e => e.stopPropagation()} onChange={e => setDraft(e.target.value)} onBlur={save}
        onKeyDown={e => { if (e.key === 'Enter') save(); else if (e.key === 'Escape') { setDraft(job.note); setEditing(false) } }} />
    )
  }
  return (
    <span className={`note-cell${block ? ' block' : ''}${busy ? ' busy' : ''}`} onClick={start} title={job.note ? 'click to edit the note' : 'click to add a note'}>
      {job.note ? <span className="note-text">{job.note}</span> : <span className="note-empty">add a note…</span>}
      <span className="note-pen" aria-hidden>✎</span>
      {error && <span className="error"> {error}</span>}
    </span>
  )
}
