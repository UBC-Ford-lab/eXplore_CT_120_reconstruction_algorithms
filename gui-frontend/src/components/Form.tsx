import { useState, type ReactNode } from 'react'
import { Field, Schema } from '../api'
import Browser, { BrowseButton, browseModeFor } from './Browser'

export type Values = Record<string, any>

export function defaultsOf(schema: Schema): Values {
  const v: Values = {}
  for (const f of schema.fields) v[f.dest] = f.positional ? '' : f.default
  return v
}

// Any field that names a scan folder offers the machine's scans, wherever it sits in the form.
const SCAN_FOLDER_DESTS = new Set(['data_folder', 'template', 'scan_folder'])

function FieldInput({ f, value, onChange, scans, machine }: { f: Field; value: any; onChange: (v: any) => void; scans: { name: string; path: string }[]; machine: string }) {
  const id = `f-${f.dest}`
  const mode = (f.kind === 'text' || f.kind === 'list') ? browseModeFor(f.dest, f.metavar) : null
  const [browsing, setBrowsing] = useState(false)
  const browser = mode && browsing ? (
    <Browser machine={machine} mode={mode} title={f.positional ? f.dest : f.flags[0]}
      initial={f.kind === 'list' ? (Array.isArray(value) && value.length ? value[value.length - 1] : '') : (value ?? '')}
      onClose={() => setBrowsing(false)}
      onPick={p => { setBrowsing(false); onChange(f.kind === 'list' ? [...(Array.isArray(value) ? value : []), p] : p) }} />
  ) : null
  const button = mode ? <BrowseButton onClick={() => setBrowsing(true)} title={mode === 'dir' ? 'choose a folder' : mode === 'file' ? 'choose a file' : 'choose where to write'} /> : null
  if (f.kind === 'flag') {
    return <input id={id} type="checkbox" checked={!!value} onChange={e => onChange(e.target.checked)} />
  }
  if (f.kind === 'choice') {
    return (
      <select id={id} value={value ?? ''} onChange={e => onChange(e.target.value)}>
        {value == null && <option value="">(none)</option>}
        {f.choices!.map(c => <option key={String(c)} value={String(c)}>{String(c)}</option>)}
      </select>
    )
  }
  if (f.kind === 'list') {
    const text = Array.isArray(value) ? value.join(' ') : (value ?? '')
    return <>
      <input id={id} type="text" value={text} placeholder="space separated"
        onChange={e => onChange(e.target.value.trim() === '' ? [] : e.target.value.split(/\s+/))} />
      {button}{browser}
    </>
  }
  if (SCAN_FOLDER_DESTS.has(f.dest) && scans.length > 0) {
    return (
      <>
        <input id={id} type="text" list="scan-list" value={value ?? ''} onChange={e => onChange(e.target.value)} />
        <datalist id="scan-list">{scans.map(s => <option key={s.path} value={s.path}>{s.name}</option>)}</datalist>
        {button}{browser}
      </>
    )
  }
  const numeric = f.kind === 'integer' || f.kind === 'number'
  return <>
    <input id={id} type={numeric ? 'number' : 'text'} step={f.kind === 'number' ? 'any' : undefined}
      value={value ?? ''} placeholder={f.default == null ? '' : String(f.default)}
      onChange={e => onChange(e.target.value === '' ? null : (numeric ? Number(e.target.value) : e.target.value))} />
    {button}{browser}
  </>
}

export default function Form({ schema, values, setValues, scans, showAll, afterInput, machine }: {
  schema: Schema; values: Values; setValues: (v: Values) => void
  scans: { name: string; path: string }[]; showAll: boolean
  afterInput?: ReactNode   // extra rows shown at the end of the Input group (the job note)
  machine: string          // whose file system the browse buttons show
}) {
  const groups = new Map<string, Field[]>()
  for (const f of schema.fields) {
    if (f.dest === 'output' || f.dest === 'out' || f.dest === 'report') continue   // shown in the run panel
    const key = f.positional ? 'Input' : (f.group || 'options')
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(f)
  }
  const changed = (f: Field) => JSON.stringify(values[f.dest]) !== JSON.stringify(f.positional ? '' : f.default)
  return (
    <div className="form">
      {[...groups.entries()].map(([title, fields]) => {
        const visible = showAll ? fields : fields.filter(f => f.positional || f.required || changed(f))
        if (visible.length === 0) return null
        const nChanged = fields.filter(f => !f.positional && changed(f)).length
        const name = title === 'options' ? 'Options' : title === 'positional arguments' ? 'Input' : title
        return (
          <details className="group" key={title} open>
            <summary>{name}<span className="n">{visible.length} of {fields.length}{nChanged ? ` · ${nChanged} changed` : ''}</span></summary>
            {visible.map(f => (
              <div className={`opt ${f.positional ? 'pos' : ''} ${changed(f) ? 'changed' : ''}`} key={f.dest} title={f.help}>
                <label htmlFor={`f-${f.dest}`} className="flag">
                  {f.positional ? f.dest : f.flags[0]}
                  {f.required && <span className="req">req</span>}
                </label>
                <div className="input">
                  <FieldInput f={f} value={values[f.dest]} scans={scans} machine={machine} onChange={v => setValues({ ...values, [f.dest]: v })} />
                  {f.kind === 'flag' && f.off && <span className="faint small">off: <span className="mono">{f.off}</span></span>}
                </div>
                <div className="help">{f.help}</div>
              </div>
            ))}
            {title === 'Input' && afterInput}
          </details>
        )
      })}
    </div>
  )
}
