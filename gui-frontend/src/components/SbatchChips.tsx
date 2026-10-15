import { useEffect, useState } from 'react'
import type { Directive } from '../api'

// Common sbatch options shown even when the template leaves them to the scheduler's defaults.
const COMMON = ['nodes', 'ntasks', 'gpus-per-node', 'cpus-per-task', 'mem', 'time', 'partition', 'account']

/** Split the overrides field into tokens; values with spaces are not supported (sbatch would need quoting). */
function tokens(resources: string): string[] { return resources.trim() ? resources.trim().split(/\s+/) : [] }

/** The overrides field with ``--key=value`` set (value null removes the override). */
export function withOverride(resources: string, key: string, value: string | null): string {
  const toks = tokens(resources)
  const out: string[] = []
  let done = false
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]
    if (!t.startsWith('-')) { out.push(t); continue }
    const hasEq = t.includes('=')
    const k = t.split('=')[0].replace(/^-+/, '')
    const span = !hasEq && i + 1 < toks.length && !toks[i + 1].startsWith('-') ? 2 : 1
    if (k === key) {
      if (value != null && !done) { out.push(`--${key}=${value}`); done = true }
      i += span - 1
      continue
    }
    out.push(...toks.slice(i, i + span))
    i += span - 1
  }
  if (value != null && !done) out.push(`--${key}=${value}`)
  return out.join(' ')
}

function Chip({ d, templateValue, onSet }: { d: Directive & { unset?: boolean }; templateValue: string | null | undefined; onSet: (v: string | null) => void }) {
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState(d.value ?? '')
  useEffect(() => { if (!editing) setText(d.value ?? '') }, [d.value, editing])
  const editable = d.value != null || d.unset
  function commit() {
    setEditing(false)
    const v = text.trim()
    if (v === (d.value ?? '')) return
    // typing the template's own value back (or clearing) drops the override
    onSet(v === '' || v === templateValue ? null : v)
  }
  const cls = 'chip' + (d.source === 'override' ? ' override' : '') + (d.unset ? ' unset' : '') + (editable ? ' editable' : '')
  const title = d.unset ? `${d.key}: not set; the scheduler's default. Click to set it for this job only`
    : `${d.key}=${d.value ?? ''} · ${d.source === 'override' ? 'set for this job only (overrides field)' : 'from the machine\'s template; click to change it for this job only'}`
  return (
    <span className={cls} title={title} onClick={() => editable && !editing && setEditing(true)}>
      <span className="k">{d.key}</span>
      {editing ? (
        <input autoFocus value={text} size={Math.max(3, text.length + 1)} onChange={e => setText(e.target.value)}
          onBlur={commit} onKeyDown={e => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') { setText(d.value ?? ''); setEditing(false) } }} />
      ) : d.value != null ? <span className="v">{d.value}</span> : d.unset ? <span className="v faint">–</span> : null}
      {d.source === 'override' && !editing && (
        <button className="x" title="back to the template's value" onClick={e => { e.stopPropagation(); onSet(null) }}>×</button>
      )}
    </span>
  )
}

export default function SbatchChips({ directives, templateDirectives, resources, setResources }: {
  directives: Directive[]; templateDirectives: Directive[]; resources: string; setResources: (s: string) => void
}) {
  const template = new Map(templateDirectives.map(d => [d.key, d.value]))
  const shown: (Directive & { unset?: boolean })[] = [...directives]
  for (const k of COMMON) if (!shown.some(d => d.key === k)) shown.push({ key: k, value: null, source: 'template', unset: true })
  return (
    <>
      {shown.map(d => <Chip key={d.key} d={d} templateValue={template.get(d.key)} onSet={v => setResources(withOverride(resources, d.key, v))} />)}
    </>
  )
}
