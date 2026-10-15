import { useEffect, useRef, useState } from 'react'
import { api, DirListing } from '../api'

export type BrowseMode = 'dir' | 'file' | 'save'

/** A modal directory browser on one machine.
 *  dir: pick a folder · file: pick a file · save: pick a folder and type a name */
export default function Browser({ machine, mode, initial, title, onPick, onClose }: {
  machine: string; mode: BrowseMode; initial: string; title: string
  onPick: (path: string) => void; onClose: () => void
}) {
  const [listing, setListing] = useState<DirListing | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [typed, setTyped] = useState(initial)
  const [name, setName] = useState(() => mode === 'save' ? (initial.split('/').pop() ?? '') : '')
  const [hidden, setHidden] = useState(false)
  const [filter, setFilter] = useState('')
  const box = useRef<HTMLDivElement>(null)

  function go(path: string | null) {
    setError(null)
    api.browse(machine, path, hidden).then(l => { setListing(l); setTyped(l.path); if (l.error) setError(l.error) })
      .catch(e => setError(e.message))
  }
  useEffect(() => { go(initial || null) }, [machine, hidden])   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])

  const join = (dir: string, leaf: string) => dir.endsWith('/') ? dir + leaf : `${dir}/${leaf}`
  const q = filter.toLowerCase()
  const entries = (listing?.entries ?? []).filter(e => !q || e.name.toLowerCase().includes(q))
  const canUseFolder = mode === 'dir' || (mode === 'save' && name.trim() !== '')
  return (
    <div className="modal-back" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal browser" ref={box} role="dialog" aria-label={title}>
        <div className="browser-head">
          <b>{title}</b><span className="faint small">on {machine}</span>
          <span className="right"><button className="quiet" onClick={onClose} title="close">✕</button></span>
        </div>
        <form className="browser-path" onSubmit={e => { e.preventDefault(); go(typed) }}>
          <button type="button" className="quiet" disabled={!listing?.parent} onClick={() => go(listing!.parent)} title="up one folder">↑</button>
          <input className="mono" value={typed} onChange={e => setTyped(e.target.value)} spellCheck={false} />
          <button type="submit">Go</button>
        </form>
        <div className="browser-places">
          {listing?.places.map(([label, path]) => <button key={label} className="quiet" onClick={() => go(path)}>{label}</button>)}
          <label className="faint small"><input type="checkbox" checked={hidden} onChange={e => setHidden(e.target.checked)} /> hidden</label>
          <input type="search" placeholder="filter…" value={filter} onChange={e => setFilter(e.target.value)} />
        </div>
        {error && <p className="error small">{error}</p>}
        <ul className="browser-list">
          {entries.map(e => {
            const full = listing ? join(listing.path, e.name) : e.name
            const pickable = e.dir ? mode === 'dir' : mode === 'file'
            return (
              <li key={e.name} className={(e.dir ? 'dir' : 'file') + (e.broken ? ' broken' : '')}
                  onDoubleClick={() => { if (e.dir) go(full); else if (mode === 'file') onPick(full) }}
                  onClick={() => { if (e.dir) go(full); else if (mode === 'file') onPick(full); else if (mode === 'save') setName(e.name) }}
                  title={full}>
                <span className="icon">{e.dir ? '▸' : '·'}</span>
                <span className="name">{e.name}</span>
                <span className="faint small">{e.size != null ? fmtSize(e.size) : ''}</span>
                {pickable && e.dir && <button className="quiet small" onClick={ev => { ev.stopPropagation(); onPick(full) }}>select</button>}
              </li>
            )
          })}
          {listing && entries.length === 0 && <li className="faint">empty</li>}
        </ul>
        <div className="browser-foot">
          {mode === 'save' && listing && (
            <label className="ctl"><span>Name</span>
              <input className="mono" value={name} onChange={e => setName(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && canUseFolder) onPick(join(listing.path, name.trim())) }} />
            </label>
          )}
          <span className="faint small mono browser-current">{listing ? (mode === 'save' && name ? join(listing.path, name.trim()) : listing.path) : ''}</span>
          <span className="right">
            {mode !== 'file' && <button className="primary" disabled={!listing || !canUseFolder} onClick={() => onPick(mode === 'save' ? join(listing!.path, name.trim()) : listing!.path)}>
              {mode === 'save' ? 'Use this name' : 'Use this folder'}
            </button>}
            <button onClick={onClose}>Cancel</button>
          </span>
        </div>
      </div>
    </div>
  )
}

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} kB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

/** The folder button next to a path field. */
export function BrowseButton({ onClick, title }: { onClick: () => void; title: string }) {
  return (
    <button type="button" className="browse" onClick={onClick} title={title} aria-label={title}>
      <svg width="14" height="12" viewBox="0 0 14 12" aria-hidden="true">
        <path d="M1 2.5A1 1 0 0 1 2 1.5h3l1.2 1.3H12a1 1 0 0 1 1 1V10a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1z" fill="none" stroke="currentColor" strokeWidth="1.2" />
      </svg>
    </button>
  )
}

/** Which fields get a browse button, and how the picker behaves for them. */
const DIR_DESTS = new Set(['data_folder', 'scan_folder', 'template', 'checkpoint_dir', 'calibration_dir'])
const FILE_DESTS = new Set(['volume', 'infile', 'gauss_resume'])
const SAVE_DESTS = new Set(['output', 'out', 'report', 'gauss_checkpoint'])
export function browseModeFor(dest: string, metavar: string | null): BrowseMode | null {
  if (DIR_DESTS.has(dest)) return 'dir'
  if (FILE_DESTS.has(dest)) return 'file'
  if (SAVE_DESTS.has(dest)) return 'save'
  if (metavar === 'DIR') return 'dir'
  if (metavar === 'PATH' || metavar === 'FILE') return 'file'
  return null
}
