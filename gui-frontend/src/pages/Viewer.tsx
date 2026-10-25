import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, Info, VolumeInfo } from '../api'
import Browser, { BrowseButton } from '../components/Browser'

type Axis = 'x' | 'y' | 'z'
const AXES: Axis[] = ['z', 'y', 'x']
const AXIS_NAME: Record<Axis, string> = { z: 'axial (z)', y: 'coronal (y)', x: 'sagittal (x)' }
// window presets: [centre, width] in HU
const PRESETS: Record<string, [number, number]> = { lung: [-600, 1500], 'soft tissue': [40, 400], bone: [500, 2000], full: [500, 3000] }

interface Slice { rows: number; cols: number; data: Int16Array }

/** Slices of one .vff volume, read where it lives, windowed in the browser. */
export default function Viewer({ info }: { info: Info | null }) {
  const [params, setParams] = useSearchParams()
  const [machine, setMachine] = useState(params.get('machine') ?? 'local')
  const [path, setPath] = useState(params.get('path') ?? '')
  const [typed, setTyped] = useState(path)
  const [vol, setVol] = useState<VolumeInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [browsing, setBrowsing] = useState(false)
  const [pos, setPos] = useState<[number, number, number]>([0, 0, 0])   // x, y, z voxel indices
  // ?window=lung | soft%20tissue | bone | full | auto | <centre>,<width>
  const wantWindow = params.get('window') ?? 'full'
  const [win, setWin] = useState<[number, number]>(PRESETS[wantWindow] ?? (wantWindow.includes(',') ? wantWindow.split(',').map(Number) as [number, number] : PRESETS.full))
  const [preset, setPreset] = useState(PRESETS[wantWindow] ? wantWindow : wantWindow === 'auto' ? 'auto' : wantWindow.includes(',') ? 'custom' : 'full')
  const [slices, setSlices] = useState<Partial<Record<Axis, Slice>>>({})
  const [hover, setHover] = useState<{ axis: Axis; r: number; c: number; v: number } | null>(null)
  // one view enlarged over the whole window (its own slider on top); Escape closes it
  // ?view=x|y|z opens that view enlarged
  const [focus, setFocus] = useState<Axis | null>((AXES as string[]).includes(params.get('view') ?? '') ? params.get('view') as Axis : null)
  useEffect(() => {
    if (!focus) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setFocus(null) }
    window.addEventListener('keydown', onKey)
    document.body.classList.add('no-scroll')
    return () => { window.removeEventListener('keydown', onKey); document.body.classList.remove('no-scroll') }
  }, [focus])

  // open a volume: header first, then the middle slices
  useEffect(() => {
    if (!path) { setVol(null); setSlices({}); setFocus(null); return }
    setError(null); setVol(null); setSlices({})
    api.volumeInfo(machine, path).then(v => {
      setVol(v)
      setPos([Math.floor(v.size[0] / 2), Math.floor(v.size[1] / 2), Math.floor(v.size[2] / 2)])
      if (preset === 'auto') setWin([(v.sample[0] + v.sample[1]) / 2, Math.max(1, v.sample[1] - v.sample[0])])
    }).catch(e => setError(e.message))
  }, [machine, path])

  const cache = useRef(new Map<string, Promise<Slice>>())
  const fetchSlice = useCallback((axis: Axis, index: number): Promise<Slice> => {
    const key = `${machine}|${path}|${axis}|${index}|${vol?.mtime}`
    let p = cache.current.get(key)
    if (!p) {
      p = api.volumeSlice(machine, path, axis, index, vol?.mtime ?? 0)
      cache.current.set(key, p)
      if (cache.current.size > 300) cache.current.delete(cache.current.keys().next().value!)
    }
    return p
  }, [machine, path, vol?.mtime])

  // fetch the three slices for the current position (debounced while scrubbing)
  useEffect(() => {
    if (!vol) return
    let live = true
    const t = setTimeout(() => {
      for (const axis of AXES) {
        const index = axis === 'x' ? pos[0] : axis === 'y' ? pos[1] : pos[2]
        fetchSlice(axis, index).then(s => { if (live) setSlices(prev => ({ ...prev, [axis]: s })) }).catch(e => { if (live) setError(e.message) })
      }
    }, 60)
    return () => { live = false; clearTimeout(t) }
  }, [vol, pos, fetchSlice])

  function open(p: string) {
    setPath(p); setTyped(p)
    setParams({ machine, path: p })
  }
  function setPreset_(name: string) {
    setPreset(name)
    if (PRESETS[name]) setWin(PRESETS[name])
    else if (name === 'auto' && vol) setWin([(vol.sample[0] + vol.sample[1]) / 2, Math.max(1, vol.sample[1] - vol.sample[0])])
  }
  const spacing = vol?.spacing ?? [1, 1, 1]
  const units = vol?.units === 'HU' ? 'HU' : ''
  const readout = hover ? `${AXIS_NAME[hover.axis].split(' ')[0]} · ${hover.c}, ${hover.r} · ${hover.v}${units ? ' ' + units : ''}` : `x ${pos[0]} · y ${pos[1]} · z ${pos[2]}`
  function renderView(axis: Axis, big: boolean) {
    if (!vol) return null
    const n = axis === 'x' ? vol.size[0] : axis === 'y' ? vol.size[1] : vol.size[2]
    const index = axis === 'x' ? pos[0] : axis === 'y' ? pos[1] : pos[2]
    // pixel aspect (height/width) from the voxel spacing of the two in-plane axes
    const aspect = axis === 'z' ? spacing[1] / spacing[0] : axis === 'y' ? spacing[2] / spacing[0] : spacing[2] / spacing[1]
    // crosshair: column/row of the other two axes in this plane
    const cross: [number, number] = axis === 'z' ? [pos[0], pos[1]] : axis === 'y' ? [pos[0], pos[2]] : [pos[1], pos[2]]
    return (
      <div className={`view${big ? ' big' : ''}`} key={axis}>
        <div className="view-head">
          <b>{AXIS_NAME[axis]}</b>
          <input type="range" min={0} max={n - 1} value={index} onChange={e => setPos(p => { const q: [number, number, number] = [...p]; q[axis === 'x' ? 0 : axis === 'y' ? 1 : 2] = Number(e.target.value); return q })} />
          <span className="mono small">{index} / {n - 1}</span>
          {big && <span className="mono small faint view-readout">{readout}</span>}
          <button type="button" className="quiet view-full" title={big ? 'back to the three views (Esc)' : 'show only this view, full window'}
            onClick={() => setFocus(big ? null : axis)}>{big ? '✕' : '⤢'}</button>
        </div>
        <SliceCanvas slice={slices[axis]} win={win} aspect={aspect} cross={cross} fit={big}
          onWheel={d => setPos(p => { const q: [number, number, number] = [...p]; const k = axis === 'x' ? 0 : axis === 'y' ? 1 : 2; q[k] = Math.max(0, Math.min(n - 1, q[k] + d)); return q })}
          onHover={(r, c, v) => setHover(r < 0 ? null : { axis, r, c, v })}
          onClick={(r, c) => setPos(p => {
            const q: [number, number, number] = [...p]
            if (axis === 'z') { q[0] = c; q[1] = r } else if (axis === 'y') { q[0] = c; q[2] = r } else { q[1] = c; q[2] = r }
            return q
          })} />
      </div>
    )
  }
  const executors = info?.executors ?? []
  return (
    <>
      <div className="page-head">
        <h1>Volume</h1>
        <div className="ctl-row">
          <label className="ctl"><span>Machine</span>
            <select value={machine} onChange={e => { setMachine(e.target.value); setParams({ machine: e.target.value, path }) }}>
              {executors.map(m => <option key={m.name} value={m.name} disabled={m.status !== 'connected'}>{m.name}{m.status !== 'connected' ? ` (${m.status.replace('_', ' ')})` : ''}</option>)}
            </select>
          </label>
          <form className="ctl grow" onSubmit={e => { e.preventDefault(); open(typed) }}>
            <span>Volume (.vff)</span>
            <span className="with-browse">
              <input type="text" className="mono" value={typed} onChange={e => setTyped(e.target.value)} placeholder="path on the machine" spellCheck={false} />
              <BrowseButton onClick={() => setBrowsing(true)} title="choose a volume" />
              <button type="submit">Open</button>
            </span>
          </form>
        </div>
        {browsing && <Browser machine={machine} mode="file" title="Volume" initial={typed} onClose={() => setBrowsing(false)} onPick={p => { setBrowsing(false); open(p) }} />}
      </div>
      {error && <p className="error">{error}</p>}
      {vol && (
        <div className="viewer">
          <div className="viewer-bar">
            <span className="mono small faint" title={vol.path}>{vol.size[0]} × {vol.size[1]} × {vol.size[2]} · {spacing.map(s => s.toFixed(3)).join(' × ')} mm · {vol.bits} bit{vol.units ? ` · ${vol.units}` : ''}{vol.sidecar ? '' : ' · no sidecar'}</span>
            <span className="viewer-window">
              <label>window
                <select value={preset} onChange={e => setPreset_(e.target.value)}>
                  {Object.keys(PRESETS).map(k => <option key={k} value={k}>{k}</option>)}
                  <option value="auto">auto</option>
                  <option value="custom">custom</option>
                </select>
              </label>
              <label>centre <input type="number" value={Math.round(win[0])} onChange={e => { setPreset('custom'); setWin([Number(e.target.value), win[1]]) }} /></label>
              <label>width <input type="number" value={Math.round(win[1])} min={1} onChange={e => { setPreset('custom'); setWin([win[0], Math.max(1, Number(e.target.value))]) }} /></label>
            </span>
            <span className="viewer-readout mono small">{readout}</span>
          </div>
          <div className="viewer-grid">
            {AXES.map(axis => renderView(axis, false))}
          </div>
          {focus && <div className="view-overlay">{renderView(focus, true)}</div>}
          <p className="note">Slices are read where the volume lives and shown in on-disk index order (rows down, columns right); click to move the crosshair, scroll or use the slider to change slice. Vendor volumes keep the vendor's axis conventions.</p>
        </div>
      )}
      {!vol && !error && !path && <p className="muted">Open a volume: a reconstruction's <span className="mono">.vff</span> on any connected machine. From a job page, "view volume" opens its output here. <Link to="/jobs">Jobs</Link></p>}
    </>
  )
}

function SliceCanvas({ slice, win, aspect, cross, fit = false, onHover, onClick, onWheel }: {
  slice?: Slice; win: [number, number]; aspect: number; cross: [number, number]; fit?: boolean
  onHover: (r: number, c: number, v: number) => void; onClick: (r: number, c: number) => void; onWheel: (delta: number) => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const box = useRef<HTMLDivElement>(null)
  // fit: scale the slice (up or down) to the largest aspect-correct rectangle inside the box
  const [boxSize, setBoxSize] = useState<[number, number]>([0, 0])
  useEffect(() => {
    if (!fit || !box.current) return
    const ro = new ResizeObserver(([e]) => setBoxSize([e.contentRect.width, e.contentRect.height]))
    ro.observe(box.current)
    return () => ro.disconnect()
  }, [fit])
  let size: React.CSSProperties = { aspectRatio: slice ? `${slice.cols} / ${slice.rows * aspect}` : '1 / 1' }
  if (fit && slice && boxSize[0] > 0 && boxSize[1] > 0) {
    const k = Math.min(boxSize[0] / slice.cols, boxSize[1] / (slice.rows * aspect))
    size = { width: `${Math.floor(slice.cols * k)}px`, height: `${Math.floor(slice.rows * aspect * k)}px`, maxWidth: 'none', maxHeight: 'none' }
  }
  const img = useMemo(() => {
    if (!slice) return null
    const { rows, cols, data } = slice
    const out = new Uint8ClampedArray(rows * cols * 4)
    const lo = win[0] - win[1] / 2, scale = 255 / win[1]
    for (let i = 0; i < rows * cols; i++) {
      const g = Math.max(0, Math.min(255, (data[i] - lo) * scale))
      out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = g
      out[i * 4 + 3] = 255
    }
    return new ImageData(out, cols, rows)
  }, [slice, win])
  useEffect(() => {
    const cv = ref.current
    if (!cv || !img || !slice) return
    cv.width = slice.cols; cv.height = slice.rows
    const ctx = cv.getContext('2d')!
    ctx.putImageData(img, 0, 0)
    ctx.strokeStyle = 'rgba(14, 107, 102, 0.8)'
    ctx.lineWidth = Math.max(1, slice.cols / 600)
    ctx.beginPath(); ctx.moveTo(cross[0] + 0.5, 0); ctx.lineTo(cross[0] + 0.5, slice.rows); ctx.moveTo(0, cross[1] + 0.5); ctx.lineTo(slice.cols, cross[1] + 0.5); ctx.stroke()
  }, [img, slice, cross])
  function voxelAt(e: React.MouseEvent<HTMLCanvasElement>): [number, number] | null {
    const cv = ref.current
    if (!cv || !slice) return null
    const b = cv.getBoundingClientRect()
    const c = Math.floor((e.clientX - b.left) / b.width * slice.cols)
    const r = Math.floor((e.clientY - b.top) / b.height * slice.rows)
    return c >= 0 && r >= 0 && c < slice.cols && r < slice.rows ? [r, c] : null
  }
  return (
    <div className="view-box" ref={box}>
      <canvas ref={ref} style={size}
        onMouseMove={e => { const v = voxelAt(e); if (v && slice) onHover(v[0], v[1], slice.data[v[0] * slice.cols + v[1]]) }}
        onMouseLeave={() => onHover(-1, -1, 0)}
        onWheel={e => { e.preventDefault(); onWheel(e.deltaY > 0 ? 1 : -1) }}
        onClick={e => { const v = voxelAt(e); if (v) onClick(v[0], v[1]) }} />
      {!slice && <span className="muted small">loading…</span>}
    </div>
  )
}
