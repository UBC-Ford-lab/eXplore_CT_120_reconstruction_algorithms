import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, RoiBounds, RoiPreview } from '../api'

/** The ``--roi`` box drawn on a coarse preview of the scan.
 *
 *  The preview is a whole-field FDK at 0.4 mm (gui/roi_preview.py), shown as
 *  three slices with a slider each (or maximum-intensity projections). The box is one set of six
 *  millimetre bounds shared by the three panes: dragging a corner, an edge or
 *  the box itself in any pane moves the bounds that pane shows. "Use this ROI"
 *  snaps the extents to whole voxels of the job's own pitch, so the numbers
 *  rebuild exactly the lattice the FDK driver prints as "Replicate this grid
 *  with". */

type Axis = 'x' | 'y' | 'z'
interface Pane { look: Axis; h: Axis; v: Axis; name: string }
// each pane looks along one axis; columns run along ``h``, rows along ``v`` (index up = mm up = down the screen)
const PANES: Pane[] = [
  { look: 'z', h: 'x', v: 'y', name: 'axial' },
  { look: 'y', h: 'x', v: 'z', name: 'coronal' },
  { look: 'x', h: 'y', v: 'z', name: 'sagittal' },
]
const AXIS_INDEX: Record<Axis, number> = { x: 0, y: 1, z: 2 }
const MIN_MM = 1.0           // a box can never be thinner than this
const HANDLE = 9             // px

interface Vol { data: Uint8Array; nx: number; ny: number; nz: number }

function decode(p: RoiPreview): Vol {
  const bin = atob(p.data)
  const data = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i)
  return { data, nx: p.shape[0], ny: p.shape[1], nz: p.shape[2] }
}

/** rows x cols image of one pane: the maximum along the viewing axis, or the slice at ``index``. */
function paneImage(vol: Vol, pane: Pane, index: number | null): { rows: number; cols: number; px: Uint8Array } {
  const { data, nx, ny, nz } = vol
  const n = { x: nx, y: ny, z: nz }
  const rows = n[pane.v], cols = n[pane.h]
  const px = new Uint8Array(rows * cols)
  const along = n[pane.look]
  const lo = index == null ? 0 : Math.max(0, Math.min(along - 1, index))
  const hi = index == null ? along : lo + 1
  // flat index of (x, y, z) is (z * ny + y) * nx + x
  for (let k = lo; k < hi; k++) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = pane.h === 'x' ? c : pane.look === 'x' ? k : 0
        const y = pane.h === 'y' ? c : pane.v === 'y' ? r : k
        const z = pane.v === 'z' ? r : k
        const v = data[(z * ny + y) * nx + x]
        const i = r * cols + c
        if (v > px[i]) px[i] = v
      }
    }
  }
  return { rows, cols, px }
}

const KEYS: (keyof RoiBounds)[] = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
const lo = (a: Axis) => `${a}_min` as keyof RoiBounds
const hi = (a: Axis) => `${a}_max` as keyof RoiBounds

function clampBox(b: RoiBounds, field: RoiBounds): RoiBounds {
  const out = { ...b }
  for (const a of ['x', 'y', 'z'] as Axis[]) {
    out[lo(a)] = Math.max(field[lo(a)], Math.min(out[lo(a)], field[hi(a)] - MIN_MM))
    out[hi(a)] = Math.min(field[hi(a)], Math.max(out[hi(a)], out[lo(a)] + MIN_MM))
  }
  return out
}

/** The six numbers of a ``--roi`` value, or null for auto/full/anything else. */
export function parseRoi(value: unknown): RoiBounds | null {
  const parts = Array.isArray(value) ? value : typeof value === 'string' ? value.trim().split(/\s+/) : []
  if (parts.length !== 6) return null
  const nums = parts.map(Number)
  if (nums.some(n => !Number.isFinite(n))) return null
  const b = { x_min: nums[0], x_max: nums[1], y_min: nums[2], y_max: nums[3], z_min: nums[4], z_max: nums[5] }
  return b.x_max > b.x_min && b.y_max > b.y_min && b.z_max > b.z_min ? b : null
}

/** Whole voxels of the job's pitch, so the bounds round-trip into the same grid. */
export function snapToVoxels(b: RoiBounds, voxelXY: number, voxelZ: number): RoiBounds {
  const snap = (min: number, max: number, d: number): [number, number] => {
    const n = Math.max(1, Math.round((max - min) / d))
    const lo = Math.round(min * 1e4) / 1e4
    return [lo, Math.round((lo + n * d) * 1e4) / 1e4]
  }
  const [x0, x1] = snap(b.x_min, b.x_max, voxelXY)
  const [y0, y1] = snap(b.y_min, b.y_max, voxelXY)
  const [z0, z1] = snap(b.z_min, b.z_max, voxelZ)
  return { x_min: x0, x_max: x1, y_min: y0, y_max: y1, z_min: z0, z_max: z1 }
}

export function roiValues(b: RoiBounds): string[] {
  return KEYS.map(k => b[k].toFixed(4))
}

export default function RoiPicker({ machine, dataFolder, scanFolder, phase, voxelXY, voxelZ, initial, onPick, onClose }: {
  machine: string; dataFolder: string; scanFolder?: string | null; phase?: string
  voxelXY: number; voxelZ: number
  initial: RoiBounds | null              // the form's current --roi, when it is six numbers
  onPick: (b: RoiBounds) => void; onClose: () => void
}) {
  const [preview, setPreview] = useState<RoiPreview | null>(null)
  const [vol, setVol] = useState<Vol | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [box, setBox] = useState<RoiBounds | null>(null)
  const [mode, setMode] = useState<'mip' | 'slice'>('slice')
  const [slice, setSlice] = useState<Record<Axis, number>>({ x: 0, y: 0, z: 0 })
  const [levels, setLevels] = useState<Levels>([0, 128, 255])   // display black point, level (mid-grey) and white point on the 8-bit preview

  const load = useCallback((refresh: boolean) => {
    setLoading(true); setError(null)
    api.roiPreview({ executor: machine, data_folder: dataFolder, scan_folder: scanFolder || null, phase: phase || '00', refresh })
      .then(p => {
        const v = decode(p)
        setPreview(p); setVol(v)
        setSlice({ x: Math.floor(v.nx / 2), y: Math.floor(v.ny / 2), z: Math.floor(v.nz / 2) })
        setBox(prev => prev ?? clampBox(initial ?? p.scanner_roi ?? centred(p.bounds), p.bounds))
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [machine, dataFolder, scanFolder, phase])   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { load(false) }, [load])
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])

  const field = preview?.bounds
  function setBound(k: keyof RoiBounds, v: number) {
    if (!box || !field || !Number.isFinite(v)) return
    setBox(clampBox({ ...box, [k]: v }, field))
  }
  const snapped = box ? snapToVoxels(box, voxelXY, voxelZ) : null
  const voxels = snapped ? [
    Math.round((snapped.x_max - snapped.x_min) / voxelXY),
    Math.round((snapped.y_max - snapped.y_min) / voxelXY),
    Math.round((snapped.z_max - snapped.z_min) / voxelZ)] : null

  return (
    <div className="modal-back" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal roi" role="dialog" aria-label="Choose the region of interest">
        <div className="browser-head">
          <b>Region of interest</b>
          <span className="faint small mono" title={dataFolder}>{dataFolder}</span>
          <span className="faint small">on {machine}</span>
          <span className="right"><button className="quiet" onClick={onClose} title="close">✕</button></span>
        </div>
        <div className="roi-body">
          {loading && (
            <div className="roi-wait">
              <p><b>Reconstructing a coarse preview…</b></p>
              <p className="muted small">The whole field at {0.4} mm from every second projection: a few seconds on a GPU, up to a minute on a CPU, then cached.</p>
            </div>
          )}
          {error && <p className="error small">{error}</p>}
          {!loading && preview && vol && box && field && (
            <>
              <div className="roi-bar">
                <span className="roi-modes">
                  <button className={mode === 'slice' ? 'primary' : ''} onClick={() => setMode('slice')} title="one slice; each pane's slider moves it">slice</button>
                  <button className={mode === 'mip' ? 'primary' : ''} onClick={() => setMode('mip')} title="brightest voxel along the line of sight: bone and the bed stand out, soft tissue is a flat silhouette">projection</button>
                </span>
                <WindowLevel levels={levels} onChange={setLevels} />
                <span className="right small">
                  {preview.scanner_roi && <button className="small" onClick={() => setBox(clampBox(preview.scanner_roi!, field))} title="the vendor's own sub-volume (SubVolumeCoordinates.xml)">scanner ROI</button>}
                  <button className="small" onClick={() => setBox(field)} title="everything the preview shows">whole field</button>
                  <button className="quiet small" onClick={() => load(true)} title="reconstruct the preview again">recompute</button>
                </span>
              </div>
              <div className="roi-grid">
                {PANES.map(pane => (
                  <PaneView key={pane.look} pane={pane} vol={vol} preview={preview} box={box} levels={levels}
                    sliceIndex={mode === 'slice' ? slice[pane.look] : null} slicePos={slice[pane.look]}
                    onSlice={i => { setSlice(s => ({ ...s, [pane.look]: i })); setMode('slice') }}
                    onBox={b => setBox(clampBox(b, field))} />
                ))}
              </div>
              <div className="roi-bounds">
                {(['x', 'y', 'z'] as Axis[]).map(a => (
                  <span key={a} className="roi-axis">
                    <b>{a}</b>
                    <input type="number" step={0.1} value={round(box[lo(a)])} onChange={e => setBound(lo(a), Number(e.target.value))} />
                    <span className="faint">to</span>
                    <input type="number" step={0.1} value={round(box[hi(a)])} onChange={e => setBound(hi(a), Number(e.target.value))} />
                    <span className="faint small">mm</span>
                  </span>
                ))}
              </div>
              <p className="note">
                Drag the corners or edges, or the box itself, in any pane; the three panes share one box. Millimetres about the isocentre, the frame <span className="mono">--roi</span> takes.
                {' '}Preview {preview.voxel_mm} mm · {preview.n_views} views · {preview.cached ? 'cached' : `reconstructed in ${preview.seconds.reconstruct} s (loaded in ${preview.seconds.load} s)`} ·
                field x [{round(field.x_min)}, {round(field.x_max)}] y [{round(field.y_min)}, {round(field.y_max)}] z [{round(field.z_min)}, {round(field.z_max)}].
              </p>
            </>
          )}
        </div>
        <div className="browser-foot">
          <span className="faint small mono browser-current">
            {snapped && voxels ? `--roi ${roiValues(snapped).join(' ')} · ${voxels[0]} × ${voxels[1]} × ${voxels[2]} voxels at ${voxelXY} / ${voxelZ} mm` : ''}
          </span>
          <span className="right">
            <button className="primary" disabled={!snapped} onClick={() => snapped && onPick(snapped)}>Use this ROI</button>
            <button onClick={onClose}>Cancel</button>
          </span>
        </div>
      </div>
    </div>
  )
}

/** Window and level on the 8-bit preview as one three-point slider. The two
 *  round handles are the black and white points: what is shown black and
 *  what is shown white. The vertical bar between them is the level: the
 *  value shown mid-grey. It moves on its own anywhere inside the window
 *  (close to the white point stretches the dark half of the window over
 *  most of the greys and squeezes the bright half), and it keeps its
 *  relative position when an end handle moves. Clicking the bare track
 *  puts the level there. */
type Levels = [number, number, number]   // black, level, white on the 8-bit scale

function WindowLevel({ levels, onChange }: { levels: Levels; onChange: (l: Levels) => void }) {
  const track = useRef<HTMLDivElement>(null)
  const drag = useRef<{ what: 'black' | 'white' | 'level'; start: Levels } | null>(null)
  const [black, level, white] = levels
  const pct = (v: number) => `${v / 255 * 100}%`
  const valueAt = (clientX: number) => {
    const r = track.current!.getBoundingClientRect()
    return Math.round(Math.max(0, Math.min(255, (clientX - r.left) / r.width * 255)))
  }
  // an end handle at ``b``/``w``: the level keeps its fraction of the window
  const withEnds = (b: number, w: number, s: Levels): Levels => {
    const frac = (s[1] - s[0]) / Math.max(1, s[2] - s[0])
    return [b, Math.round(Math.max(b + 1, Math.min(w - 1, b + frac * (w - b)))), w]
  }
  const down = (what: 'black' | 'white' | 'level') => (e: React.PointerEvent) => {
    e.preventDefault(); e.stopPropagation()
    drag.current = { what, start: levels }
    ;(e.currentTarget as Element).setPointerCapture(e.pointerId)
  }
  function move(e: React.PointerEvent) {
    const d = drag.current
    if (!d) return
    const v = valueAt(e.clientX)
    if (d.what === 'black') onChange(withEnds(Math.min(v, d.start[2] - 2), d.start[2], d.start))
    else if (d.what === 'white') onChange(withEnds(d.start[0], Math.max(v, d.start[0] + 2), d.start))
    else onChange([d.start[0], Math.max(d.start[0] + 1, Math.min(d.start[2] - 1, v)), d.start[2]])
  }
  function up(e: React.PointerEvent) {
    if (!drag.current) return
    drag.current = null
    try { (e.currentTarget as Element).releasePointerCapture(e.pointerId) } catch { /* already released */ }
  }
  function trackDown(e: React.PointerEvent) {
    // a click on the bare track puts the level there (inside the window)
    onChange([black, Math.max(black + 1, Math.min(white - 1, valueAt(e.clientX))), white])
  }
  return (
    <span className="wl small" title={`black ${black}, level ${level}, white ${white} of 255: the ends set the window, the bar is the value shown mid-grey`}>
      <span className="faint">window</span>
      <div className="wl-track" ref={track} onPointerDown={trackDown}>
        <div className="wl-range" style={{ left: pct(black), width: pct(white - black) }} />
        <div className="wl-level" style={{ left: pct(level) }} onPointerDown={down('level')} onPointerMove={move} onPointerUp={up} onPointerCancel={up} />
        <div className="wl-handle" style={{ left: pct(black) }} onPointerDown={down('black')} onPointerMove={move} onPointerUp={up} onPointerCancel={up} />
        <div className="wl-handle" style={{ left: pct(white) }} onPointerDown={down('white')} onPointerMove={move} onPointerUp={up} onPointerCancel={up} />
      </div>
      <span className="mono faint">{black}–{white} · L {level}</span>
      {(black !== 0 || level !== 128 || white !== 255) && <button className="quiet small" onClick={() => onChange([0, 128, 255])} title="full window, level in the middle">reset</button>}
    </span>
  )
}

function centred(f: RoiBounds): RoiBounds {
  const mid = (a: number, b: number, k: number): [number, number] => { const c = (a + b) / 2, h = (b - a) * k / 2; return [c - h, c + h] }
  const [x0, x1] = mid(f.x_min, f.x_max, 0.6), [y0, y1] = mid(f.y_min, f.y_max, 0.6), [z0, z1] = mid(f.z_min, f.z_max, 0.8)
  return { x_min: x0, x_max: x1, y_min: y0, y_max: y1, z_min: z0, z_max: z1 }
}
const round = (v: number) => Math.round(v * 100) / 100

type Grip = 'move' | 'n' | 's' | 'w' | 'e' | 'nw' | 'ne' | 'sw' | 'se'

function PaneView({ pane, vol, preview, box, levels, sliceIndex, slicePos, onSlice, onBox }: {
  pane: Pane; vol: Vol; preview: RoiPreview; box: RoiBounds; levels: Levels
  sliceIndex: number | null; slicePos: number; onSlice: (i: number) => void; onBox: (b: RoiBounds) => void
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const wrap = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState<[number, number]>([0, 0])
  const image = useMemo(() => paneImage(vol, pane, sliceIndex), [vol, pane, sliceIndex])
  const f = preview.bounds
  const hLo = f[lo(pane.h)], hHi = f[hi(pane.h)], vLo = f[lo(pane.v)], vHi = f[hi(pane.v)]
  const along = vol[`n${pane.look}` as 'nx' | 'ny' | 'nz']

  useEffect(() => {
    const cv = canvas.current
    if (!cv) return
    const { rows, cols, px } = image
    cv.width = cols; cv.height = rows
    const out = new Uint8ClampedArray(rows * cols * 4)
    const [black, level, white] = levels
    // piecewise linear: black -> 0, level -> 128, white -> 255
    const lower = 128 / Math.max(1, level - black), upper = 127 / Math.max(1, white - level)
    for (let i = 0; i < rows * cols; i++) {
      const v = px[i]
      const g = v <= level ? (v - black) * lower : 128 + (v - level) * upper   // Uint8ClampedArray clamps to 0..255
      out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = g; out[i * 4 + 3] = 255
    }
    cv.getContext('2d')!.putImageData(new ImageData(out, cols, rows), 0, 0)
  }, [image, levels])
  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize([el.clientWidth, el.clientHeight]))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // mm <-> px of the displayed pane
  const [w, h] = size
  const toPx = (hm: number, vm: number): [number, number] => [(hm - hLo) / (hHi - hLo) * w, (vm - vLo) / (vHi - vLo) * h]
  const toMm = (x: number, y: number): [number, number] => [hLo + x / w * (hHi - hLo), vLo + y / h * (vHi - vLo)]
  const [x0, y0] = toPx(box[lo(pane.h)], box[lo(pane.v)])
  const [x1, y1] = toPx(box[hi(pane.h)], box[hi(pane.v)])

  const drag = useRef<{ grip: Grip; start: [number, number]; box: RoiBounds } | null>(null)
  function down(grip: Grip) {
    return (e: React.PointerEvent) => {
      e.preventDefault(); e.stopPropagation()
      const r = wrap.current!.getBoundingClientRect()
      drag.current = { grip, start: toMm(e.clientX - r.left, e.clientY - r.top), box }
      ;(e.currentTarget as Element).setPointerCapture(e.pointerId)
    }
  }
  function move(e: React.PointerEvent) {
    const d = drag.current
    if (!d) return
    const r = wrap.current!.getBoundingClientRect()
    const [hm, vm] = toMm(e.clientX - r.left, e.clientY - r.top)
    const dh = hm - d.start[0], dv = vm - d.start[1]
    const b = { ...d.box }
    const H = pane.h, V = pane.v, g = d.grip
    if (g === 'move') {
      // keep the whole box inside the field
      const sh = Math.max(hLo - b[lo(H)], Math.min(hHi - b[hi(H)], dh))
      const sv = Math.max(vLo - b[lo(V)], Math.min(vHi - b[hi(V)], dv))
      b[lo(H)] += sh; b[hi(H)] += sh; b[lo(V)] += sv; b[hi(V)] += sv
    } else {
      if (g.includes('w')) b[lo(H)] = Math.min(d.box[lo(H)] + dh, d.box[hi(H)] - MIN_MM)
      if (g.includes('e')) b[hi(H)] = Math.max(d.box[hi(H)] + dh, d.box[lo(H)] + MIN_MM)
      if (g.includes('n')) b[lo(V)] = Math.min(d.box[lo(V)] + dv, d.box[hi(V)] - MIN_MM)
      if (g.includes('s')) b[hi(V)] = Math.max(d.box[hi(V)] + dv, d.box[lo(V)] + MIN_MM)
    }
    onBox(b)
  }
  function up(e: React.PointerEvent) {
    if (drag.current) { drag.current = null; try { (e.currentTarget as Element).releasePointerCapture(e.pointerId) } catch { /* already released */ } }
  }
  const grips: { grip: Grip; x: number; y: number; cursor: string }[] = [
    { grip: 'nw', x: x0, y: y0, cursor: 'nwse-resize' }, { grip: 'ne', x: x1, y: y0, cursor: 'nesw-resize' },
    { grip: 'sw', x: x0, y: y1, cursor: 'nesw-resize' }, { grip: 'se', x: x1, y: y1, cursor: 'nwse-resize' },
    { grip: 'n', x: (x0 + x1) / 2, y: y0, cursor: 'ns-resize' }, { grip: 's', x: (x0 + x1) / 2, y: y1, cursor: 'ns-resize' },
    { grip: 'w', x: x0, y: (y0 + y1) / 2, cursor: 'ew-resize' }, { grip: 'e', x: x1, y: (y0 + y1) / 2, cursor: 'ew-resize' },
  ]
  const aspect = `${(hHi - hLo)} / ${(vHi - vLo)}`
  return (
    <div className="roi-view">
      <div className="view-head">
        <b>{pane.name}</b>
        <span className="faint small">{pane.h} → · {pane.v} ↓</span>
        <input type="range" min={0} max={along - 1} value={slicePos} onChange={e => onSlice(Number(e.target.value))}
               title={sliceIndex == null ? 'move to show this slice instead of the projection' : `slice along ${pane.look}; the mouse wheel over the image steps it`} />
        <span className="mono small">{sliceIndex == null ? 'projection' : `${pane.look} = ${round(f[lo(pane.look)] + (sliceIndex + 0.5) * preview.spacing[AXIS_INDEX[pane.look]])} mm`}</span>
      </div>
      <div className="roi-pane" ref={wrap} style={{ aspectRatio: aspect }}
           onPointerMove={move} onPointerUp={up} onPointerCancel={up}
           onWheel={e => { e.preventDefault(); onSlice(Math.max(0, Math.min(along - 1, slicePos + (e.deltaY > 0 ? 1 : -1)))) }}>
        <canvas ref={canvas} />
        {w > 0 && (
          <svg width={w} height={h}>
            <rect className="box" x={Math.min(x0, x1)} y={Math.min(y0, y1)} width={Math.abs(x1 - x0)} height={Math.abs(y1 - y0)} onPointerDown={down('move')} />
            {grips.map(g => (
              <rect key={g.grip} className="handle" x={g.x - HANDLE / 2} y={g.y - HANDLE / 2} width={HANDLE} height={HANDLE}
                    style={{ cursor: g.cursor }} onPointerDown={down(g.grip)} />
            ))}
          </svg>
        )}
      </div>
    </div>
  )
}
