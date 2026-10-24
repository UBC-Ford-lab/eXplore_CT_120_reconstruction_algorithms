// The "detail" row of a Gaussian job: one preset for the three flags that set
// the reconstruction's effective resolution (primitive spacing, detector
// pooling, export voxel), and what the run will cost at them — primitives,
// VRAM and time, from /api/drivers/learned/estimate (gui/detail.py).
import { Estimate, Machine, Schema, fmtBytes, fmtDuration } from '../api'
import { Values } from './Form'

export interface Preset { name: string; spacing: number; downsample: number; voxel: number; blurb: string }
export const PRESETS: Preset[] = [
  { name: 'fast', spacing: 0.45, downsample: 6, voxel: 0.15, blurb: 'a quick look: a tenth of the primitives on coarse pixels' },
  { name: 'standard', spacing: 0.22, downsample: 3, voxel: 0.075, blurb: 'primitives spaced at the scanner PSF; the production default' },
  { name: 'fine', spacing: 0.15, downsample: 2, voxel: 0.05, blurb: 'finer than the PSF: several times the primitives and VRAM, mostly sharper noise' },
]
export const DETAIL_ALGORITHMS = ['gaussian', 'gaussian_dynamic']

const DESTS = ['gauss_seed_spacing', 'downsample', 'voxel_xy', 'voxel_z'] as const

function fieldDefault(schema: Schema, dest: string): any {
  return schema.fields.find(f => f.dest === dest)?.default
}
function current(schema: Schema, values: Values) {
  const num = (dest: string) => Number(values[dest] ?? fieldDefault(schema, dest))
  return { spacing: num('gauss_seed_spacing'), downsample: num('downsample'), voxelXY: num('voxel_xy'), voxelZ: num('voxel_z') }
}
const near = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b))

/** Which preset the form's flags are, or 'custom'. */
export function presetOf(schema: Schema, values: Values): string {
  const c = current(schema, values)
  const p = PRESETS.find(p => near(p.spacing, c.spacing) && p.downsample === c.downsample && near(p.voxel, c.voxelXY) && near(p.voxel, c.voxelZ))
  return p ? p.name : 'custom'
}

/** The values with a preset applied: a flag equal to the parser's default is dropped, as typing it would. */
export function applyPreset(schema: Schema, values: Values, p: Preset): Values {
  const next = { ...values }
  const put = (dest: string, v: number) => { if (near(Number(fieldDefault(schema, dest)), v)) delete next[dest]; else next[dest] = v }
  put('gauss_seed_spacing', p.spacing); put('downsample', p.downsample); put('voxel_xy', p.voxel); put('voxel_z', p.voxel)
  return next
}

/** The form values the estimate depends on, as a key for the page's fetch. */
export function estimateKey(values: Values, machine: string, algorithm: string, resources: string): string {
  const { gauss_seed_spacing, downsample, voxel_xy, voxel_z, scan_folder, phase, roi, iterations, gauss_seeds, gauss_max, gauss_seed_roi_weight, loss, rays_per_batch } = values
  return JSON.stringify([gauss_seed_spacing, downsample, voxel_xy, voxel_z, scan_folder, phase, roi, iterations, gauss_seeds, gauss_max,
    gauss_seed_roi_weight, loss, rays_per_batch, machine, algorithm, resources])
}

export default function DetailPresets({ schema, values, setValues, machine, machines, scanPath, est, state, error }: {
  schema: Schema; values: Values; setValues: (v: Values) => void
  machine: string; machines: Machine[]; scanPath: string
  est: Estimate | null; state: 'idle' | 'busy' | 'error'; error: string
}) {
  const chosen = presetOf(schema, values)
  const c = current(schema, values)
  const gpu = machines.find(m => m.name === machine)?.gpus ?? []
  const free = gpu.length ? Math.max(...gpu.map(g => g.free_gib)) : null
  const total = gpu.length ? Math.max(...gpu.map(g => g.total_gib)) : null
  const need = est?.gpu_bytes != null ? est.gpu_bytes / 1024 ** 3 : null
  const fit = need == null || free == null ? '' : need > (total ?? free) ? 'fail' : need > 0.85 * free ? 'warn' : 'ok'

  return (
    <div className={'opt pos detail-row' + (chosen !== 'standard' ? ' changed' : '')}
      title="primitive spacing (--gauss-seed-spacing), detector pooling (--downsample) and export voxel (--voxel-xy/-z) as one choice; every flag stays editable below">
      <label className="flag">detail</label>
      <div className="input">
        <div className="seg" role="radiogroup">
          {PRESETS.map(p => (
            <button key={p.name} type="button" className={chosen === p.name ? 'on' : ''} aria-checked={chosen === p.name} role="radio"
              onClick={() => setValues(applyPreset(schema, values, p))} title={p.blurb}>
              <b>{p.name}</b><small>{p.spacing} mm · 1/{p.downsample} px · {p.voxel} mm</small>
            </button>
          ))}
          {chosen === 'custom' && <span className="custom" title="the three flags do not match a preset"><b>custom</b><small>{c.spacing} mm · 1/{c.downsample} px · {c.voxelXY}{near(c.voxelXY, c.voxelZ) ? '' : `/${c.voxelZ}`} mm</small></span>}
        </div>
      </div>
      <div className="help detail-help">
        <span>Primitives {c.spacing} mm apart, detector pooled {c.downsample}×, exported on {c.voxelXY}{near(c.voxelXY, c.voxelZ) ? '' : ` × ${c.voxelZ}`} mm voxels.</span>
        {!scanPath.trim() ? <span className="faint"> Give the scan folder to see the cost.</span>
          : state === 'error' ? <span className="error"> estimate unavailable: {error}</span>
          : est && est.primitives != null ? (
            <span className={state === 'busy' ? 'faint' : ''}>
              {' '}≈ <b>{(est.primitives / 1e6).toFixed(2)} M primitives</b>
              {' · '}<b className={fit}>{fmtBytes(est.gpu_bytes ?? 0)} VRAM</b>{free != null && <span className="faint"> of {free.toFixed(1)} GiB free on {machine}</span>}
              {' · '}<b>about {fmtDuration(est.seconds)}</b> for {est.iterations.toLocaleString()} iterations
              <span className="faint"> ({est.rate_source})</span>
              {est.pooled_pixel_mm != null && <span className="faint"> · pixel {est.pooled_pixel_mm.toFixed(3)} mm at the isocentre, PSF {est.psf_mm} mm</span>}
              {est.warnings.map((w, i) => <span key={i} className="warn"> ⚠ {w}</span>)}
            </span>
          ) : <span className="faint"> estimating… {state === 'busy' && '(the first time reconstructs a coarse preview of the scan, a few seconds on a GPU)'}</span>}
      </div>
    </div>
  )
}
