// Thin client for the JSON API in gui/app.py.

export interface Gpu { index: number; name: string; total_gib: number; free_gib: number; used_gib?: number; utilisation: number | null }
// one row of the user's scheduler queue; ``job`` is set when it was submitted from this GUI
export interface QueueRow {
  id: string; name: string; state: string; elapsed: string; limit: string; nodes: string
  cpus: string; memory: string; gres: string; where: string; partition: string; job?: string | null
}
// the user's fair-share standing on a Slurm cluster, per account (see gui/remote.py parse_fairshare)
export interface FairShareAccount {
  account: string; kind: 'gpu' | 'cpu' | ''; level_fs: number | null; rank: number | null; user_level_fs: number | null
  half_life_days: number | null; running_share: number | null; level_fs_after_running: number | null
  fair_in_days_idle: number | null; fair_in_days_after_running: number | null
}
export interface MachineConfig {
  name: string; type: 'local' | 'ssh' | 'slurm'; host: string; python: string; workdir: string; scans_root: string
  slots: number; env: Record<string, string>; sbatch_template: string | null; template_is_default?: boolean; connection?: 'auto' | 'openssh' | 'library'
}
export interface Machine {
  name: string; kind: string; status: 'connected' | 'needs_login' | 'unreachable' | 'logged_out'; host?: string; error?: string
  hostname: string; platform: string; cpus: number | null
  memory: { total_gib: number | null; available_gib: number | null }
  gpus: Gpu[]; slots: number; running: string[]; pending: string[]; cwd: string
  package: { version: string | null; commit: string | null; python: string | null }
  queue?: { running: number; pending: number; jobs: QueueRow[] } | null
  fairshare?: { fetched: number; accounts: FairShareAccount[] } | null
  login_command?: string
  config: MachineConfig
}
export interface Info { version: string; executors: Machine[]; scans_root: string | null; cwd: string; data_dir: string }
export interface Algorithm { name: string; summary: string }
export interface Driver { name: string; title: string; script: string; kind: string; primary: string; algorithms: Algorithm[] }
export interface Field {
  dest: string; group: string; positional: boolean; flags: string[]; help: string; required: boolean
  default: any; metavar: string | null
  kind: 'flag' | 'choice' | 'integer' | 'number' | 'text' | 'list'
  choices?: any[]; item?: string; on?: string; off?: string
}
export type CheckStatus = 'ok' | 'warn' | 'fail' | 'info'
export interface CheckItem { label: string; status: CheckStatus; detail: string }
export interface CheckGroup { title: string; path: string; items: CheckItem[] }
export interface Directive { key: string; value: string | null; source: 'template' | 'override' }
export interface Preview { command: string; cwd: string; script: string | null; directives: Directive[]; error?: string }
export interface ServerInfo { pid: number; url: string | null; started: number; version: string; data_dir: string; active: number; stoppable: boolean }
export interface DirEntry { name: string; dir: boolean; size: number | null; mtime: number | null; broken?: boolean }
export interface DirListing { path: string; parent: string | null; entries: DirEntry[]; error: string | null; places: [string, string][] }
export interface VolumeInfo { path: string; size: [number, number, number]; bits: number; spacing: [number, number, number] | null; units: string | null; sidecar: boolean; bytes: number; mtime: number; sample: [number, number] }
export interface Schema { driver: string; script: string; algorithm: string | null; description: string; fields: Field[] }
export interface RoiBounds { x_min: number; x_max: number; y_min: number; y_max: number; z_min: number; z_max: number }
/** A coarse FDK of a whole scan (gui/roi_preview.py): ``data`` is base64 of nz*ny*nx uint8 in [z][y][x] order, index up = mm up. */
export interface RoiPreview {
  shape: [number, number, number]; spacing: [number, number, number]; bounds: RoiBounds; scanner_roi: RoiBounds | null
  window: [number, number]; n_views: number; voxel_mm: number; data_folder: string
  seconds: { load: number; reconstruct: number }; data: string; cached: boolean
}
export interface Scan { name: string; path: string }
/** What a Gaussian job at the form's flags will cost (gui/detail.py). */
export interface Walltime { estimate_s: number; raw_s: number; suggested_s: number; suggested: string; bucket_h: number; bucket: string }
export interface Estimate {
  algorithm: string; iterations: number; rate_source: string; gpu: string | null; warnings: string[]
  // the time half, for every learned algorithm; null before a run of that algorithm exists on the machine
  seconds: number | null; seconds_per_it: number | null; walltime: Walltime | null
  // the Gaussian half (null for the other representations)
  primitives: number | null
  spacing_mm?: number; downsample?: number; voxel_xy?: number; voxel_z?: number
  psf_mm?: number; pooled_pixel_mm?: number | null; cap?: number; seeds?: 'auto' | number
  n_views?: number; detector?: [number, number]; vol_shape?: [number, number, number]
  gpu_bytes?: number; host_bytes?: number; preview_cached?: boolean
}
export interface Job {
  id: string; driver: string; algorithm: string | null; values: Record<string, any>; command: string[]
  executor: string; status: 'queued' | 'pending' | 'running' | 'finished' | 'failed' | 'cancelled'
  created: number; started: number | null; finished: number | null; duration: number | null
  note: string; input: string; cwd: string; hostname: string | null; gpu_index: number | null; gpu_name: string | null
  remote_id: string | null; remote_state: string | null; node: string | null; resources: string
  pid: number | null; returncode: number | null; log_path: string | null; output_path: string | null
  wandb_url: string | null; package_version: string | null; error: string | null; shell?: string
  progress: Progress | null
}
export interface Progress { phase: string; done: number; total: number; fraction: number | null; phase_fraction: number | null; rate: number | null; eta: number | null; updated: number }

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...init })
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail ?? detail } catch { /* keep statusText */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return r.json()
}

export const api = {
  info: () => request<Info>('/api/info'),
  server: () => request<ServerInfo>('/api/server'),
  stopServer: () => request<{ stopping: boolean; active: number }>('/api/server/stop', { method: 'POST' }),
  drivers: () => request<Driver[]>('/api/drivers'),
  schema: (driver: string, algorithm?: string | null) =>
    request<Schema>(`/api/drivers/${driver}/schema${algorithm ? `?algorithm=${encodeURIComponent(algorithm)}` : ''}`),
  output: (driver: string, algorithm: string | null, values: Record<string, any>) =>
    request<{ dest?: string | null; path?: string; writes?: string[]; explicit?: boolean }>(`/api/drivers/${driver}/output`, { method: 'POST', body: JSON.stringify({ algorithm, values }) }),
  volumeInfo: (machine: string, path: string) => request<VolumeInfo>('/api/volumes/info', { method: 'POST', body: JSON.stringify({ path, executor: machine }) }),
  volumeSlice: async (machine: string, path: string, axis: 'x' | 'y' | 'z', index: number, mtime: number) => {
    const q = new URLSearchParams({ path, axis, index: String(index), executor: machine, mtime: String(mtime) })
    const r = await fetch(`/api/volumes/slice?${q}`)
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? r.statusText)
    const [rows, cols] = (r.headers.get('X-Shape') ?? '0,0').split(',').map(Number)
    return { rows, cols, data: new Int16Array(await r.arrayBuffer()) }
  },
  roiPreview: (body: { executor: string; data_folder: string; scan_folder?: string | null; phase?: string; refresh?: boolean }) =>
    request<RoiPreview>('/api/roi/preview', { method: 'POST', body: JSON.stringify(body) }),
  browse: (machine: string, path: string | null, hidden = false) =>
    request<DirListing>(`/api/machines/${encodeURIComponent(machine)}/browse`, { method: 'POST', body: JSON.stringify({ path, hidden }) }),
  scans: (machine = 'local') => request<Scan[]>(`/api/machines/${encodeURIComponent(machine)}/scans`),
  machines: () => request<Machine[]>('/api/machines'),
  putMachine: (name: string, body: Omit<MachineConfig, 'name'>) => request<Machine>(`/api/machines/${encodeURIComponent(name)}`, { method: 'PUT', body: JSON.stringify(body) }),
  deleteMachine: (name: string) => request<{ removed: string }>(`/api/machines/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  checkMachine: (name: string) => request<{ checks: { name: string; ok: boolean; detail: string }[]; machine: Machine }>(`/api/machines/${encodeURIComponent(name)}/check`, { method: 'POST' }),
  disconnect: (name: string) => request<Machine>(`/api/machines/${encodeURIComponent(name)}/disconnect`, { method: 'POST' }),
  connect: (name: string) => request<Machine>(`/api/machines/${encodeURIComponent(name)}/connect`, { method: 'POST' }),
  cancelQueueRow: (name: string, slurmId: string) => request<{ cancelled: string; job: string | null; machine: Machine }>(`/api/machines/${encodeURIComponent(name)}/queue/${encodeURIComponent(slurmId)}/cancel`, { method: 'POST' }),
  defaultTemplate: () => fetch('/api/machines/template/default').then(r => r.text()),
  jobs: (limit = 500) => request<Job[]>(`/api/jobs?limit=${limit}`),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  log: (id: string, offset: number) => request<{ offset: number; size: number; text: string }>(`/api/jobs/${id}/log?offset=${offset}`),
  submit: (body: { driver: string; algorithm?: string | null; values: Record<string, any>; executor: string; note: string; resources?: string }) =>
    request<Job>('/api/jobs', { method: 'POST', body: JSON.stringify(body) }),
  preview: (body: { driver: string; algorithm?: string | null; values: Record<string, any>; executor: string; resources?: string }) =>
    request<Preview>('/api/jobs/preview', { method: 'POST', body: JSON.stringify(body) }),
  estimate: (algorithm: string, values: Record<string, any>, executor: string, resources = '') =>
    request<Estimate>('/api/drivers/learned/estimate', { method: 'POST', body: JSON.stringify({ algorithm, values, executor, resources }) }),
  checks: (driver: string, algorithm: string | null, values: Record<string, any>, executor: string) =>
    request<{ groups: CheckGroup[] }>(`/api/drivers/${driver}/checks`, { method: 'POST', body: JSON.stringify({ algorithm, values, executor }) }),
  files: (id: string) => request<{ files: { path: string; dir: boolean; bytes: number }[]; output_path: string | null }>(`/api/jobs/${id}/files`),
  remove: (id: string, data: boolean) => request<{ deleted: string; files: any[] }>(`/api/jobs/${id}?data=${data}`, { method: 'DELETE' }),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  setNote: (id: string, note: string) => request<Job>(`/api/jobs/${id}`, { method: 'PATCH', body: JSON.stringify({ note }) }),
  checkpoint: (id: string) => request<{ checkpoint: string | null; bytes: number | null; resumable: boolean }>(`/api/jobs/${id}/checkpoint`),
}

export function fmtDuration(s: number | null): string {
  if (s == null) return ''
  if (s < 90) return `${Math.round(s)} s`
  if (s < 5400) return `${(s / 60).toFixed(1)} min`
  return `${(s / 3600).toFixed(2)} h`
}
/** 'filter · 42 % · about 3.1 min left', or without the estimate while the rate is unknown */
export function fmtProgress(p: Progress | null, finished = false): string {
  if (!p) return ''
  const step = `${p.phase} ${p.done}/${p.total}`
  const pct = p.fraction != null ? `${Math.round(p.fraction * 100)} %` : ''
  if (finished) return `${step}${pct ? ` · ${pct}` : ''}`
  if (p.eta === 0) return `${step} · done`
  return `${step}${pct ? ` · ${pct}` : ''}${p.eta != null ? ` · about ${fmtDuration(p.eta)} left` : ' · estimating…'}`
}
export function fmtTime(t: number | null): string {
  return t ? new Date(t * 1000).toLocaleString() : ''
}

export function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(0)} KiB`
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)} MiB`
  return `${(b / 1024 ** 3).toFixed(2)} GiB`
}
