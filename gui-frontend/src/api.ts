// Thin client for the JSON API in gui/app.py.

export interface Gpu { index: number; name: string; total_gib: number; free_gib: number; used_gib?: number; utilisation: number | null }
// one row of the user's scheduler queue; ``job`` is set when it was submitted from this GUI
export interface QueueRow {
  id: string; name: string; state: string; elapsed: string; limit: string; nodes: string
  cpus: string; memory: string; gres: string; where: string; partition: string; job?: string | null
}
export interface MachineConfig {
  name: string; type: 'local' | 'ssh' | 'slurm'; host: string; python: string; workdir: string; scans_root: string
  slots: number; env: Record<string, string>; sbatch_template: string | null; template_is_default?: boolean
}
export interface Machine {
  name: string; kind: string; status: 'connected' | 'needs_login' | 'unreachable' | 'logged_out'; host?: string; error?: string
  hostname: string; platform: string; cpus: number | null
  memory: { total_gib: number | null; available_gib: number | null }
  gpus: Gpu[]; slots: number; running: string[]; pending: string[]; cwd: string
  package: { version: string | null; commit: string | null; python: string | null }
  queue?: { running: number; pending: number; jobs: QueueRow[] } | null
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
export interface Scan { name: string; path: string }
export interface Job {
  id: string; driver: string; algorithm: string | null; values: Record<string, any>; command: string[]
  executor: string; status: 'queued' | 'pending' | 'running' | 'finished' | 'failed' | 'cancelled'
  created: number; started: number | null; finished: number | null; duration: number | null
  note: string; input: string; cwd: string; hostname: string | null; gpu_index: number | null; gpu_name: string | null
  remote_id: string | null; remote_state: string | null; node: string | null; resources: string
  pid: number | null; returncode: number | null; log_path: string | null; output_path: string | null
  wandb_url: string | null; package_version: string | null; error: string | null; shell?: string
}

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
  checks: (driver: string, algorithm: string | null, values: Record<string, any>, executor: string) =>
    request<{ groups: CheckGroup[] }>(`/api/drivers/${driver}/checks`, { method: 'POST', body: JSON.stringify({ algorithm, values, executor }) }),
  files: (id: string) => request<{ files: { path: string; dir: boolean; bytes: number }[]; output_path: string | null }>(`/api/jobs/${id}/files`),
  remove: (id: string, data: boolean) => request<{ deleted: string; files: any[] }>(`/api/jobs/${id}?data=${data}`, { method: 'DELETE' }),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
}

export function fmtDuration(s: number | null): string {
  if (s == null) return ''
  if (s < 90) return `${Math.round(s)} s`
  if (s < 5400) return `${(s / 60).toFixed(1)} min`
  return `${(s / 3600).toFixed(2)} h`
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
