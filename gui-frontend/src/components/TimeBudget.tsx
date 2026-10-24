// The run-time estimate of a learned job against the Slurm --time it would be
// submitted with: the estimate (from the page's /api/drivers/learned/estimate
// call), the request, and a one-click suggestion when the request is short.
// A request below the estimate is the run that ends in a TIMEOUT.
import type { CheckGroup, Directive, Estimate } from '../api'
import { fmtDuration } from '../api'
import { withOverride } from './SbatchChips'

/** Seconds from a Slurm time string (M, M:S, H:M:S, D-H, D-H:M, D-H:M:S), or null. */
export function parseSlurmTime(text: string | null | undefined): number | null {
  if (!text) return null
  const t = text.trim()
  if (!/^(\d+-)?\d+(:\d+){0,2}$/.test(t)) return null
  let days = 0, h = 0, m = 0, s = 0
  if (t.includes('-')) {
    const [d, rest] = t.split('-', 2)
    days = Number(d)
    const parts = rest.split(':').map(Number)
    while (parts.length < 3) parts.push(0)
    ;[h, m, s] = parts
  } else {
    const parts = t.split(':').map(Number)
    if (parts.length === 1) m = parts[0]
    else if (parts.length === 2) [m, s] = parts
    else [h, m, s] = parts
  }
  return days * 86400 + h * 3600 + m * 60 + s
}

/** How the requested --time compares with the estimate: the check item the
 * submit bar shows, or null when there is nothing to compare. */
export function walltimeCheck(est: Estimate | null, directives: Directive[] | null): { group: CheckGroup; short: boolean; suggested: string } | null {
  if (!est?.walltime || !directives) return null
  const w = est.walltime
  const d = directives.find(x => x.key === 'time')
  const asked = parseSlurmTime(d?.value)
  const where = d ? (d.source === 'override' ? 'this job\'s override' : 'the machine\'s template') : null
  let item
  if (asked == null) {
    item = { label: 'walltime', status: 'warn' as const, detail: `no --time is set; the estimate is ${fmtDuration(w.estimate_s)}, suggested --time=${w.suggested} (${est.rate_source})` }
  } else if (asked < w.estimate_s) {
    item = { label: 'walltime', status: 'warn' as const, detail: `--time=${d!.value} (${where}) is below the estimated ${fmtDuration(w.estimate_s)} (${est.rate_source}); the run would hit the limit — suggested ${w.suggested}` }
  } else if (asked < w.raw_s) {
    item = { label: 'walltime', status: 'info' as const, detail: `--time=${d!.value} (${where}) leaves ${fmtDuration(asked - w.estimate_s)} over the estimated ${fmtDuration(w.estimate_s)}; ${w.suggested} would add the usual margin` }
  } else {
    item = { label: 'walltime', status: 'ok' as const, detail: `--time=${d!.value} covers the estimated ${fmtDuration(w.estimate_s)} (${est.rate_source})` }
  }
  return { group: { title: 'Walltime', path: '', items: [item] }, short: item.status === 'warn', suggested: w.suggested }
}

export default function TimeBudget({ est, state, error, directives, resources, setResources }: {
  est: Estimate | null; state: 'idle' | 'busy' | 'error'; error: string
  directives: Directive[] | null; resources: string; setResources: (s: string) => void
}) {
  if (state === 'error') return <span className="time-budget error small">time estimate unavailable: {error}</span>
  if (!est) return state === 'busy' ? <span className="time-budget faint small">estimating the run time…</span> : null
  if (est.seconds == null || !est.walltime) return <span className="time-budget faint small">run time: {est.rate_source}</span>
  const check = walltimeCheck(est, directives)
  const asked = directives?.find(x => x.key === 'time')?.value ?? null
  const cls = check?.short ? 'warn' : ''
  return (
    <span className={'time-budget small ' + (state === 'busy' ? 'faint' : '')} title={`${est.iterations.toLocaleString()} iterations at ${est.seconds_per_it?.toFixed(2)} s each plus the fixed part; the suggestion is the estimate x 1.25 + 10 min, rounded up to a quarter hour (${est.rate_source})`}>
      run time ≈ <b className={cls}>{fmtDuration(est.seconds)}</b>
      <span className="faint"> ({est.rate_source})</span>
      {directives && (
        check?.short
          ? <> · <span className="warn">--time={asked ?? 'unset'} is short</span> <button type="button" className="small" onClick={() => setResources(withOverride(resources, 'time', check.suggested))}>use {check.suggested}</button>
              <span className="faint"> · the {est.walltime.bucket_h} h class</span></>
          : <> · fits <span className="mono">--time={asked}</span>{asked !== est.walltime.suggested && parseSlurmTime(asked)! > est.walltime.suggested_s * 2
              ? <> <button type="button" className="quiet small" title="the request is more than twice the suggestion; a shorter one schedules sooner" onClick={() => setResources(withOverride(resources, 'time', est.walltime!.suggested))}>use {est.walltime.suggested}</button></> : null}</>
      )}
    </span>
  )
}
