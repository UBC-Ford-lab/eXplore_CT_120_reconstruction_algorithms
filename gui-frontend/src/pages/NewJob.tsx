import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import InputChecks, { worst } from '../components/InputChecks'
import { api, CheckGroup, Driver, Estimate, Info, Job, Preview, Scan, Schema } from '../api'
import Form, { defaultsOf, Values } from '../components/Form'
import SbatchChips from '../components/SbatchChips'
import Browser, { BrowseButton } from '../components/Browser'
import RoiPicker, { parseRoi, roiValues } from '../components/RoiPicker'
import DetailPresets, { DETAIL_ALGORITHMS, estimateKey } from '../components/DetailPresets'
import TimeBudget, { walltimeCheck } from '../components/TimeBudget'

// Driver groups in the dropdown, in this order.
const KINDS = [
  { kind: 'reconstruction', label: 'Reconstruction' },
  { kind: 'report', label: 'Reports and post-processing' },
  { kind: 'calibration', label: 'Calibration' },
  { kind: 'simulation', label: 'Simulation' },
]

export default function NewJob({ info }: { info: Info | null }) {
  const nav = useNavigate()
  // ?driver=learned&algorithm=gaussian&input=data/scans/Scan_1510&machine=fir&all=1 pre-fills the page; &roi=pick opens the ROI picker
  // ?from=<job id> pre-fills driver, algorithm, machine and every value from that job; with &resume=<cloud.pt>
  // the run continues from that checkpoint (--gauss-resume) into the same output
  const [params] = useSearchParams()
  const [fromJob, setFromJob] = useState<Job | null>(null)
  const [drivers, setDrivers] = useState<Driver[]>([])
  const [scans, setScans] = useState<Scan[]>([])
  const [driver, setDriver] = useState(params.get('driver') ?? 'fdk')
  const [algorithm, setAlgorithm] = useState<string | null>(params.get('algorithm'))
  const [schema, setSchema] = useState<Schema | null>(null)
  const [values, setValues] = useState<Values>({})
  const [note, setNote] = useState('')
  const [machine, setMachine] = useState(params.get('machine') ?? 'local')
  const [resources, setResources] = useState(params.get('resources') ?? '')
  const [predicted, setPredicted] = useState<{ dest?: string | null; path?: string; writes?: string[]; explicit?: boolean } | null>(null)
  const [showAll, setShowAll] = useState(params.get('all') === '1')
  const [checks, setChecks] = useState<CheckGroup[] | null>(null)
  const [est, setEst] = useState<Estimate | null>(null)                    // the learned job's cost (gui/detail.py)
  const [estState, setEstState] = useState<'idle' | 'busy' | 'error'>('idle')
  const [estError, setEstError] = useState('')
  const [preview, setPreview] = useState<Preview | null>(null)
  const [templatePreview, setTemplatePreview] = useState<Preview | null>(null)   // the same without overrides
  const [showScript, setShowScript] = useState(false)
  const [browseOut, setBrowseOut] = useState(false)
  const [pickingRoi, setPickingRoi] = useState(params.get('roi') === 'pick')
  const [checking, setChecking] = useState(false)
  const [filter, setFilter] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const bar = useRef<HTMLDivElement>(null)
  const [barHeight, setBarHeight] = useState(0)
  useLayoutEffect(() => {
    const el = bar.current
    if (!el) return
    const ro = new ResizeObserver(() => setBarHeight(el.offsetHeight))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => { api.drivers().then(setDrivers) }, [])
  useEffect(() => {
    const id = params.get('from')
    if (!id) return
    api.job(id).then(j => {
      setFromJob(j)
      setDriver(j.driver); setAlgorithm(j.algorithm); setMachine(j.executor); setResources(j.resources ?? '')
      if (params.get('resume')) setNote(`resumed job ${j.id}${j.note ? ` · ${j.note}` : ''}`)
    }).catch(e => setError(String(e.message)))
  }, [])
  useEffect(() => { setScans([]); api.scans(machine).then(setScans).catch(() => setScans([])) }, [machine])
  const d = drivers.find(x => x.name === driver)
  useEffect(() => {
    if (!d) return
    const algo = d.algorithms.length ? (algorithm && d.algorithms.some(a => a.name === algorithm) ? algorithm : d.algorithms[0].name) : null
    if (algo !== algorithm) { setAlgorithm(algo); return }
    setSchema(null)
    api.schema(driver, algo).then(s => {
      const v = defaultsOf(s)
      const input = params.get('input')
      if (input && d.primary in v) v[d.primary] = input
      if (fromJob && fromJob.driver === driver && (fromJob.algorithm ?? null) === algo) {
        const known = new Set(s.fields.map(f => f.dest))
        for (const [k, val] of Object.entries(fromJob.values)) if (known.has(k)) v[k] = val
        const resume = params.get('resume')
        if (resume && known.has('gauss_resume')) {
          v.gauss_resume = resume
          if (!v.output && fromJob.output_path && known.has('output')) v.output = fromJob.output_path
        }
      }
      setSchema(s); setValues(v)
    }).catch(e => setError(String(e.message)))
  }, [driver, algorithm, d, fromJob])

  useEffect(() => {
    if (!schema) return
    const t = setTimeout(() => api.output(driver, algorithm, values).then(setPredicted).catch(() => setPredicted(null)), 250)
    return () => clearTimeout(t)
  }, [schema, values, driver, algorithm])
  useEffect(() => {
    if (!schema) { setChecks(null); return }
    const primary = values[d?.primary ?? 'data_folder']
    const filled = Array.isArray(primary) ? primary.some(Boolean) : Boolean(primary)
    if (!filled) { setChecks(null); setChecking(false); return }
    setChecking(true)
    let live = true
    const t = setTimeout(() => api.checks(driver, algorithm, values, machine)
      .then(r => { if (live) { setChecks(r.groups); setChecking(false) } })
      .catch(() => { if (live) { setChecks(null); setChecking(false) } }), 500)
    return () => { live = false; clearTimeout(t) }
  }, [schema, values, driver, algorithm, machine])
  const selectedKind = info?.executors.find(m => m.name === machine)?.kind
  useEffect(() => {
    // on a Slurm machine, the sbatch script this form would submit
    if (!schema || selectedKind !== 'slurm') { setPreview(null); return }
    let live = true
    const t = setTimeout(() => {
      api.preview({ driver, algorithm, values, executor: machine, resources })
        .then(p => { if (live) setPreview(p) })
        .catch(e => { if (live) setPreview({ command: '', cwd: '', script: null, directives: [], error: `preview unavailable: ${e.message}` }) })
      api.preview({ driver, algorithm, values, executor: machine, resources: '' })
        .then(p => { if (live) setTemplatePreview(p) }).catch(() => { if (live) setTemplatePreview(null) })
    }, 300)
    return () => { live = false; clearTimeout(t) }
  }, [schema, values, driver, algorithm, machine, resources, selectedKind])
  // the run-time (and, for a Gaussian job, primitives/VRAM) estimate; the
  // detail row and the walltime line both read it
  const scanPathNow = String(values[d?.primary ?? 'data_folder'] ?? '')
  const estKey = estimateKey(values, machine, algorithm ?? '', resources)
  useEffect(() => {
    if (!schema || driver !== 'learned' || !algorithm || !scanPathNow.trim()) { setEst(null); setEstState('idle'); return }
    let live = true
    setEstState('busy')
    const t = setTimeout(() => api.estimate(algorithm, values, machine, resources)
      .then(e => { if (live) { setEst(e); setEstState('idle') } })
      .catch(e => { if (live) { setEstError(String(e.message)); setEstState('error') } }), 700)
    return () => { live = false; clearTimeout(t) }
  }, [schema, driver, estKey, scanPathNow])
  const walltime = selectedKind === 'slurm' ? walltimeCheck(est, preview?.directives ?? null) : null
  const allChecks = checks || walltime ? [...(checks ?? []), ...(walltime ? [walltime.group] : [])] : null
  const verdict = allChecks ? worst(allChecks) : null

  const outDest = predicted?.dest ?? 'output'
  const outField = predicted?.dest === null ? undefined : schema?.fields.find(f => f.dest === outDest)
  const outValue: string = values[outDest] ?? ''
  const outShown = outValue || predicted?.path || ''
  function setOutput(text: string) {
    // typing the automatic path back, or clearing, returns to the driver's own naming
    const next = { ...values }
    if (!text || (!predicted?.explicit && text === predicted?.path)) delete next[outDest]; else next[outDest] = text
    setValues(next)
  }

  const filtered = useMemo(() => {
    if (!schema || !filter.trim()) return schema
    const q = filter.toLowerCase()
    return { ...schema, fields: schema.fields.filter(f => f.positional || f.dest.includes(q) || f.flags.some(x => x.includes(q)) || f.help.toLowerCase().includes(q)) }
  }, [schema, filter])

  const changedCount = schema ? schema.fields.filter(f => !f.positional && JSON.stringify(values[f.dest]) !== JSON.stringify(f.default)).length : 0

  async function submit() {
    if (!schema) return
    setBusy(true); setError(null)
    try {
      const job = await api.submit({ driver, algorithm, values, executor: machine, note, resources })
      nav(`/jobs/${job.id}`)
    } catch (e: any) { setError(e.message) } finally { setBusy(false) }
  }

  const executors = info?.executors ?? []
  const selected = executors.find(m => m.name === machine)
  const algoSummary = d && algorithm ? d.algorithms.find(a => a.name === algorithm)?.summary : null

  // --roi, drawn on a coarse preview of the scan (reconstruction drivers only)
  const roiField = d?.kind === 'reconstruction' ? schema?.fields.find(f => f.dest === 'roi') : undefined
  const scanPath: string = roiField ? String(values[d!.primary] ?? '') : ''
  const roiRaw = values.roi
  const roiText: string = Array.isArray(roiRaw) ? roiRaw.join(' ') : (roiRaw ?? '')
  const roiBounds = parseRoi(roiRaw)
  const voxelXY = Number(values.voxel_xy ?? schema?.fields.find(f => f.dest === 'voxel_xy')?.default ?? 0.075) || 0.075
  const voxelZ = Number(values.voxel_z ?? schema?.fields.find(f => f.dest === 'voxel_z')?.default ?? 0.075) || 0.075
  const roiSummary = roiBounds
    ? `x [${roiBounds.x_min}, ${roiBounds.x_max}]  y [${roiBounds.y_min}, ${roiBounds.y_max}]  z [${roiBounds.z_min}, ${roiBounds.z_max}] mm · `
      + `${Math.round((roiBounds.x_max - roiBounds.x_min) / voxelXY)} × ${Math.round((roiBounds.y_max - roiBounds.y_min) / voxelXY)} × ${Math.round((roiBounds.z_max - roiBounds.z_min) / voxelZ)} voxels`
    : roiText.trim() === 'full' ? 'the whole reconstructible field'
    : roiText.trim() === '' || roiText.trim() === 'auto' ? "auto: the scanner's own ROI (SubVolumeCoordinates.xml), or the whole field without one"
    : roiField?.help ?? ''
  return (
    <>
      <div className="page-head">
        <h1>New job</h1>
        <div className="ctl-row">
          <label className="ctl"><span>Driver</span>
            <select value={driver} onChange={e => { setDriver(e.target.value); setAlgorithm(null) }}>
              {KINDS.filter(k => drivers.some(x => x.kind === k.kind)).map(k => (
                <optgroup key={k.kind} label={k.label}>
                  {drivers.filter(x => x.kind === k.kind).map(x => <option key={x.name} value={x.name}>{x.title}</option>)}
                </optgroup>
              ))}
            </select>
          </label>
          {d && d.algorithms.length > 0 && (
            <label className="ctl"><span>Algorithm</span>
              <select value={algorithm ?? ''} onChange={e => setAlgorithm(e.target.value)}>
                {d.algorithms.map(a => <option key={a.name} value={a.name}>{a.name}</option>)}
              </select>
            </label>
          )}
          <label className="ctl"><span>Runs on</span>
            <select value={machine} onChange={e => setMachine(e.target.value)}>
              {executors.map(m => <option key={m.name} value={m.name} disabled={m.status !== 'connected'}>
                {m.name}{m.kind !== 'local' ? ` · ${m.kind}` : ` · ${m.hostname}`}{m.status !== 'connected' ? ` (${m.status.replace('_', ' ')})` : ''}
              </option>)}
            </select>
          </label>
          {selected?.kind === 'slurm' && (
            <label className="ctl"><span>sbatch overrides</span>
              <input type="text" value={resources} onChange={e => setResources(e.target.value)} placeholder="--mem=96G --time=12:00:00" style={{ width: '20em' }} />
            </label>
          )}
        </div>
        {d && <span className="faint mono small right">{d.script}</span>}
      </div>

      <div className="newjob" style={{ paddingBottom: barHeight + 12 }}>
        {schema?.description && <p className="blurb">{schema.description}{algoSummary ? <> <b>{algorithm}</b>: {algoSummary}.</> : null}</p>}
        {selected && selected.kind !== 'local' && <p className="note">Paths are on {selected.host}; relative paths resolve in <span className="mono">{selected.cwd}</span>.</p>}
        {selectedKind !== 'slurm' && driver === 'learned' && algorithm && !DETAIL_ALGORITHMS.includes(algorithm) && (est || estState !== 'idle') && (
          <p className="note"><TimeBudget est={est} state={estState} error={estError} directives={null} resources={resources} setResources={setResources} /></p>
        )}
        {selected?.kind === 'slurm' && preview && (
          <div className="sbatch">
            <span className="sbatch-label" title="the #SBATCH directives this job is submitted with; click a value to change it for this job only (it goes into the overrides field, the machine\'s template is untouched)">sbatch</span>
            <SbatchChips directives={preview.directives} templateDirectives={templatePreview?.directives ?? []} resources={resources} setResources={setResources} />
            {preview.error && <span className="error small">{preview.error}</span>}
            {driver === 'learned' && <TimeBudget est={est} state={estState} error={estError} directives={preview.directives} resources={resources} setResources={setResources} />}
            <button className="quiet small" onClick={() => setShowScript(v => !v)}>{showScript ? 'hide script' : 'script'}</button>
            {showScript && preview.script && <pre className="script">{preview.script}</pre>}
          </div>
        )}
        <div className="formbar">
          <label><input type="checkbox" checked={showAll} onChange={e => setShowAll(e.target.checked)} /> every flag{schema ? ` (${schema.fields.length})` : ''}</label>
          <input type="search" placeholder="find a flag…" value={filter} onChange={e => { setFilter(e.target.value); if (e.target.value) setShowAll(true) }} />
          <span className="right">{changedCount} changed from the defaults</span>
        </div>
        {!schema && !error && <p className="muted">Loading the form…</p>}
        {fromJob && schema && (
          <p className="note">{params.get('resume')
            ? <>Continuing job <span className="mono">{fromJob.id}</span> from its checkpoint: same flags and output, <span className="mono">--gauss-resume</span> added. Training picks up at the iteration the checkpoint records and runs to <span className="mono">--iterations</span>.</>
            : <>Flags copied from job <span className="mono">{fromJob.id}</span>.</>}</p>
        )}
        {filtered && <Form schema={filtered} values={values} setValues={setValues} scans={scans} showAll={showAll} machine={machine} afterInput={<>
          {driver === 'learned' && algorithm && DETAIL_ALGORITHMS.includes(algorithm) && schema && (
            <DetailPresets schema={schema} values={values} setValues={setValues} machine={machine} machines={executors} scanPath={scanPathNow} est={est} state={estState} error={estError} />
          )}
          {roiField && (
            <div className={'opt pos roi-row' + (roiText.trim() && roiText.trim() !== 'auto' ? ' changed' : '')} title={roiField.help}>
              <label htmlFor="f-roi-row" className="flag">{roiField.flags[0]}</label>
              <div className="input">
                <input id="f-roi-row" type="text" value={roiText} placeholder="auto"
                  onChange={e => { const t = e.target.value; setValues({ ...values, roi: t.trim() === '' ? null : t.split(/\s+/).filter(Boolean) }) }} />
                <button type="button" className="small" disabled={!scanPath.trim()} onClick={() => setPickingRoi(true)}
                  title={scanPath.trim() ? 'draw the box on a coarse preview of this scan' : 'give the scan folder first'}>choose visually…</button>
              </div>
              <div className="help">{roiSummary}</div>
            </div>
          )}
          <div className="opt pos note-row">
            <label htmlFor="f-note" className="flag">note</label>
            <div className="input"><input id="f-note" type="text" value={note} onChange={e => setNote(e.target.value)} placeholder="what this job is for" /></div>
            <div className="help">Free text stored with the job record; not passed to the driver.</div>
          </div>
        </>} />}
        {pickingRoi && roiField && (
          <RoiPicker machine={machine} dataFolder={scanPath} scanFolder={values.scan_folder ?? null} phase={String(values.phase ?? '00')}
            voxelXY={voxelXY} voxelZ={voxelZ} initial={roiBounds}
            onClose={() => setPickingRoi(false)}
            onPick={b => { setPickingRoi(false); setValues({ ...values, roi: roiValues(b) }) }} />
        )}
      </div>

      <div className="runbar" ref={bar}>
        <div className="runbar-main">
          {outField ? (
            <label className="ctl out"><span>Output</span>
              <span className="with-browse">
                <input type="text" value={outShown} onChange={e => setOutput(e.target.value)} placeholder="set the input first" />
                <BrowseButton onClick={() => setBrowseOut(true)} title="choose where to write" />
              </span>
              {browseOut && <Browser machine={machine} mode="save" title="Output" initial={outShown} onClose={() => setBrowseOut(false)}
                onPick={p => { setBrowseOut(false); setOutput(p) }} />}
              <span className="note">
                {outValue ? <>passed as <span className="mono">{outField.flags[0]}</span></> : 'automatic from the driver\'s naming rule; edit to override'}
                {predicted?.writes?.length ? <> · writes {predicted.writes.join(', ')}</> : null}
              </span>
            </label>
          ) : predicted?.dest === null && predicted.writes?.length ? (
            <div className="ctl out"><span>Output</span><span className="note">writes {predicted.writes[0]}</span></div>
          ) : <div className="ctl out" />}
          <div className="submit">
            <button className={'primary' + (verdict === 'fail' ? ' danger' : '')} disabled={!schema || busy} onClick={submit}>Submit</button>
            <span className="submit-where">
              {selected && <span className="faint small">on {selected.name}</span>}
              {verdict === 'fail' && <span className="verdict fail">inputs have problems</span>}
              {verdict === 'warn' && <span className="verdict warn">see the warnings</span>}
            </span>
          </div>
        </div>
        {(allChecks || checking) && <div className="runbar-checks"><InputChecks groups={allChecks} checking={checking} /></div>}
        {error && <p className="error small">{error}</p>}
      </div>
    </>
  )
}
