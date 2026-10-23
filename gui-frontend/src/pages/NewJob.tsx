import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import InputChecks, { worst } from '../components/InputChecks'
import { api, CheckGroup, Driver, Info, Preview, Scan, Schema } from '../api'
import Form, { defaultsOf, Values } from '../components/Form'
import SbatchChips from '../components/SbatchChips'
import Browser, { BrowseButton } from '../components/Browser'
import RoiPicker, { parseRoi, roiValues } from '../components/RoiPicker'

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
  const [params] = useSearchParams()
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
      setSchema(s); setValues(v)
    }).catch(e => setError(String(e.message)))
  }, [driver, algorithm, d])

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
  const verdict = checks ? worst(checks) : null

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
        {selected?.kind === 'slurm' && preview && (
          <div className="sbatch">
            <span className="sbatch-label" title="the #SBATCH directives this job is submitted with; click a value to change it for this job only (it goes into the overrides field, the machine\'s template is untouched)">sbatch</span>
            <SbatchChips directives={preview.directives} templateDirectives={templatePreview?.directives ?? []} resources={resources} setResources={setResources} />
            {preview.error && <span className="error small">{preview.error}</span>}
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
        {filtered && <Form schema={filtered} values={values} setValues={setValues} scans={scans} showAll={showAll} machine={machine} afterInput={<>
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
        {(checks || checking) && <div className="runbar-checks"><InputChecks groups={checks} checking={checking} /></div>}
        {error && <p className="error small">{error}</p>}
      </div>
    </>
  )
}
