import { useEffect, useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { api, Info, ServerInfo } from './api'
import Machines from './pages/Machines'
import NewJob from './pages/NewJob'
import Jobs from './pages/Jobs'
import JobDetail from './pages/JobDetail'

export default function App() {
  const [info, setInfo] = useState<Info | null>(null)
  const [server, setServer] = useState<ServerInfo | null>(null)
  const [stopped, setStopped] = useState(false)
  const load = () => {
    api.info().then(setInfo).catch(() => setInfo(null))
    api.server().then(setServer).catch(() => setServer(null))
  }
  useEffect(() => {
    load()
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [])
  async function stopServer() {
    const n = server?.active ?? 0
    const msg = `Stop the server${n ? ` while ${n} job${n === 1 ? ' is' : 's are'} active` : ''}? Jobs keep running on their machines; ` +
      'the page goes away until ct120-gui (or the desktop icon) starts a server again.'
    if (!confirm(msg)) return
    try { await api.stopServer(); setStopped(true) } catch (e: any) { alert(e.message) }
  }
  if (stopped) {
    return (
      <div className="stopped">
        <h1>Server stopped</h1>
        <p>Jobs that were running continue on their machines and are picked up again next time.</p>
        <p>Start again from the desktop icon, or in a terminal:</p>
        <pre>ct120-gui</pre>
      </div>
    )
  }
  const local = info?.executors.find(e => e.name === 'local')
  const active = info ? info.executors.reduce((n, m) => n + m.running.length + m.pending.length, 0) : 0
  return (
    <div className="app">
      <aside className="rail">
        <div className="brand"><b>eXplore CT 120</b><span>reconstruction</span></div>
        <nav>
          <NavLink to="/new">New job</NavLink>
          <NavLink to="/jobs">Jobs {active > 0 && <span className="count">{active} active</span>}</NavLink>
          <NavLink to="/" end>Machines {info && <span className="count">{info.executors.length}</span>}</NavLink>
        </nav>
        <div className="rail-machines">
          <h3>Machines</h3>
          {info?.executors.map(m => (
            <NavLink key={m.name} to="/" end title={`${m.status.replace('_', ' ')}${m.kind !== 'local' ? ` · ${m.host}` : ''}`}>
              <span className={`dot ${m.status}`} />{m.name}
              <span className="faint">{m.running.length + m.pending.length > 0 ? `${m.running.length + m.pending.length} job${m.running.length + m.pending.length === 1 ? '' : 's'}` : m.kind}</span>
            </NavLink>
          ))}
          {!info && <span className="error small">server unreachable</span>}
        </div>
        <div className="rail-foot">
          <span>{local ? `${local.hostname} · v${info!.version}` : ''}</span>
          {server?.stoppable && <button className="quiet" onClick={stopServer} title={`stop the server (pid ${server.pid}); jobs keep running`}>stop server</button>}
        </div>
      </aside>
      <main>
        <Routes>
          <Route path="/" element={<Machines info={info} reload={load} />} />
          <Route path="/new" element={<NewJob info={info} />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/jobs/:id" element={<JobDetail />} />
        </Routes>
      </main>
    </div>
  )
}
