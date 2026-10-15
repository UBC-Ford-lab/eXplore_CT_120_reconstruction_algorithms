import { useEffect, useRef, useState } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import '@xterm/xterm/css/xterm.css'

// Relays the machine's interactive SSH login (password, Duo passcode or push)
// between the browser and the server's pty. Nothing typed here is stored.
export default function LoginTerminal({ machine, onDone }: { machine: string; onDone: (status: string) => void }) {
  const host = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<'open' | 'closed'>('open')
  const [result, setResult] = useState<string | null>(null)
  useEffect(() => {
    if (!host.current) return
    const term = new Terminal({ cursorBlink: true, fontSize: 13, convertEol: true, theme: { background: '#15171a' } })
    const fit = new FitAddon()
    term.loadAddon(fit)
    term.open(host.current)
    fit.fit()
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/api/machines/${encodeURIComponent(machine)}/login`)
    ws.onopen = () => ws.send(JSON.stringify({ resize: [term.rows, term.cols] }))
    ws.onmessage = ev => {
      const text: string = ev.data
      if (text.startsWith('{"exit"')) {
        const d = JSON.parse(text)
        setResult(d.status)
        setState('closed')
        term.write(`\r\n[ssh exited with ${d.exit}; machine is ${d.status}]\r\n`)
        onDone(d.status)
        return
      }
      term.write(text)
    }
    ws.onclose = () => setState('closed')
    term.onData(data => { if (ws.readyState === WebSocket.OPEN) ws.send(data) })
    const onResize = () => { fit.fit(); if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ resize: [term.rows, term.cols] })) }
    window.addEventListener('resize', onResize)
    term.focus()
    return () => { window.removeEventListener('resize', onResize); ws.close(); term.dispose() }
  }, [machine])
  return (
    <div className="login-terminal">
      <p className="muted small">Answer the host's prompts here (password, passcode or Duo option). The connection stays open afterwards; nothing you type is stored.</p>
      <div ref={host} style={{ height: '260px' }} />
      {state === 'closed' && <p className={result === 'connected' ? 'ok' : 'error'}>{result === 'connected' ? 'Connected.' : `Login ended: ${result ?? 'closed'}.`}</p>}
    </div>
  )
}
