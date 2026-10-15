import type { CheckGroup, CheckStatus } from '../api'

const MARK: Record<CheckStatus, string> = { ok: '✓', warn: '!', fail: '✕', info: '·' }

export function worst(groups: CheckGroup[]): CheckStatus | null {
  let w: CheckStatus | null = null
  const rank = { info: 0, ok: 1, warn: 2, fail: 3 }
  for (const g of groups) for (const it of g.items) if (w === null || rank[it.status] > rank[w]) w = it.status
  return w
}

export default function InputChecks({ groups, checking }: { groups: CheckGroup[] | null; checking: boolean }) {
  if (!groups && !checking) return null
  return (
    <div className={'input-checks' + (checking ? ' checking' : '')}>
      {checking && !groups && <span className="note">checking inputs…</span>}
      {groups?.map(g => {
        const bad = g.items.some(i => i.status === 'fail')
        const warn = g.items.some(i => i.status === 'warn')
        return (
          <div key={g.title + g.path} className={'check-group' + (bad ? ' has-fail' : warn ? ' has-warn' : '')}>
            <div className="check-head">
              <span className="check-title">{g.title}</span>
              {g.path && <span className="path" title={g.path}>{'\u200e' + g.path + '\u200e'}</span>}
            </div>
            <ul className="check-items">
              {g.items.map((it, i) => (
                <li key={i} className={'check-item ' + it.status} title={it.detail}>
                  <span className={'check-mark ' + it.status}>{MARK[it.status]}</span>
                  <span><span className="check-label">{it.label}</span>{it.detail && <span className="check-detail"> {it.detail}</span>}</span>
                </li>
              ))}
            </ul>
          </div>
        )
      })}
    </div>
  )
}
