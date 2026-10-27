// A small "?" that explains the value beside it on hover or keyboard focus.
export default function Hint({ text }: { text: string }) {
  return (
    <span className="hint" tabIndex={0} aria-label={text}>
      ?<span className="hint-pop" role="tooltip">{text}</span>
    </span>
  )
}
