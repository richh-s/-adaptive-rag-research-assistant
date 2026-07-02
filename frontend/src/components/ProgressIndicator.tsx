import './ProgressIndicator.css'

interface ProgressIndicatorProps {
  messages: string[]
}

export function ProgressIndicator({ messages }: ProgressIndicatorProps) {
  if (messages.length === 0) return null

  return (
    <div className="progress">
      <ul className="progress-list">
        {messages.map((message, i) => (
          <li key={i} className={i === messages.length - 1 ? 'progress-current' : ''}>
            {message}
          </li>
        ))}
      </ul>
    </div>
  )
}
