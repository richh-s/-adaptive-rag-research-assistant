import './ErrorBanner.css'

interface ErrorBannerProps {
  message: string
  /** When provided, renders a "Try again" action beside the message. */
  onRetry?: () => void
}

export function ErrorBanner({ message, onRetry }: ErrorBannerProps) {
  return (
    <div className="error" role="alert">
      <span>{message}</span>
      {onRetry && (
        <button type="button" className="error-retry" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}
