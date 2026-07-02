import './ErrorBanner.css'

interface ErrorBannerProps {
  message: string
}

export function ErrorBanner({ message }: ErrorBannerProps) {
  return <div className="error">{message}</div>
}
