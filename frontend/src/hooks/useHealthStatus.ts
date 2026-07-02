import { useEffect, useState } from 'react'
import { checkHealth } from '../api/client'

export function useHealthStatus(): boolean | null {
  const [backendUp, setBackendUp] = useState<boolean | null>(null)

  useEffect(() => {
    checkHealth().then(setBackendUp)
  }, [])

  return backendUp
}
