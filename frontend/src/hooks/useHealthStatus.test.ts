import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

const checkHealthMock = vi.fn()

vi.mock('../api/client', () => ({
  checkHealth: () => checkHealthMock(),
}))

import { useHealthStatus } from './useHealthStatus'

describe('useHealthStatus', () => {
  it('starts null, then reflects a healthy backend', async () => {
    checkHealthMock.mockResolvedValue(true)

    const { result } = renderHook(() => useHealthStatus())

    expect(result.current).toBeNull()
    await waitFor(() => expect(result.current).toBe(true))
  })

  it('reflects an unreachable backend', async () => {
    checkHealthMock.mockResolvedValue(false)

    const { result } = renderHook(() => useHealthStatus())

    await waitFor(() => expect(result.current).toBe(false))
  })
})
