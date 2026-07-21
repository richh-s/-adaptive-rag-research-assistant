import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import type { StreamEvent } from '../api/client'

const streamResearchMock = vi.fn()

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    streamResearch: (...args: Parameters<typeof actual.streamResearch>) => streamResearchMock(...args),
  }
})

import { useResearchStream } from './useResearchStream'

beforeEach(() => {
  streamResearchMock.mockReset()
})

describe('useResearchStream', () => {
  it('starts idle', () => {
    const { result } = renderHook(() => useResearchStream())

    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBeNull()
    expect(result.current.result).toBeNull()
    expect(result.current.visits).toEqual([])
  })

  it('accumulates progress events into visits and resolves with the final result', async () => {
    streamResearchMock.mockImplementation(async (_question: string, onEvent: (e: StreamEvent) => void) => {
      onEvent({ type: 'progress', node: 'route_query', message: 'Routing question...' })
      onEvent({ type: 'progress', node: 'decompose_query', message: 'Decomposing...' })
      onEvent({
        type: 'done',
        report: '# Report',
        route: 'vector',
        confidence_score: 0.9,
        summary: null,
      })
    })

    const { result } = renderHook(() => useResearchStream())

    await act(async () => {
      await result.current.submit('Who founded Anthropic?')
    })

    expect(result.current.visits).toEqual([
      { node: 'route_query', message: 'Routing question...', seq: 1 },
      { node: 'decompose_query', message: 'Decomposing...', seq: 2 },
    ])
    expect(result.current.result).toEqual({
      question: 'Who founded Anthropic?',
      report: '# Report',
      route: 'vector',
      confidence_score: 0.9,
      summary: null,
    })
    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('surfaces an error event without setting a result', async () => {
    streamResearchMock.mockImplementation(async (_question: string, onEvent: (e: StreamEvent) => void) => {
      onEvent({ type: 'error', detail: 'quota exceeded' })
    })

    const { result } = renderHook(() => useResearchStream())

    await act(async () => {
      await result.current.submit('Who founded Anthropic?')
    })

    expect(result.current.error).toBe('quota exceeded')
    expect(result.current.result).toBeNull()
  })

  it('ignores blank questions and never calls the API', async () => {
    const { result } = renderHook(() => useResearchStream())

    await act(async () => {
      await result.current.submit('   ')
    })

    expect(streamResearchMock).not.toHaveBeenCalled()
    expect(result.current.loading).toBe(false)
  })

  it('does not accumulate visits across separate submissions', async () => {
    streamResearchMock.mockImplementation(async (_question: string, onEvent: (e: StreamEvent) => void) => {
      onEvent({ type: 'progress', node: 'route_query', message: 'Routing question...' })
      onEvent({ type: 'done', report: 'r', route: 'vector', confidence_score: 1, summary: null })
    })

    const { result } = renderHook(() => useResearchStream())
    await act(async () => {
      await result.current.submit('first question')
    })
    expect(result.current.visits).toHaveLength(1)

    streamResearchMock.mockImplementation(async (_question: string, onEvent: (e: StreamEvent) => void) => {
      onEvent({ type: 'progress', node: 'route_query', message: 'Routing question...' })
      onEvent({ type: 'progress', node: 'decompose_query', message: 'Decomposing...' })
      onEvent({ type: 'done', report: 'r2', route: 'web', confidence_score: 1, summary: null })
    })
    await act(async () => {
      await result.current.submit('second question')
    })

    expect(result.current.visits).toHaveLength(2)
    expect(result.current.result?.report).toBe('r2')
  })
})
