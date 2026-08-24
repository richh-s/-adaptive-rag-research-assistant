import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import type { ConversationDetail, StreamEvent, StreamResearchOptions } from '../api/client'

const streamResearchMock = vi.fn()
const getConversationMock = vi.fn()

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    streamResearch: (...args: Parameters<typeof actual.streamResearch>) => streamResearchMock(...args),
    getConversation: (...args: Parameters<typeof actual.getConversation>) => getConversationMock(...args),
  }
})

import { useResearchStream } from './useResearchStream'

beforeEach(() => {
  streamResearchMock.mockReset()
  getConversationMock.mockReset()
})

function doneEvent(overrides: Partial<StreamEvent> = {}): StreamEvent {
  return {
    type: 'done',
    report: '# Report',
    answer: 'The answer.',
    route: 'vector',
    confidence_score: 0.9,
    summary: null,
    conversation_id: 'conv-1',
    ...overrides,
  }
}

describe('useResearchStream', () => {
  it('starts idle', () => {
    const { result } = renderHook(() => useResearchStream())

    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBeNull()
    expect(result.current.turns).toEqual([])
    expect(result.current.conversationId).toBeNull()
    expect(result.current.pendingQuestion).toBeNull()
    expect(result.current.visits).toEqual([])
  })

  it('accumulates progress events into visits and appends a completed turn', async () => {
    streamResearchMock.mockImplementation(async (_q: string, onEvent: (e: StreamEvent) => void) => {
      onEvent({ type: 'progress', node: 'route_query', message: 'Routing question...' })
      onEvent({ type: 'progress', node: 'decompose_query', message: 'Decomposing...' })
      onEvent(doneEvent())
    })

    const { result } = renderHook(() => useResearchStream())

    await act(async () => {
      await result.current.submit('Who founded Anthropic?')
    })

    expect(result.current.visits).toEqual([
      { node: 'route_query', message: 'Routing question...', seq: 1 },
      { node: 'decompose_query', message: 'Decomposing...', seq: 2 },
    ])
    expect(result.current.turns).toHaveLength(1)
    expect(result.current.turns[0].question).toBe('Who founded Anthropic?')
    expect(result.current.turns[0].result.answer).toBe('The answer.')
    expect(result.current.loading).toBe(false)
    expect(result.current.pendingQuestion).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('adopts the server conversation id and sends it on follow-ups', async () => {
    streamResearchMock.mockImplementation(async (_q: string, onEvent: (e: StreamEvent) => void) => {
      onEvent(doneEvent({ conversation_id: 'conv-42' }))
    })

    const { result } = renderHook(() => useResearchStream())
    await act(async () => {
      await result.current.submit('first question')
    })
    expect(result.current.conversationId).toBe('conv-42')

    await act(async () => {
      await result.current.submit('a follow-up')
    })

    const firstOptions = streamResearchMock.mock.calls[0][2] as StreamResearchOptions
    const secondOptions = streamResearchMock.mock.calls[1][2] as StreamResearchOptions
    expect(firstOptions.conversationId).toBeNull()
    expect(secondOptions.conversationId).toBe('conv-42')
  })

  it('surfaces an error event without appending a turn', async () => {
    streamResearchMock.mockImplementation(async (_q: string, onEvent: (e: StreamEvent) => void) => {
      onEvent({ type: 'error', detail: 'quota exceeded' })
    })

    const { result } = renderHook(() => useResearchStream())

    await act(async () => {
      await result.current.submit('Who founded Anthropic?')
    })

    expect(result.current.error).toBe('quota exceeded')
    expect(result.current.turns).toEqual([])
  })

  it('ignores blank questions and never calls the API', async () => {
    const { result } = renderHook(() => useResearchStream())

    await act(async () => {
      await result.current.submit('   ')
    })

    expect(streamResearchMock).not.toHaveBeenCalled()
    expect(result.current.loading).toBe(false)
  })

  it('openConversation loads persisted turns and continues under that id', async () => {
    const detail: ConversationDetail = {
      id: 'conv-9',
      title: 'Who founded Anthropic?',
      created_at: 1,
      updated_at: 2,
      messages: [
        { role: 'user', content: 'Who founded Anthropic?', created_at: 1 },
        { role: 'assistant', content: 'The Amodeis.', report: '# R1', summary: null, created_at: 1 },
        { role: 'user', content: 'what about their models?', created_at: 2 },
        { role: 'assistant', content: 'Claude.', report: '# R2', summary: null, created_at: 2 },
      ],
    }
    getConversationMock.mockResolvedValue(detail)

    const { result } = renderHook(() => useResearchStream())
    await act(async () => {
      await result.current.openConversation('conv-9')
    })

    expect(result.current.conversationId).toBe('conv-9')
    expect(result.current.turns).toHaveLength(2)
    expect(result.current.turns[0].question).toBe('Who founded Anthropic?')
    expect(result.current.turns[0].result.report).toBe('# R1')
    expect(result.current.turns[1].result.answer).toBe('Claude.')
  })

  it('reset clears local state so the next submit starts a new server conversation', async () => {
    streamResearchMock.mockImplementation(async (_q: string, onEvent: (e: StreamEvent) => void) => {
      onEvent(doneEvent({ conversation_id: 'conv-1' }))
    })

    const { result } = renderHook(() => useResearchStream())
    await act(async () => {
      await result.current.submit('first question')
    })
    expect(result.current.conversationId).toBe('conv-1')

    act(() => {
      result.current.reset()
    })
    expect(result.current.turns).toEqual([])
    expect(result.current.conversationId).toBeNull()

    await act(async () => {
      await result.current.submit('brand new topic')
    })
    const options = streamResearchMock.mock.calls[1][2] as StreamResearchOptions
    expect(options.conversationId).toBeNull()
  })
})
