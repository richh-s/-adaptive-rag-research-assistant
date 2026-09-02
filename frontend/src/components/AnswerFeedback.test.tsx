import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ResearchResponse } from '../api/client'

const submitFeedbackMock = vi.fn()

// Mocks the client module rather than global fetch, matching the other component tests --
// the unit under test is the component's behaviour, not its serialisation.
vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return { ...actual, submitFeedback: (...args: unknown[]) => submitFeedbackMock(...args) }
})

const { AnswerFeedback } = await import('./AnswerFeedback')

const result: ResearchResponse = {
  question: 'Who founded Anthropic?',
  report: '# Answer',
  answer: 'Dario and Daniela Amodei.',
  route: 'vector',
  confidence_score: 0.87,
  conversation_id: 'conv-1',
}

function lastBody() {
  return submitFeedbackMock.mock.calls.at(-1)?.[0]
}

describe('AnswerFeedback', () => {
  beforeEach(() => {
    submitFeedbackMock.mockReset()
    submitFeedbackMock.mockResolvedValue(true)
  })

  it('offers both ratings', () => {
    render(<AnswerFeedback question={result.question} result={result} />)

    expect(screen.getByRole('button', { name: /yes/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /no/i })).toBeInTheDocument()
  })

  it('submits an upvote with the answer context attached', async () => {
    render(<AnswerFeedback question={result.question} result={result} />)

    await userEvent.click(screen.getByRole('button', { name: /yes/i }))

    await waitFor(() => expect(submitFeedbackMock).toHaveBeenCalled())
    const body = lastBody()
    expect(body.rating).toBe('up')
    expect(body.question).toBe('Who founded Anthropic?')
    // Route and confidence travel with the rating so a downvote can later be correlated with
    // how the answer was produced -- which is the whole point of collecting it.
    expect(body.route).toBe('vector')
    expect(body.confidence_score).toBe(0.87)
    expect(body.conversation_id).toBe('conv-1')
  })

  it('marks the chosen rating with aria-pressed for screen readers', async () => {
    render(<AnswerFeedback question={result.question} result={result} />)

    await userEvent.click(screen.getByRole('button', { name: /yes/i }))

    expect(screen.getByRole('button', { name: /yes/i })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: /no/i })).toHaveAttribute('aria-pressed', 'false')
  })

  it('confirms the rating was recorded', async () => {
    render(<AnswerFeedback question={result.question} result={result} />)

    await userEvent.click(screen.getByRole('button', { name: /no/i }))

    expect(screen.getByText(/recorded/i)).toBeInTheDocument()
  })

  it('does not resend the same rating when clicked twice', async () => {
    render(<AnswerFeedback question={result.question} result={result} />)

    await userEvent.click(screen.getByRole('button', { name: /yes/i }))
    await userEvent.click(screen.getByRole('button', { name: /yes/i }))

    expect(submitFeedbackMock).toHaveBeenCalledTimes(1)
  })

  it('asks for a note only on a downvote', async () => {
    render(<AnswerFeedback question={result.question} result={result} />)

    await userEvent.click(screen.getByRole('button', { name: /yes/i }))
    expect(screen.queryByPlaceholderText(/what was wrong/i)).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /no/i }))
    expect(screen.getByPlaceholderText(/what was wrong/i)).toBeInTheDocument()
  })

  it('records the downvote immediately, without waiting for a note', async () => {
    // Demanding a note before accepting the rating would cost most of the ratings.
    render(<AnswerFeedback question={result.question} result={result} />)

    await userEvent.click(screen.getByRole('button', { name: /no/i }))

    await waitFor(() => expect(submitFeedbackMock).toHaveBeenCalledTimes(1))
    expect(lastBody().note).toBeNull()
  })

  it('sends the note when one is supplied', async () => {
    render(<AnswerFeedback question={result.question} result={result} />)
    await userEvent.click(screen.getByRole('button', { name: /no/i }))

    await userEvent.type(screen.getByPlaceholderText(/what was wrong/i), 'cited the wrong file')
    await userEvent.click(screen.getByRole('button', { name: /^send$/i }))

    await waitFor(() => expect(submitFeedbackMock).toHaveBeenCalledTimes(2))
    expect(lastBody().note).toBe('cited the wrong file')
    expect(screen.getByText(/note sent/i)).toBeInTheDocument()
  })

  it('does not offer to send an empty note', async () => {
    render(<AnswerFeedback question={result.question} result={result} />)

    await userEvent.click(screen.getByRole('button', { name: /no/i }))

    expect(screen.getByRole('button', { name: /^send$/i })).toBeDisabled()
  })

  it('stays silent when the request fails', async () => {
    // Feedback is a side channel: the user already has their answer, and an error over it
    // would be a worse experience than failing to record one opinion.
    submitFeedbackMock.mockRejectedValue(new Error('offline'))
    render(<AnswerFeedback question={result.question} result={result} />)

    await userEvent.click(screen.getByRole('button', { name: /yes/i }))

    expect(screen.getByText(/recorded/i)).toBeInTheDocument()
    expect(screen.queryByText(/error/i)).not.toBeInTheDocument()
  })
})
