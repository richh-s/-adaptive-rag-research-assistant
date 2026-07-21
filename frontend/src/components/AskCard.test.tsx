import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { AskCard } from './AskCard'

describe('AskCard', () => {
  it('disables the submit button until a question is entered', () => {
    render(
      <AskCard question="" onQuestionChange={() => {}} onSubmit={() => {}} loading={false} />,
    )

    expect(screen.getByRole('button', { name: /ask/i })).toBeDisabled()
  })

  it('enables submit once a question is present and calls onSubmit', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn((e) => e.preventDefault())

    render(
      <AskCard
        question="Who founded Anthropic?"
        onQuestionChange={() => {}}
        onSubmit={onSubmit}
        loading={false}
      />,
    )

    const button = screen.getByRole('button', { name: /ask/i })
    expect(button).toBeEnabled()
    await user.click(button)

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('fills in the textarea when an example question is clicked', async () => {
    const user = userEvent.setup()
    const onQuestionChange = vi.fn()

    render(
      <AskCard question="" onQuestionChange={onQuestionChange} onSubmit={() => {}} loading={false} />,
    )

    const firstExample = screen.getAllByRole('button', { name: /.+/ }).find((btn) =>
      btn.className.includes('example-chip'),
    )
    expect(firstExample).toBeDefined()
    await user.click(firstExample!)

    expect(onQuestionChange).toHaveBeenCalled()
  })

  it('shows a loading label and disables the button while loading', () => {
    render(
      <AskCard
        question="Who founded Anthropic?"
        onQuestionChange={() => {}}
        onSubmit={() => {}}
        loading={true}
      />,
    )

    expect(screen.getByRole('button', { name: /researching/i })).toBeDisabled()
  })
})
