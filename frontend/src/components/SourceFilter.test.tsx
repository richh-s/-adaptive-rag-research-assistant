import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { IndexedSource } from '../api/client'

const listSourcesMock = vi.fn()

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return { ...actual, listSources: () => listSourcesMock() }
})

const { SourceFilter } = await import('./SourceFilter')

const sources: IndexedSource[] = [
  { source: 'anthropic.md', display_name: 'anthropic.md', chunk_count: 4, owner: 'public' },
  { source: '_t/alice/plan.md', display_name: 'plan.md', chunk_count: 2, owner: 'alice' },
]

describe('SourceFilter', () => {
  beforeEach(() => {
    listSourcesMock.mockReset()
    listSourcesMock.mockResolvedValue(sources)
  })

  it('is collapsed until opened', async () => {
    render(<SourceFilter selected={[]} onChange={vi.fn()} />)

    await screen.findByRole('button', { name: /all sources/i })
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument()
  })

  it('lists the indexed sources once opened', async () => {
    render(<SourceFilter selected={[]} onChange={vi.fn()} />)

    await userEvent.click(await screen.findByRole('button', { name: /all sources/i }))

    expect(screen.getByLabelText(/anthropic\.md/)).toBeInTheDocument()
    expect(screen.getByLabelText(/plan\.md/)).toBeInTheDocument()
  })

  it("shows a tenant's file by name, not by its internal path", async () => {
    render(<SourceFilter selected={[]} onChange={vi.fn()} />)

    await userEvent.click(await screen.findByRole('button', { name: /all sources/i }))

    expect(screen.queryByText('_t/alice/plan.md')).not.toBeInTheDocument()
    expect(screen.getByText('plan.md')).toBeInTheDocument()
  })

  it('selects a source by its real identifier, not its display name', async () => {
    // The API matches `filters.sources` on the path, so sending the display name would
    // silently match nothing and return an empty local corpus.
    const onChange = vi.fn()
    render(<SourceFilter selected={[]} onChange={onChange} />)
    await userEvent.click(await screen.findByRole('button', { name: /all sources/i }))

    await userEvent.click(screen.getByLabelText(/plan\.md/))

    expect(onChange).toHaveBeenCalledWith(['_t/alice/plan.md'])
  })

  it('deselects an already-selected source', async () => {
    const onChange = vi.fn()
    render(<SourceFilter selected={['anthropic.md']} onChange={onChange} />)
    await userEvent.click(await screen.findByRole('button', { name: /1 of 2 sources/i }))

    await userEvent.click(screen.getByLabelText(/anthropic\.md/))

    expect(onChange).toHaveBeenCalledWith([])
  })

  it('summarises the selection in the toggle', async () => {
    render(<SourceFilter selected={['anthropic.md']} onChange={vi.fn()} />)

    expect(await screen.findByRole('button', { name: /1 of 2 sources/i })).toBeInTheDocument()
  })

  it('offers a clear control only when something is selected', async () => {
    const onChange = vi.fn()
    const { rerender } = render(<SourceFilter selected={[]} onChange={onChange} />)
    await userEvent.click(await screen.findByRole('button', { name: /all sources/i }))
    expect(screen.queryByRole('button', { name: /clear/i })).not.toBeInTheDocument()

    rerender(<SourceFilter selected={['anthropic.md']} onChange={onChange} />)
    await userEvent.click(screen.getByRole('button', { name: /clear/i }))

    expect(onChange).toHaveBeenCalledWith([])
  })

  it('says the filter does not apply to web search', async () => {
    render(<SourceFilter selected={[]} onChange={vi.fn()} />)

    await userEvent.click(await screen.findByRole('button', { name: /all sources/i }))

    expect(screen.getByText(/web search is unaffected/i)).toBeInTheDocument()
  })

  it('renders nothing when the corpus is empty', async () => {
    listSourcesMock.mockResolvedValue([])
    const { container } = render(<SourceFilter selected={[]} onChange={vi.fn()} />)

    await waitFor(() => expect(listSourcesMock).toHaveBeenCalled())

    expect(container).toBeEmptyDOMElement()
  })

  it('hides itself rather than erroring when sources cannot be loaded', async () => {
    // Not being able to narrow a search isn't worth interrupting anyone over; the
    // unfiltered search still works.
    listSourcesMock.mockRejectedValue(new Error('offline'))
    const { container } = render(<SourceFilter selected={[]} onChange={vi.fn()} />)

    await waitFor(() => expect(listSourcesMock).toHaveBeenCalled())

    expect(container).toBeEmptyDOMElement()
  })
})
