import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { IngestResponse } from '../api/client'

const ingestFileMock = vi.fn()

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    ingestFile: (...args: Parameters<typeof actual.ingestFile>) => ingestFileMock(...args),
  }
})

import { CorpusManager } from './CorpusManager'

beforeEach(() => {
  ingestFileMock.mockReset()
})

function makeFile(name: string, contents = 'hello world') {
  return new File([contents], name, { type: 'text/plain' })
}

describe('CorpusManager', () => {
  it('renders nothing when closed', () => {
    render(<CorpusManager open={false} onClose={() => {}} />)

    expect(screen.queryByText(/manage corpus/i)).not.toBeInTheDocument()
  })

  it('renders the drawer when open', () => {
    render(<CorpusManager open={true} onClose={() => {}} />)

    expect(screen.getByText(/manage corpus/i)).toBeInTheDocument()
    expect(screen.getByText(/drop files here/i)).toBeInTheDocument()
  })

  it('calls onClose when the close button is clicked', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<CorpusManager open={true} onClose={onClose} />)

    await user.click(screen.getByRole('button', { name: /close/i }))

    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('uploads a selected file and shows a success badge once ingestFile resolves', async () => {
    const user = userEvent.setup()
    const response: IngestResponse = {
      filename: 'notes_ab12cd34.md',
      original_filename: 'notes.md',
      size_bytes: 11,
      status: 'queued',
      message: 'File saved; indexing has started in the background.',
    }
    ingestFileMock.mockResolvedValue(response)

    const { container } = render(<CorpusManager open={true} onClose={() => {}} />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = makeFile('notes.md')

    await user.upload(input, file)

    expect(ingestFileMock).toHaveBeenCalledWith(file)
    expect(await screen.findByText('notes.md')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText(/queued for indexing/i)).toBeInTheDocument())
  })

  it('shows an error badge when ingestFile rejects', async () => {
    const user = userEvent.setup()
    const { ResearchApiError } = await import('../api/client')
    ingestFileMock.mockRejectedValue(new ResearchApiError('Unsupported file type'))

    const { container } = render(<CorpusManager open={true} onClose={() => {}} />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement

    await user.upload(input, makeFile('notes.md'))

    expect(await screen.findByText('Unsupported file type')).toBeInTheDocument()
  })

  it('rejects an unsupported file client-side without calling ingestFile', async () => {
    // The dropzone's <input accept=".pdf,.md,.txt"> makes userEvent.upload filter out
    // non-matching files before they ever reach the component -- disable that filtering so
    // this test can exercise the component's own isAcceptedFile() rejection path instead of
    // just re-testing the browser's native accept-attribute behavior.
    const user = userEvent.setup({ applyAccept: false })
    const { container } = render(<CorpusManager open={true} onClose={() => {}} />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement

    await user.upload(input, makeFile('notes.json'))

    expect(await screen.findByText(/unsupported file type/i)).toBeInTheDocument()
    expect(ingestFileMock).not.toHaveBeenCalled()
  })
})
