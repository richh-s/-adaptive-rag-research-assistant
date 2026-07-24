import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { IngestResponse, IngestTaskStatus } from '../api/client'

const ingestFileMock = vi.fn()
const getIngestStatusMock = vi.fn()

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    ingestFile: (...args: Parameters<typeof actual.ingestFile>) => ingestFileMock(...args),
    getIngestStatus: (...args: Parameters<typeof actual.getIngestStatus>) => getIngestStatusMock(...args),
  }
})

import { CorpusManager } from './CorpusManager'

// The component polls every 1.5s (POLL_INTERVAL_MS in CorpusManager.tsx); these tests use real
// timers and size waitFor's timeout comfortably above that so each poll tick is caught.
const POLL_WAIT = { timeout: 3000 }

beforeEach(() => {
  ingestFileMock.mockReset()
  getIngestStatusMock.mockReset()
})

function makeFile(name: string, contents = 'hello world') {
  return new File([contents], name, { type: 'text/plain' })
}

function ingestResponse(overrides: Partial<IngestResponse> = {}): IngestResponse {
  return {
    task_id: 'task-1',
    filename: 'notes_ab12cd34.md',
    original_filename: 'notes.md',
    size_bytes: 11,
    status: 'queued',
    message: 'File saved; indexing has started in the background.',
    ...overrides,
  }
}

function taskStatus(overrides: Partial<IngestTaskStatus> = {}): IngestTaskStatus {
  return {
    task_id: 'task-1',
    filename: 'notes_ab12cd34.md',
    original_filename: 'notes.md',
    stage: 'parsing',
    message: 'Loading and parsing corpus files...',
    ...overrides,
  }
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

  it('polls task status after upload and updates the badge through to indexed', async () => {
    const user = userEvent.setup()
    ingestFileMock.mockResolvedValue(ingestResponse())
    getIngestStatusMock
      .mockResolvedValueOnce(taskStatus({ stage: 'parsing', message: 'Loading and parsing corpus files...' }))
      .mockResolvedValueOnce(taskStatus({ stage: 'indexing', message: 'Embedding and indexing changed files...' }))
      .mockResolvedValueOnce(
        taskStatus({ stage: 'indexed', message: 'Indexed 3 chunk(s) from 1 file(s); local search is up to date.' }),
      )

    const { container } = render(<CorpusManager open={true} onClose={() => {}} />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = makeFile('notes.md')

    await user.upload(input, file)

    expect(ingestFileMock).toHaveBeenCalledWith(file)
    expect(await screen.findByText('notes.md')).toBeInTheDocument()
    // The first status check fires immediately (not after a delay) so results show up as soon
    // as possible, which means the initial "queued" message can be superseded by this "parsing"
    // response before it ever renders -- don't assert on it, go straight to what's guaranteed.
    expect(await screen.findByText(/Loading and parsing corpus files/i, {}, POLL_WAIT)).toBeInTheDocument()
    expect(await screen.findByText(/Embedding and indexing changed files/i, {}, POLL_WAIT)).toBeInTheDocument()
    expect(
      await screen.findByText(/Indexed 3 chunk\(s\) from 1 file\(s\)/i, {}, POLL_WAIT),
    ).toBeInTheDocument()

    expect(getIngestStatusMock).toHaveBeenCalledTimes(3)
    expect(getIngestStatusMock).toHaveBeenCalledWith('task-1')

    // Polling must stop once a terminal stage is reached -- no further requests after that.
    await new Promise((resolve) => setTimeout(resolve, 2000))
    expect(getIngestStatusMock).toHaveBeenCalledTimes(3)
    // Three real poll intervals plus the trailing no-more-polls check exceed vitest's default
    // 5s test timeout.
  }, 10000)

  it('shows a failed badge with the backend error once polling reaches the failed stage', async () => {
    const user = userEvent.setup()
    ingestFileMock.mockResolvedValue(ingestResponse())
    getIngestStatusMock.mockResolvedValueOnce(
      taskStatus({ stage: 'failed', message: 'Indexing failed.', error: 'Corrupt PDF.' }),
    )

    const { container } = render(<CorpusManager open={true} onClose={() => {}} />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement

    await user.upload(input, makeFile('notes.md'))
    expect(await screen.findByText('Corrupt PDF.', {}, POLL_WAIT)).toBeInTheDocument()

    await new Promise((resolve) => setTimeout(resolve, 2000))
    expect(getIngestStatusMock).toHaveBeenCalledTimes(1)
  })

  it('shows an error badge when ingestFile itself rejects, without polling', async () => {
    const user = userEvent.setup()
    const { ResearchApiError } = await import('../api/client')
    ingestFileMock.mockRejectedValue(new ResearchApiError('Unsupported file type'))

    const { container } = render(<CorpusManager open={true} onClose={() => {}} />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement

    await user.upload(input, makeFile('notes.md'))

    expect(await screen.findByText('Unsupported file type')).toBeInTheDocument()
    expect(getIngestStatusMock).not.toHaveBeenCalled()
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
