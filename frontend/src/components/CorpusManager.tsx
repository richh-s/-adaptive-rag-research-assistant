import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { getIngestStatus, ingestFile, ResearchApiError, type IngestStage } from '../api/client'
import './CorpusManager.css'

interface CorpusManagerProps {
  open: boolean
  onClose: () => void
}

type UploadStage = 'uploading' | IngestStage

interface UploadEntry {
  id: string
  filename: string
  stage: UploadStage
  message?: string
}

const ACCEPTED_EXTENSIONS = ['.pdf', '.md', '.txt']

const IN_PROGRESS_STAGES: ReadonlySet<UploadStage> = new Set(['uploading', 'queued', 'parsing', 'indexing'])

const STAGE_LABELS: Record<UploadStage, string> = {
  uploading: 'Uploading...',
  queued: 'Queued for indexing',
  parsing: 'Parsing...',
  indexing: 'Indexing...',
  indexed: 'Indexed',
  failed: 'Failed',
}

// Recursive setTimeout, not setInterval: each poll waits for the previous request to finish
// before scheduling the next one, so a slow response can't cause overlapping requests to pile up.
const POLL_INTERVAL_MS = 1500
// Give up after this long without reaching a terminal stage, so a stuck/orphaned task doesn't
// leave the badge spinning forever.
const POLL_TIMEOUT_MS = 5 * 60 * 1000

function isAcceptedFile(filename: string): boolean {
  const lower = filename.toLowerCase()
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext))
}

function makeUploadId(filename: string): string {
  return `${filename}-${Math.random().toString(36).slice(2)}`
}

export function CorpusManager({ open, onClose }: CorpusManagerProps) {
  const [dragActive, setDragActive] = useState(false)
  const [uploads, setUploads] = useState<UploadEntry[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const mountedRef = useRef(true)
  const pollTimeoutsRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  useEffect(() => {
    mountedRef.current = true
    const timeouts = pollTimeoutsRef.current
    return () => {
      mountedRef.current = false
      timeouts.forEach((timeoutId) => clearTimeout(timeoutId))
      timeouts.clear()
    }
  }, [])

  const updateUpload = useCallback((id: string, patch: Partial<UploadEntry>) => {
    setUploads((prev) => prev.map((u) => (u.id === id ? { ...u, ...patch } : u)))
  }, [])

  const pollTask = useCallback(
    (id: string, taskId: string, startedAt: number) => {
      const poll = async () => {
        if (!mountedRef.current) return

        try {
          const status = await getIngestStatus(taskId)
          if (!mountedRef.current) return

          updateUpload(id, {
            stage: status.stage,
            message: status.stage === 'failed' ? (status.error ?? status.message) : status.message,
          })

          if (status.stage === 'indexed' || status.stage === 'failed') {
            pollTimeoutsRef.current.delete(id)
            return
          }
        } catch {
          // Transient network/server hiccup while polling -- keep retrying rather than failing
          // the entry on a single bad request; the overall POLL_TIMEOUT_MS still bounds this.
        }

        if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
          if (mountedRef.current) {
            updateUpload(id, { stage: 'failed', message: 'Timed out waiting for indexing to finish.' })
          }
          pollTimeoutsRef.current.delete(id)
          return
        }

        const timeoutId = setTimeout(poll, POLL_INTERVAL_MS)
        pollTimeoutsRef.current.set(id, timeoutId)
      }

      void poll()
    },
    [updateUpload],
  )

  const uploadFile = useCallback(
    async (file: File) => {
      const id = makeUploadId(file.name)

      if (!isAcceptedFile(file.name)) {
        setUploads((prev) => [
          { id, filename: file.name, stage: 'failed', message: 'Unsupported file type.' },
          ...prev,
        ])
        return
      }

      setUploads((prev) => [{ id, filename: file.name, stage: 'uploading' }, ...prev])

      try {
        const result = await ingestFile(file)
        if (!mountedRef.current) return
        updateUpload(id, { stage: 'queued', message: result.message })
        pollTask(id, result.task_id, Date.now())
      } catch (err) {
        if (!mountedRef.current) return
        const message = err instanceof ResearchApiError ? err.message : 'Upload failed.'
        updateUpload(id, { stage: 'failed', message })
      }
    },
    [pollTask, updateUpload],
  )

  function handleFiles(files: FileList | null) {
    if (!files) return
    Array.from(files).forEach((file) => void uploadFile(file))
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragActive(false)
    handleFiles(e.dataTransfer.files)
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragActive(true)
  }

  function handleDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragActive(false)
  }

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    handleFiles(e.target.files)
    e.target.value = ''
  }

  if (!open) return null

  return (
    <div className="corpus-overlay" onClick={onClose}>
      <aside
        className="corpus-drawer"
        onClick={(e) => e.stopPropagation()}
        aria-label="Corpus manager"
      >
        <div className="corpus-drawer-header">
          <h2>Manage corpus</h2>
          <button type="button" className="corpus-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <p className="corpus-subtitle">
          Upload PDF, Markdown, or text files to add them to the local knowledge base. Each file
          is saved immediately and indexed in the background.
        </p>

        <div
          className={`dropzone ${dragActive ? 'dropzone-active' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
          }}
          role="button"
          tabIndex={0}
        >
          <span className="dropzone-title">Drop files here or click to browse</span>
          <span className="dropzone-hint">PDF, Markdown (.md), or text (.txt)</span>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".pdf,.md,.txt"
            onChange={handleInputChange}
            hidden
          />
        </div>

        {uploads.length > 0 && (
          <ul className="upload-list">
            {uploads.map((upload) => {
              const inProgress = IN_PROGRESS_STAGES.has(upload.stage)
              return (
                <li key={upload.id} className={`upload-item upload-${upload.stage}`}>
                  <span className="upload-filename">{upload.filename}</span>
                  {inProgress && <span className="spinner" role="status" aria-label="Indexing" />}
                  {inProgress && (
                    <span className="upload-badge upload-badge-progress">
                      {upload.message ?? STAGE_LABELS[upload.stage]}
                    </span>
                  )}
                  {upload.stage === 'indexed' && (
                    <span className="upload-badge upload-badge-success">
                      {upload.message ?? STAGE_LABELS.indexed}
                    </span>
                  )}
                  {upload.stage === 'failed' && (
                    <span className="upload-badge upload-badge-error">
                      {upload.message ?? STAGE_LABELS.failed}
                    </span>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </aside>
    </div>
  )
}
