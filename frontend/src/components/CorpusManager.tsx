import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { ingestFile, ResearchApiError } from '../api/client'
import './CorpusManager.css'

interface CorpusManagerProps {
  open: boolean
  onClose: () => void
}

type UploadStatus = 'uploading' | 'success' | 'error'

interface UploadEntry {
  id: string
  filename: string
  status: UploadStatus
  message?: string
}

const ACCEPTED_EXTENSIONS = ['.pdf', '.md', '.txt']

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

  const uploadFile = useCallback(async (file: File) => {
    const id = makeUploadId(file.name)

    if (!isAcceptedFile(file.name)) {
      setUploads((prev) => [
        { id, filename: file.name, status: 'error', message: 'Unsupported file type.' },
        ...prev,
      ])
      return
    }

    setUploads((prev) => [{ id, filename: file.name, status: 'uploading' }, ...prev])

    try {
      const result = await ingestFile(file)
      setUploads((prev) =>
        prev.map((u) => (u.id === id ? { ...u, status: 'success', message: result.message } : u)),
      )
    } catch (err) {
      const message = err instanceof ResearchApiError ? err.message : 'Upload failed.'
      setUploads((prev) => prev.map((u) => (u.id === id ? { ...u, status: 'error', message } : u)))
    }
  }, [])

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
            {uploads.map((upload) => (
              <li key={upload.id} className={`upload-item upload-${upload.status}`}>
                <span className="upload-filename">{upload.filename}</span>
                {upload.status === 'uploading' && (
                  <span className="spinner" role="status" aria-label="Uploading" />
                )}
                {upload.status === 'success' && (
                  <span className="upload-badge upload-badge-success">Queued for indexing</span>
                )}
                {upload.status === 'error' && (
                  <span className="upload-badge upload-badge-error">
                    {upload.message ?? 'Upload failed.'}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </aside>
    </div>
  )
}
