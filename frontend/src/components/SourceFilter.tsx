import { useEffect, useState } from 'react'
import { listSources, type IndexedSource } from '../api/client'
import './SourceFilter.css'

interface SourceFilterProps {
  selected: string[]
  onChange: (sources: string[]) => void
}

/**
 * Restricts local retrieval to chosen files.
 *
 * Collapsed by default and hidden entirely when there is nothing to filter. Filtering is a
 * power-user affordance — the default path is "ask a question", and a permanently expanded
 * list of checkboxes above the input would make the common case look more complicated than
 * it is.
 *
 * Failing to load the source list hides the control rather than showing an error: not being
 * able to *narrow* a search is not a problem worth interrupting someone over, and the
 * unfiltered search still works.
 */
export function SourceFilter({ selected, onChange }: SourceFilterProps) {
  const [sources, setSources] = useState<IndexedSource[]>([])
  const [open, setOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    listSources()
      .then((loaded) => {
        if (!cancelled) setSources(loaded)
      })
      .catch(() => {
        if (!cancelled) setSources([])
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (sources.length === 0) return null

  function toggle(source: string) {
    onChange(
      selected.includes(source)
        ? selected.filter((s) => s !== source)
        : [...selected, source],
    )
  }

  const label =
    selected.length === 0 ? 'All sources' : `${selected.length} of ${sources.length} sources`

  return (
    <div className="source-filter">
      <button
        type="button"
        className="source-filter-toggle"
        aria-expanded={open}
        aria-controls="source-filter-list"
        onClick={() => setOpen((current) => !current)}
      >
        <span aria-hidden="true">{open ? '▾' : '▸'}</span> {label}
      </button>

      {open && (
        <div className="source-filter-list" id="source-filter-list">
          <fieldset>
            <legend className="visually-hidden">Limit retrieval to these sources</legend>
            {sources.map((source) => (
              <label key={source.source} className="source-filter-item">
                <input
                  type="checkbox"
                  checked={selected.includes(source.source)}
                  onChange={() => toggle(source.source)}
                />
                <span className="source-filter-name">{source.display_name}</span>
                <span className="source-filter-count">{source.chunk_count}</span>
              </label>
            ))}
          </fieldset>
          {selected.length > 0 && (
            <button type="button" className="source-filter-clear" onClick={() => onChange([])}>
              Clear
            </button>
          )}
          <p className="source-filter-hint">
            Applies to the local knowledge base only — web search is unaffected.
          </p>
        </div>
      )}
    </div>
  )
}
