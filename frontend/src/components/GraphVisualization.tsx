import type { NodeVisit } from '../hooks/useResearchStream'
import './GraphVisualization.css'

interface GraphVisualizationProps {
  visits: NodeVisit[]
  loading: boolean
}

type StageStatus = 'pending' | 'active' | 'done' | 'skipped'

interface Stage {
  key: string
  label: string
  group?: string[]
}

// Mirrors the node wiring in graph/build_graph.py. "retrieve" collapses the three
// Send-fanned nodes into one visual step since they run concurrently for each sub-query.
const STAGES: Stage[] = [
  { key: 'route_query', label: 'Route question' },
  { key: 'decompose_query', label: 'Decompose into sub-queries' },
  { key: 'retrieve', label: 'Retrieve', group: ['retrieve_vector', 'retrieve_bm25', 'web_search'] },
  { key: 'fuse_results', label: 'Fuse results' },
  { key: 'grade_and_score', label: 'Grade & score confidence' },
  { key: 'corrective_web_search', label: 'Corrective web search' },
  { key: 'synthesize_answer', label: 'Synthesize answer' },
  { key: 'format_report', label: 'Format report' },
]

const RETRIEVAL_LABELS: Record<string, string> = {
  retrieve_vector: 'Vector',
  retrieve_bm25: 'BM25',
  web_search: 'Web',
}

function countsByNode(visits: NodeVisit[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const visit of visits) {
    counts[visit.node] = (counts[visit.node] ?? 0) + 1
  }
  return counts
}

export function GraphVisualization({ visits, loading }: GraphVisualizationProps) {
  if (visits.length === 0) return null

  const counts = countsByNode(visits)
  const visitedNodes = new Set(visits.map((v) => v.node))
  const latestNode = visits[visits.length - 1]?.node

  // grade_and_score decides whether corrective_web_search fires -- once grading has
  // happened and synthesis has started without ever visiting it, the loop was skipped.
  const gradeVisited = visitedNodes.has('grade_and_score')
  const synthesizeVisited = visitedNodes.has('synthesize_answer')

  function statusFor(stage: Stage): StageStatus {
    if (stage.key === 'corrective_web_search') {
      if (visitedNodes.has('corrective_web_search')) {
        return latestNode === 'corrective_web_search' && loading ? 'active' : 'done'
      }
      return gradeVisited && synthesizeVisited ? 'skipped' : 'pending'
    }

    const nodeKeys = stage.group ?? [stage.key]
    if (!nodeKeys.some((k) => visitedNodes.has(k))) return 'pending'
    const isLatest = nodeKeys.includes(latestNode ?? '')
    return isLatest && loading ? 'active' : 'done'
  }

  return (
    <div className="graph-viz">
      <ol className="graph-viz-list">
        {STAGES.map((stage) => {
          const status = statusFor(stage)
          return (
            <li key={stage.key} className={`graph-viz-stage graph-viz-${status}`}>
              <span className="graph-viz-dot" aria-hidden="true" />
              <div className="graph-viz-body">
                <span className="graph-viz-label">{stage.label}</span>
                {stage.group && (
                  <div className="graph-viz-badges">
                    {stage.group.map((node) =>
                      counts[node] ? (
                        <span key={node} className="graph-viz-badge">
                          {RETRIEVAL_LABELS[node]} ×{counts[node]}
                        </span>
                      ) : null,
                    )}
                  </div>
                )}
                {status === 'skipped' && <span className="graph-viz-skipped-note">not needed</span>}
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
