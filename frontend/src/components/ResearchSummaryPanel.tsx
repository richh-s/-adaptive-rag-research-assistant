import type { ResearchSummary } from '../api/client'
import './ResearchSummaryPanel.css'

interface ResearchSummaryPanelProps {
  summary: ResearchSummary
}

// Several graph nodes fan out per sub-query (retrieve_vector/retrieve_bm25/web_search) or can
// run twice on the corrective-retry loop (fuse_results) -- group their latencies under one
// human-readable stage so the table reads like a single pipeline instead of a raw event log.
const STAGE_LABELS: Record<string, string> = {
  route_query: 'Route',
  decompose_query: 'Decompose',
  retrieve_vector: 'Retrieval',
  retrieve_bm25: 'Retrieval',
  web_search: 'Retrieval',
  fuse_results: 'Fusion',
  grade_and_score: 'Grading',
  corrective_web_search: 'Corrective Search',
  synthesize_answer: 'Synthesis',
  format_report: 'Formatting',
}

function groupLatencies(summary: ResearchSummary): { label: string; latency_ms: number }[] {
  const order: string[] = []
  const totals = new Map<string, number>()
  for (const { node, latency_ms } of summary.node_latencies_ms) {
    const label = STAGE_LABELS[node] ?? node
    if (!totals.has(label)) {
      order.push(label)
      totals.set(label, 0)
    }
    totals.set(label, (totals.get(label) ?? 0) + latency_ms)
  }
  return order.map((label) => ({ label, latency_ms: totals.get(label) ?? 0 }))
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`
}

export function ResearchSummaryPanel({ summary }: ResearchSummaryPanelProps) {
  const stages = groupLatencies(summary)

  return (
    <div className="summary-panel">
      <h3>Research Summary</h3>

      <div className="summary-row">
        <span className="summary-label">Route</span>
        <span className="summary-value">{summary.route ?? 'unknown'}</span>
      </div>

      {summary.sub_queries.length > 0 && (
        <div className="summary-block">
          <span className="summary-label">Subqueries</span>
          <ul className="summary-checklist">
            {summary.sub_queries.map((sq) => (
              <li key={sq}>
                <span className="check">✓</span> {sq}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="summary-block">
        <span className="summary-label">Retrieved</span>
        <div className="summary-stats">
          <span>Vector: {summary.retrieval_counts.vector} docs</span>
          <span>BM25: {summary.retrieval_counts.bm25} docs</span>
          <span>Web: {summary.retrieval_counts.web} docs</span>
        </div>
      </div>

      <div className="summary-row">
        <span className="summary-label">After Fusion</span>
        <span className="summary-value">{summary.fused_document_count} unique documents</span>
      </div>

      <div className="summary-row">
        <span className="summary-label">Confidence</span>
        <span className="summary-value">
          {summary.confidence_score !== null ? summary.confidence_score.toFixed(2) : 'n/a'}
        </span>
      </div>

      <div className="summary-row">
        <span className="summary-label">Corrective Search</span>
        <span className="summary-value">{summary.correction_attempted ? 'Yes' : 'No'}</span>
      </div>

      {stages.length > 0 && (
        <div className="summary-block">
          <span className="summary-label">Latency</span>
          <table className="summary-latency">
            <tbody>
              {stages.map((stage) => (
                <tr key={stage.label}>
                  <td>{stage.label}</td>
                  <td>{formatMs(stage.latency_ms)}</td>
                </tr>
              ))}
              <tr className="summary-latency-total">
                <td>Total</td>
                <td>{formatMs(summary.total_latency_ms)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
