import type { ResearchSummary } from '../api/client'
import './ResearchSummaryPanel.css'

interface ResearchSummaryPanelProps {
  summary: ResearchSummary
}

// Several graph nodes fan out per sub-query (retrieve_vector/retrieve_bm25/web_search) or can
// run twice on the corrective-retry loop (fuse_results) -- group their latencies under one
// human-readable stage so the table reads like a single pipeline instead of a raw event log.
const STAGE_LABELS: Record<string, string> = {
  condense_question: 'Follow-up Resolution',
  route_query: 'Route',
  decompose_query: 'Decompose',
  retrieve_vector: 'Retrieval',
  retrieve_bm25: 'Retrieval',
  web_search: 'Retrieval',
  fuse_results: 'Fusion',
  grade_and_score: 'Grading',
  refine_retrieval: 'Corpus Re-ask',
  corrective_web_search: 'Corrective Search',
  synthesize_answer: 'Synthesis',
  verify_groundedness: 'Grounding Check',
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

      {summary.condensed_question && (
        <div className="summary-row">
          <span className="summary-label">Interpreted As</span>
          <span className="summary-value">{summary.condensed_question}</span>
        </div>
      )}

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
        <span className="summary-value">
          {summary.fused_document_count} unique documents
          {/* Shown only when the budget actually dropped something, so a truncated answer
              reads as a deliberate cost decision rather than as retrieval finding less. */}
          {summary.context_documents_dropped
            ? ` (${summary.context_documents_dropped} trimmed to fit the context budget)`
            : ''}
        </span>
      </div>

      <div className="summary-row">
        <span className="summary-label">Confidence</span>
        <span className="summary-value">
          {summary.confidence_score !== null ? summary.confidence_score.toFixed(2) : 'n/a'}
        </span>
      </div>

      <div className="summary-row">
        <span className="summary-label">Retry</span>
        <span className="summary-value">
          {summary.correction_attempted
            ? summary.refinement_attempted
              ? 'Corpus re-asked, then web search'
              : 'Corrective web search'
            : summary.refinement_attempted
              ? 'Corpus re-asked with rewritten queries'
              : 'No'}
        </span>
      </div>

      {/* Reported separately from Confidence above, because they answer different questions:
          that one grades retrieval, this one grades the answer written from it. "Not checked"
          is shown as itself rather than as a clean result -- an unverified answer and a
          verified one with nothing wrong both have zero unsupported claims. */}
      <div className="summary-row">
        <span className="summary-label">Answer grounded</span>
        <span className="summary-value">
          {!summary.groundedness_checked
            ? 'Not checked'
            : summary.groundedness_score === null || summary.groundedness_score === undefined
              ? 'No claims to check'
              : `${(summary.groundedness_score * 100).toFixed(0)}% of claims supported` +
                (summary.unsupported_claim_count
                  ? ` (${summary.unsupported_claim_count} unsupported)`
                  : '')}
        </span>
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
