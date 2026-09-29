import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { GraphVisualization } from './GraphVisualization'
import type { NodeVisit } from '../hooks/useResearchStream'

function visit(node: string, seq: number): NodeVisit {
  return { node, message: node, seq }
}

describe('GraphVisualization', () => {
  it('renders nothing before any progress event has arrived', () => {
    const { container } = render(<GraphVisualization visits={[]} loading={false} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('marks the most recent node active while still loading', () => {
    render(
      <GraphVisualization
        visits={[visit('route_query', 1), visit('decompose_query', 2)]}
        loading={true}
      />,
    )

    const decompose = screen.getByText('Decompose into sub-queries').closest('li')
    const route = screen.getByText('Route question').closest('li')

    expect(decompose).toHaveClass('graph-viz-active')
    expect(route).toHaveClass('graph-viz-done')
  })

  it('shows the retrieve stage as done once loading finishes, with per-source counts', () => {
    render(
      <GraphVisualization
        visits={[
          visit('route_query', 1),
          visit('decompose_query', 2),
          visit('retrieve_vector', 3),
          visit('retrieve_bm25', 4),
          visit('retrieve_vector', 5),
          visit('web_search', 6),
        ]}
        loading={false}
      />,
    )

    const retrieveStage = screen.getByText('Retrieve').closest('li')
    expect(retrieveStage).toHaveClass('graph-viz-done')
    expect(screen.getByText('Vector ×2')).toBeInTheDocument()
    expect(screen.getByText('BM25 ×1')).toBeInTheDocument()
    expect(screen.getByText('Web ×1')).toBeInTheDocument()
  })

  it('marks corrective_web_search as skipped once grading and synthesis ran without it', () => {
    render(
      <GraphVisualization
        visits={[
          visit('route_query', 1),
          visit('decompose_query', 2),
          visit('retrieve_vector', 3),
          visit('fuse_results', 4),
          visit('grade_and_score', 5),
          visit('synthesize_answer', 6),
        ]}
        loading={false}
      />,
    )

    const corrective = screen.getByText('Corrective web search').closest('li')
    expect(corrective).toHaveClass('graph-viz-skipped')
    // Scoped to this stage: both escalations are conditional, so a bare query for the note
    // would match whichever one the DOM happened to render first.
    expect(within(corrective as HTMLElement).getByText('not needed')).toBeInTheDocument()
  })

  it('marks refine_retrieval as skipped when confidence never dropped', () => {
    render(
      <GraphVisualization
        visits={[
          visit('route_query', 1),
          visit('decompose_query', 2),
          visit('retrieve_vector', 3),
          visit('fuse_results', 4),
          visit('grade_and_score', 5),
          visit('synthesize_answer', 6),
        ]}
        loading={false}
      />,
    )

    const refine = screen.getByText('Re-ask the corpus').closest('li')
    expect(refine).toHaveClass('graph-viz-skipped')
  })

  it('shows the corpus re-ask as done and the web fallback as skipped when only the first fired', () => {
    render(
      <GraphVisualization
        visits={[
          visit('route_query', 1),
          visit('decompose_query', 2),
          visit('retrieve_vector', 3),
          visit('fuse_results', 4),
          visit('grade_and_score', 5),
          visit('refine_retrieval', 6),
          visit('fuse_results', 7),
          visit('grade_and_score', 8),
          visit('synthesize_answer', 9),
        ]}
        loading={false}
      />,
    )

    expect(screen.getByText('Re-ask the corpus').closest('li')).toHaveClass('graph-viz-done')
    expect(screen.getByText('Corrective web search').closest('li')).toHaveClass(
      'graph-viz-skipped',
    )
  })

  it('shows the grounding check once synthesis has run', () => {
    render(
      <GraphVisualization
        visits={[
          visit('synthesize_answer', 1),
          visit('verify_groundedness', 2),
          visit('format_report', 3),
        ]}
        loading={false}
      />,
    )

    expect(screen.getByText('Check answer against sources').closest('li')).toHaveClass(
      'graph-viz-done',
    )
  })

  it('marks corrective_web_search as done when the correction loop actually fired', () => {
    render(
      <GraphVisualization
        visits={[
          visit('route_query', 1),
          visit('decompose_query', 2),
          visit('retrieve_vector', 3),
          visit('fuse_results', 4),
          visit('grade_and_score', 5),
          visit('corrective_web_search', 6),
          visit('fuse_results', 7),
          visit('synthesize_answer', 8),
        ]}
        loading={false}
      />,
    )

    const corrective = screen.getByText('Corrective web search').closest('li')
    expect(corrective).toHaveClass('graph-viz-done')
    expect(within(corrective as HTMLElement).queryByText('not needed')).not.toBeInTheDocument()
  })
})
