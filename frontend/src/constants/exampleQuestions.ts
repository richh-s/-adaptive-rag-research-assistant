export interface ExampleQuestion {
  label: string
  question: string
}

export const EXAMPLE_QUESTIONS: ExampleQuestion[] = [
  { label: 'Vector routing', question: 'Who founded Anthropic and what is their safety research called?' },
  { label: 'Web routing', question: 'What is the most recent Claude model release?' },
  { label: 'No retrieval', question: 'What is the capital of France?' },
  { label: 'Decomposition', question: "Compare Anthropic and Mistral AI's founding stories and safety focus." },
  { label: 'Corrective fallback', question: 'What safety research did Anthropic publish this week?' },
]
