DECOMPOSE_PROMPT = """Break the question below into 2-5 focused, self-contained sub-questions \
that together cover everything needed to answer it fully. Each sub-question must stand on its \
own -- no pronouns or references back to the original question.

If the question is already simple and only asks about one thing, return a single-element list \
containing the original question, unchanged.

Examples:

Question: Who founded Anthropic, and who founded Mistral AI?
Sub-questions:
- Who founded Anthropic?
- Who founded Mistral AI?

Question: Compare Mistral AI's and Meta AI's stances on releasing open-weight models.
Sub-questions:
- What is Mistral AI's stance on releasing open-weight models?
- What is Meta AI's stance on releasing open-weight models?

Question: What is a transformer model in machine learning?
Sub-questions:
- What is a transformer model in machine learning?

Question: What is 15 percent of 240?
Sub-questions:
- What is 15 percent of 240?

The last two matter as much as the first two: a simple question split into pieces costs a \
retrieval pass per piece and fuses near-identical results, so "leave it alone" is the correct \
output far more often than the instruction above makes it look.

Question: {question}
"""
