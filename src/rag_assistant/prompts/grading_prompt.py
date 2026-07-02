GRADING_PROMPT = """Grade how relevant each numbered document below is to answering the question. \
For each one, decide whether it is relevant (meaningfully helps answer the question) and give a \
relevance score from 0.0 to 1.0. Return exactly one grade per document, in the same order they \
are given.

Question: {question}

Documents:
{documents}
"""
