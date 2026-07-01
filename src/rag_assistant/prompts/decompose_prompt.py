DECOMPOSE_PROMPT = """Break the question below into 2-5 focused, self-contained sub-questions \
that together cover everything needed to answer it fully. Each sub-question must stand on its \
own -- no pronouns or references back to the original question.

If the question is already simple and only asks about one thing, return a single-element list \
containing the original question, unchanged.

Question: {question}
"""
