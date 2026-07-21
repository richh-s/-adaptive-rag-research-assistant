SYNTHESIS_PROMPT = """Answer the question using ONLY the numbered context below. Cite sources \
inline using their marker, e.g. [1], right after the claim it supports. If the context does not \
fully answer the question, say what's missing rather than guessing.

Question: {question}

Context:
{context}
"""

NO_CONTEXT_PROMPT = """Answer the question directly using your own general knowledge. This \
question did not require any document or web retrieval.

Question: {question}
"""

EMPTY_RETRIEVAL_PROMPT = """Local document search and web search were both attempted for this \
question but returned no usable results. Start your answer by clearly stating that no relevant \
sources were found. Only then, if you are genuinely confident, you may add an answer from your \
own general knowledge -- explicitly flagged as unverified/not grounded in retrieved sources. If \
you are not confident, say so instead of guessing.

Question: {question}
"""
