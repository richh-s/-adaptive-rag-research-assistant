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
