SYNTHESIS_PROMPT = """Answer the question using ONLY the numbered context below. Cite sources \
inline using their marker, e.g. [1], right after the claim it supports.

Write in plain, direct language for a non-technical reader, as if you simply know these things \
-- never refer to "the context", "the provided documents", or similar meta-commentary about your \
own sources. If some part of the question isn't covered, say plainly what you don't know (e.g. \
"I don't have information on X") instead of describing what the documents do or don't contain.

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
