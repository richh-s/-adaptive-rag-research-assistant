# Deliberately shown the questions and nothing retrieved. Feeding the failed attempt's
# documents back in would add a fourth untrusted surface to the pipeline for the sake of
# rewriting a query, and the documents are by definition the ones that did not answer it.
REFINE_PROMPT = """A search over a document collection returned poorly-matching results for \
these queries. Rewrite them so a second attempt has a better chance.

Original question: {question}

Queries that retrieved poorly:
{sub_queries}

The collection is searched two ways at once: by meaning (embeddings) and by exact keyword \
(BM25). A rewrite helps when it gives both something new to match on. So:

- Use the vocabulary a document would use, not the vocabulary a person asking would. Prefer \
the formal or industry term over the casual one ("revenue" over "money they made", \
"headcount" over "how many people work there").
- Spell out anything abbreviated, and abbreviate anything spelled out -- whichever the \
original did not do. Keyword search matches one or the other, never both.
- Name the thing being asked about explicitly in every query. A query that relies on the \
previous one for its subject retrieves nothing on its own.
- Split a query that is really two questions; merge two that are really one.

Do not broaden a query into a different question, and do not add constraints the original \
did not have -- a rewrite that retrieves well for something nobody asked is worse than the \
attempt it replaced. Return between one and five queries.
"""
