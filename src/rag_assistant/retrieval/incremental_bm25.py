"""An incrementally maintainable BM25 index.

`rank_bm25.BM25Okapi` computes its statistics in the constructor, so the only way to reflect a
changed corpus is to build a new one over every document. Ingestion changes a handful of files
at a time, which made every upload pay for the whole collection: fetch every chunk, re-tokenize
every chunk, recompute every statistic.

This keeps the same statistics but maintains them under `add` and `remove`, so an ingest costs
the documents that actually changed. The scoring function is BM25Okapi's exactly -- including
its epsilon floor for negative IDF -- and `test_incremental_bm25.py` asserts score-for-score
equivalence against the library on randomised corpora, because the whole point of a
hand-rolled scorer is that it must not quietly rank differently from the one it replaced.

One statistic genuinely cannot be maintained per-document: the epsilon floor is a fraction of
the *average* IDF across the vocabulary, and every IDF depends on the corpus size, so adding
one document perturbs all of them. That recomputation is O(vocabulary) and is deferred until
a query actually needs it, rather than being paid on every add.
"""

import math
from collections import Counter


class IncrementalBM25:
    """BM25Okapi's scoring with add/remove instead of a rebuild.

    Parameters match `rank_bm25.BM25Okapi` defaults so the two agree by construction.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        # doc_id -> {term: frequency}. Term frequencies rather than the token list: scoring
        # only ever needs counts, and storing counts is smaller for repetitive text.
        self._doc_freqs: dict[str, Counter[str]] = {}
        self._doc_len: dict[str, int] = {}
        # term -> number of documents containing it
        self._doc_freq: Counter[str] = Counter()
        self._total_len = 0
        self._idf: dict[str, float] = {}
        self._idf_dirty = True

    # ---- corpus maintenance ----

    def add(self, doc_id: str, tokens: list[str]) -> None:
        """Adds or replaces one document. Replacing removes the old one first, so a re-indexed
        chunk cannot double-count its own terms in the document frequencies."""
        if doc_id in self._doc_freqs:
            self.remove(doc_id)
        frequencies = Counter(tokens)
        self._doc_freqs[doc_id] = frequencies
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)
        for term in frequencies:
            self._doc_freq[term] += 1
        self._idf_dirty = True

    def remove(self, doc_id: str) -> None:
        frequencies = self._doc_freqs.pop(doc_id, None)
        if frequencies is None:
            return
        self._total_len -= self._doc_len.pop(doc_id)
        for term in frequencies:
            remaining = self._doc_freq[term] - 1
            if remaining <= 0:
                # Dropped entirely rather than left at zero: a term with df=0 would otherwise
                # accumulate in the vocabulary forever and skew the average IDF that the
                # epsilon floor is derived from.
                del self._doc_freq[term]
            else:
                self._doc_freq[term] = remaining
        self._idf_dirty = True

    def __len__(self) -> int:
        return len(self._doc_freqs)

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_freqs)

    @property
    def average_length(self) -> float:
        return self._total_len / len(self._doc_freqs) if self._doc_freqs else 0.0

    # ---- scoring ----

    def _rebuild_idf(self) -> None:
        """BM25Okapi's IDF, including its handling of negative values.

        A term appearing in more than half the corpus gets a negative IDF under this formula,
        which would let a common term *reduce* a document's score. BM25Okapi replaces those
        with a small positive floor derived from the mean IDF; reproducing that exactly is
        what keeps this interchangeable with the library.
        """
        corpus_size = len(self._doc_freqs)
        self._idf = {}
        if corpus_size == 0:
            self._idf_dirty = False
            return

        negative_terms = []
        idf_sum = 0.0
        for term, frequency in self._doc_freq.items():
            idf = math.log(corpus_size - frequency + 0.5) - math.log(frequency + 0.5)
            self._idf[term] = idf
            idf_sum += idf
            if idf < 0:
                negative_terms.append(term)

        average_idf = idf_sum / len(self._idf) if self._idf else 0.0
        floor = self.epsilon * average_idf
        for term in negative_terms:
            self._idf[term] = floor
        self._idf_dirty = False

    def scores(self, query_tokens: list[str]) -> dict[str, float]:
        """BM25 score per document id. Documents matching no query term score 0.0 and are
        included, so callers can filter on `> 0` exactly as they did before."""
        if self._idf_dirty:
            self._rebuild_idf()
        if not self._doc_freqs:
            return {}

        average_length = self.average_length
        results = {doc_id: 0.0 for doc_id in self._doc_freqs}
        for term in query_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_id, frequencies in self._doc_freqs.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                length_norm = 1 - self.b + self.b * (self._doc_len[doc_id] / average_length)
                results[doc_id] += (
                    idf * term_frequency * (self.k1 + 1) / (term_frequency + self.k1 * length_norm)
                )
        return results
