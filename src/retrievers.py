"""Retrievers: BM25, TF-IDF, and dense embeddings with a deterministic fallback.

Three retrievers share one interface:

    retriever.search(query: str, k: int) -> list[str]   # top-k doc ids

Design notes
------------
* BM25 uses ``rank_bm25.BM25Okapi`` with a simple lowercase ``[a-z0-9_]+``
  tokenizer that keeps error codes (``err_gateway_timeout``) and numbers
  (``429``) as single tokens.
* TF-IDF uses scikit-learn's ``TfidfVectorizer`` (lowercased, default word
  token pattern) with L2-normalized vectors; similarity is cosine via the
  dot product.
* Dense retrieval embeds documents and queries with ``BAAI/bge-small-en-v1.5``
  (384-dim) through ``fastembed`` (local ONNX inference, no network at query
  time, no paid API). Passages are embedded with ``embed()`` and queries with
  ``query_embed()`` (the BGE query prefix). If fastembed or the model file is
  unavailable, a deterministic scikit-learn LSA fallback
  (TF-IDF -> TruncatedSVD, fixed random_state) is used so the benchmark
  always runs; ``backend`` records which path was taken.
* Ranking ties are broken by corpus order (stable sort), so every run is
  reproducible.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9_]+")

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384
LSA_COMPONENTS = 64


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer shared by BM25 and reporting code."""
    return TOKEN_RE.findall(text.lower())


def _rank_ids(scores: np.ndarray, doc_ids: Sequence[str]) -> List[str]:
    """Descending score order; ties broken by corpus order (stable sort)."""
    order = np.argsort(-scores, kind="stable")
    return [doc_ids[i] for i in order]


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (cosine similarity via dot product)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class BM25Retriever:
    """Sparse lexical retrieval: Okapi BM25 (rank_bm25)."""

    name = "bm25"

    def __init__(self, doc_ids: Sequence[str], documents: Sequence[str]):
        from rank_bm25 import BM25Okapi

        self.doc_ids = list(doc_ids)
        self.corpus_tokens = [tokenize(d) for d in documents]
        self.model = BM25Okapi(self.corpus_tokens)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.model.get_scores(tokenize(query)), dtype=np.float64)

    def search(self, query: str, k: int) -> List[str]:
        return _rank_ids(self.scores(query), self.doc_ids)[:k]


class TfidfRetriever:
    """Sparse lexical retrieval: TF-IDF vectors + cosine similarity."""

    name = "tfidf"

    def __init__(self, doc_ids: Sequence[str], documents: Sequence[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.doc_ids = list(doc_ids)
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.matrix = self.vectorizer.fit_transform(documents)

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return np.asarray((self.matrix @ q.T).todense()).ravel()

    def search(self, query: str, k: int) -> List[str]:
        return _rank_ids(self.scores(query), self.doc_ids)[:k]


class DenseRetriever:
    """Dense semantic retrieval: sentence embeddings + cosine similarity.

    Primary backend: fastembed with BAAI/bge-small-en-v1.5 (deterministic
    local ONNX inference). Fallback backend: scikit-learn LSA
    (TF-IDF -> TruncatedSVD with random_state=42), which keeps the whole
    benchmark runnable offline and dependency-light at the cost of weaker
    semantics.
    """

    name = "dense"

    def __init__(self, doc_ids: Sequence[str], documents: Sequence[str]):
        self.doc_ids = list(doc_ids)
        self.backend = ""
        self._query_encoder = None
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=DENSE_MODEL_NAME)
            doc_vecs = np.array(list(self._model.embed(documents)), dtype=np.float32)
            self._query_encoder = self._model.query_embed
            self.backend = f"fastembed:{DENSE_MODEL_NAME} ({DENSE_DIM}-d, cosine)"
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[dense] fastembed unavailable ({type(exc).__name__}); "
                  f"falling back to LSA (TF-IDF -> TruncatedSVD, seed=42)")
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            self._vectorizer = TfidfVectorizer(lowercase=True)
            matrix = self._vectorizer.fit_transform(documents)
            n_comp = min(LSA_COMPONENTS, matrix.shape[1] - 1, len(doc_ids) - 1)
            self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
            doc_vecs = self._svd.fit_transform(matrix)
            self.backend = f"lsa-fallback:TfidfVectorizer+TruncatedSVD({n_comp})"
        self.doc_vecs = _l2_normalize(np.asarray(doc_vecs, dtype=np.float32))

    def _embed_query(self, query: str) -> np.ndarray:
        if self._query_encoder is not None:
            vec = np.array(list(self._query_encoder([query])), dtype=np.float32)
        else:
            vec = self._svd.transform(self._vectorizer.transform([query])).astype(np.float32)
        return _l2_normalize(vec)[0]

    def scores(self, query: str) -> np.ndarray:
        q = self._embed_query(query)
        return self.doc_vecs @ q

    def search(self, query: str, k: int) -> List[str]:
        return _rank_ids(self.scores(query), self.doc_ids)[:k]


def build_retrievers(doc_ids: Sequence[str], documents: Sequence[str],
                     skip_dense: bool = False) -> Dict[str, object]:
    """Instantiate all retrievers; returns {name: retriever}."""
    retrievers: Dict[str, object] = {
        "bm25": BM25Retriever(doc_ids, documents),
        "tfidf": TfidfRetriever(doc_ids, documents),
    }
    if not skip_dense:
        retrievers["dense"] = DenseRetriever(doc_ids, documents)
    return retrievers
