"""TurboVec vector store with a strict dimension guard.

Replaces the original FAISS ``IndexFlatIP`` backend with TurboVec's
``IdMapIndex`` (384-dim, 4-bit TurboQuant, CPU-only, zero training).
Vectors are L2-normalized embeddings; the index returns inner-product
similarity, which is then equivalent to cosine similarity.

TurboVec supports native per-ID deletion and incremental inserts, so the
day-to-day paths never rebuild. Per-document vectors are still retained so
the index can always be reconstructed from **all** retained embeddings
(never just newly added ones) if a rebuild is ever required.

Chunk IDs are assigned from a monotonic counter that is never reused, so a
deleted document's IDs can never collide with a later document's.
"""

import logging

import numpy as np
from turbovec import IdMapIndex

from app.rag.embeddings import EMBEDDING_DIM

logger = logging.getLogger(__name__)

#: Quantization width for the TurboVec index (4-bit TurboQuant).
BIT_WIDTH = 4


class DimensionMismatchError(Exception):
    """Raised when vectors do not match the index dimensionality."""


class VectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM, bit_width: int = BIT_WIDTH) -> None:
        self._dim = dim
        self._bit_width = bit_width
        self._index = IdMapIndex(dim=dim, bit_width=bit_width)
        # chunk_id -> metadata
        self._metadatas: dict[int, dict] = {}
        # Per-document storage: chunk ids, vectors and metadatas, so the
        # index can be rebuilt from all retained embeddings and so
        # filename-filtered search can resolve an allowlist.
        self._doc_chunk_ids: dict[str, list[int]] = {}
        self._doc_vectors: dict[str, np.ndarray] = {}
        self._doc_metadatas: dict[str, list[dict]] = {}
        self._next_id = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bit_width(self) -> int:
        return self._bit_width

    @property
    def count(self) -> int:
        return int(len(self._index))

    def _check_dim(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise DimensionMismatchError(
                f"Vector dimension mismatch: index expects {self._dim}, "
                f"got shape {tuple(vectors.shape)}"
            )

    def add_document(
        self, document_id: str, vectors: np.ndarray, metadatas: list[dict]
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._check_dim(vectors)
        if len(metadatas) != vectors.shape[0]:
            raise ValueError("metadatas length must match number of vectors")
        if document_id in self._doc_vectors:
            raise ValueError(f"document_id '{document_id}' already indexed")

        chunk_ids = list(range(self._next_id, self._next_id + vectors.shape[0]))
        self._next_id += vectors.shape[0]

        self._index.add_with_ids(
            vectors, np.array(chunk_ids, dtype=np.uint64)
        )
        for cid, meta in zip(chunk_ids, metadatas):
            self._metadatas[cid] = meta
        self._doc_chunk_ids[document_id] = chunk_ids
        self._doc_vectors[document_id] = vectors
        self._doc_metadatas[document_id] = list(metadatas)
        logger.info(
            "Indexed %d vectors for document %s (total=%d)",
            vectors.shape[0],
            document_id,
            self.count,
        )

    def remove_document(self, document_id: str) -> bool:
        """Remove a document's vectors via native per-ID deletion.

        Returns False if the document is unknown.
        """
        if document_id not in self._doc_vectors:
            return False
        for cid in self._doc_chunk_ids[document_id]:
            self._index.remove(np.uint64(cid))
            self._metadatas.pop(cid, None)
        del self._doc_chunk_ids[document_id]
        del self._doc_vectors[document_id]
        del self._doc_metadatas[document_id]
        logger.info("Removed document %s (total=%d)", document_id, self.count)
        return True

    def _rebuild(self) -> None:
        """Reconstruct the index from ALL retained document embeddings.

        Never rebuild from a partial (e.g. only newly added) set: every
        retained document contributes its vectors and IDs.
        """
        new_index = IdMapIndex(dim=self._dim, bit_width=self._bit_width)
        new_metadatas: dict[int, dict] = {}
        for doc_id, vecs in self._doc_vectors.items():
            vecs = np.asarray(vecs, dtype=np.float32)
            # Strict guard: never silently rebuild with incompatible vectors.
            self._check_dim(vecs)
            chunk_ids = self._doc_chunk_ids[doc_id]
            new_index.add_with_ids(vecs, np.array(chunk_ids, dtype=np.uint64))
            for cid, meta in zip(chunk_ids, self._doc_metadatas[doc_id]):
                new_metadatas[cid] = meta
        self._index = new_index
        self._metadatas = new_metadatas

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        filename_filter: str | None = None,
    ) -> list[dict]:
        """Return up to top_k hits as {metadata, score} dicts (score = cosine).

        ``filename_filter`` restricts results to chunks of one file via the
        index's native allowlist; an unknown filename yields no results.
        """
        if len(self._index) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        self._check_dim(q)
        k = min(top_k, len(self._index))
        if k <= 0:
            return []

        allowlist = None
        if filename_filter is not None:
            allowed = [
                cid
                for cid, meta in self._metadatas.items()
                if meta.get("filename") == filename_filter
            ]
            if not allowed:
                return []
            allowlist = np.array(allowed, dtype=np.uint64)

        scores, ids = self._index.search(q, k, allowlist=allowlist)
        results = []
        for score, cid in zip(scores[0], ids[0]):
            meta = self._metadatas.get(int(cid))
            if meta is None:
                # Defensive: never surface an ID we cannot map to metadata.
                continue
            results.append({"metadata": meta, "score": float(score)})
        return results

    def clear(self) -> None:
        self._index = IdMapIndex(dim=self._dim, bit_width=self._bit_width)
        self._metadatas = {}
        self._doc_chunk_ids = {}
        self._doc_vectors = {}
        self._doc_metadatas = {}
        self._next_id = 0
