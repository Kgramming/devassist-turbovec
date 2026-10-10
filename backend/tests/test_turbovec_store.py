"""TurboVec vector store: index integrity, filtering, deletion, rebuild.

Regression coverage for the FAISS -> TurboVec migration. Uses synthetic
L2-normalized embeddings (fast, deterministic) so these tests never need
the sentence-transformers model download.
"""

import numpy as np
import pytest

from app.rag.embeddings import EMBEDDING_DIM
from app.rag.vectorstore import BIT_WIDTH, DimensionMismatchError, VectorStore

DIM = EMBEDDING_DIM


def _vecs(n, seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(n, DIM).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _metas(doc_id, filename, n):
    return [
        {
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "source": "upload",
            "text": f"{filename} chunk {i}",
        }
        for i in range(n)
    ]


def _query_like(vecs, idx=0):
    # A query near vecs[idx]: must retrieve its own chunk top-1.
    q = vecs[idx] + np.random.RandomState(99).randn(DIM).astype(np.float32) * 1e-3
    return q / np.linalg.norm(q)


def test_index_dim_and_bit_width():
    store = VectorStore(dim=DIM)
    assert store.dim == DIM
    assert store.bit_width == BIT_WIDTH == 4
    assert store.count == 0


def test_add_and_count():
    store = VectorStore(dim=DIM)
    store.add_document("d1", _vecs(4, 1), _metas("d1", "a.txt", 4))
    assert store.count == 4
    store.add_document("d2", _vecs(3, 2), _metas("d2", "b.txt", 3))
    assert store.count == 7


def test_duplicate_document_rejected():
    store = VectorStore(dim=DIM)
    store.add_document("d1", _vecs(2, 1), _metas("d1", "a.txt", 2))
    with pytest.raises(ValueError, match="already indexed"):
        store.add_document("d1", _vecs(2, 1), _metas("d1", "a.txt", 2))
    assert store.count == 2  # failed add left the index untouched


def test_dimension_mismatch_rejected():
    store = VectorStore(dim=DIM)
    with pytest.raises(DimensionMismatchError):
        store.add_document("d1", np.zeros((2, 128), dtype=np.float32), _metas("d1", "a.txt", 2))
    assert store.count == 0


def test_single_document_retrieval():
    store = VectorStore(dim=DIM)
    vecs = _vecs(5, 3)
    store.add_document("d1", vecs, _metas("d1", "a.txt", 5))
    hits = store.search(_query_like(vecs, 2), top_k=3)
    assert len(hits) == 3
    # nearest chunk is its own vector
    assert hits[0]["metadata"]["chunk_index"] == 2
    assert hits[0]["metadata"]["filename"] == "a.txt"
    assert hits[0]["score"] > 0.9  # near-identical normalized vectors
    # scores are cosine-like, descending
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.05 for s in scores)


def test_multi_document_retrieval():
    store = VectorStore(dim=DIM)
    va = _vecs(4, 10)
    vb = _vecs(4, 20)
    store.add_document("da", va, _metas("da", "a.txt", 4))
    store.add_document("db", vb, _metas("db", "b.txt", 4))
    # query near a chunk of doc B -> top hit is from b.txt
    hits = store.search(_query_like(vb, 1), top_k=4)
    assert hits[0]["metadata"]["filename"] == "b.txt"
    assert hits[0]["metadata"]["document_id"] == "db"
    # both documents represented across a wider net
    hits = store.search(_query_like(va, 0), top_k=8)
    files = {h["metadata"]["filename"] for h in hits}
    assert files == {"a.txt", "b.txt"}


def test_repeated_uploads_in_separate_batches():
    """Uploads in separate batches must accumulate, never overwrite."""
    store = VectorStore(dim=DIM)
    v1 = _vecs(3, 30)
    v2 = _vecs(3, 31)
    store.add_document("d1", v1, _metas("d1", "one.txt", 3))
    assert store.count == 3
    store.add_document("d2", v2, _metas("d2", "two.txt", 3))
    assert store.count == 6
    # both documents still searchable
    h1 = store.search(_query_like(v1, 0), top_k=1)
    h2 = store.search(_query_like(v2, 0), top_k=1)
    assert h1[0]["metadata"]["filename"] == "one.txt"
    assert h2[0]["metadata"]["filename"] == "two.txt"


def test_rebuild_keeps_all_documents():
    """A rebuild must reconstruct from ALL retained embeddings.

    (The naive sketch rebuilds from only newly added vectors, silently
    dropping earlier documents. This test pins the correct behavior.)
    """
    store = VectorStore(dim=DIM)
    v1 = _vecs(3, 40)
    v2 = _vecs(2, 41)
    store.add_document("d1", v1, _metas("d1", "one.txt", 3))
    store.add_document("d2", v2, _metas("d2", "two.txt", 2))
    before = {(h["metadata"]["filename"], h["metadata"]["chunk_index"]) for h in store.search(_query_like(v1, 0), top_k=5)}
    store._rebuild()
    assert store.count == 5
    after = {(h["metadata"]["filename"], h["metadata"]["chunk_index"]) for h in store.search(_query_like(v1, 0), top_k=5)}
    assert before == after
    # document 2 still retrievable after the rebuild
    h = store.search(_query_like(v2, 1), top_k=1)
    assert h[0]["metadata"]["filename"] == "two.txt"
    assert h[0]["metadata"]["chunk_index"] == 1


def test_filename_filter():
    store = VectorStore(dim=DIM)
    va = _vecs(4, 50)
    vb = _vecs(4, 51)
    store.add_document("da", va, _metas("da", "a.txt", 4))
    store.add_document("db", vb, _metas("db", "b.txt", 4))
    hits = store.search(_query_like(va, 0), top_k=4, filename_filter="a.txt")
    assert 1 <= len(hits) <= 4
    assert all(h["metadata"]["filename"] == "a.txt" for h in hits)
    # unknown filename -> no results, not an error
    assert store.search(_query_like(va, 0), top_k=4, filename_filter="nope.txt") == []


def test_empty_index_and_no_match():
    store = VectorStore(dim=DIM)
    q = _vecs(1, 60)[0]
    assert store.search(q, top_k=3) == []
    assert store.search(q, top_k=0) == []


def test_remove_document_and_metadata_consistency():
    store = VectorStore(dim=DIM)
    va = _vecs(3, 70)
    vb = _vecs(3, 71)
    store.add_document("da", va, _metas("da", "a.txt", 3))
    store.add_document("db", vb, _metas("db", "b.txt", 3))
    assert store.remove_document("da") is True
    assert store.count == 3
    assert store.remove_document("da") is False  # already gone
    # removed doc's chunks are no longer returned
    hits = store.search(_query_like(va, 0), top_k=6)
    assert all(h["metadata"]["filename"] == "b.txt" for h in hits)
    # surviving doc still fully searchable
    h = store.search(_query_like(vb, 2), top_k=1)
    assert h[0]["metadata"]["chunk_index"] == 2


def test_ids_never_reused_after_delete():
    """Chunk IDs are monotonic: deleting then re-adding cannot collide."""
    store = VectorStore(dim=DIM)
    store.add_document("d1", _vecs(2, 80), _metas("d1", "a.txt", 2))
    ids_before = set(store._metadatas.keys())
    store.remove_document("d1")
    store.add_document("d2", _vecs(2, 81), _metas("d2", "b.txt", 2))
    ids_after = set(store._metadatas.keys())
    assert ids_before.isdisjoint(ids_after)
    assert store.count == 2


def test_unknown_ids_are_skipped_defensively():
    """If an ID ever lacks metadata, search must skip it, not crash."""
    store = VectorStore(dim=DIM)
    vecs = _vecs(3, 90)
    store.add_document("d1", vecs, _metas("d1", "a.txt", 3))
    # simulate an inconsistent state: drop one metadata entry
    orphan = next(iter(store._metadatas.keys()))
    del store._metadatas[orphan]
    hits = store.search(_query_like(vecs, 0), top_k=3)
    assert all("metadata" in h and "score" in h for h in hits)


def test_clear_resets_everything():
    store = VectorStore(dim=DIM)
    store.add_document("d1", _vecs(3, 100), _metas("d1", "a.txt", 3))
    store.clear()
    assert store.count == 0
    assert store.search(_vecs(1, 101)[0], top_k=3) == []
    # IDs restart cleanly after clear
    store.add_document("d2", _vecs(2, 102), _metas("d2", "b.txt", 2))
    assert store.count == 2
    assert set(store._metadatas.keys()) == {0, 1}


def test_query_dim_mismatch_rejected():
    store = VectorStore(dim=DIM)
    store.add_document("d1", _vecs(2, 110), _metas("d1", "a.txt", 2))
    with pytest.raises(DimensionMismatchError):
        store.search(np.zeros(128, dtype=np.float32), top_k=3)
