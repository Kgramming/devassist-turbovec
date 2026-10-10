# FAISS vs TurboVec — DevAssist vector-store comparison

This document describes the only architectural difference between the
original DevAssist Chatbot (`Kgramming/devassist-chatbot`) and this
TurboVec edition (`Kgramming/devassist-turbovec`). Everything outside
`backend/app/rag/vectorstore.py` (plus dependency/port/config updates)
is identical.

## The original: FAISS `IndexFlatIP`

- `faiss-cpu==1.15.1`, `faiss.IndexFlatIP(384)` — exact brute-force
  inner-product search over float32 vectors.
- Vectors are L2-normalized at embedding time, so inner product ==
  cosine similarity; scores feed the `SOURCE_SCORE_THRESHOLD` (0.4)
  citation filter unchanged.
- Metadata was kept in a Python list aligned 1:1 with FAISS index
  positions (`index.search` returns positional indices).
- FAISS has no deletion: `remove_document` rebuilt a fresh index from
  the surviving per-document vectors.
- Adds were incremental (`index.add`); the index held full float32
  vectors (~1.5 KB per 384-dim chunk).

## The new implementation: TurboVec `IdMapIndex`

- `turbovec==1.1.2`, `IdMapIndex(dim=384, bit_width=4)` — 4-bit
  TurboQuant quantized index, pure CPU (AVX2/AVX-512 SIMD), zero
  training. Verified against the installed package's actual API
  (signatures, return shapes, and edge cases probed directly).
- `add_with_ids(vectors, ids)` is incremental; `search(queries, k,
  allowlist=...)` returns `(scores, ids)` with scores as inner-product
  similarity sorted **descending** — verified to track the true
  inner products within ~±0.03 (quantization noise), so the existing
  0.4 citation threshold keeps working without retuning.
- Chunk IDs come from a monotonic counter that is never reused, so a
  deleted document's IDs can never collide with a later upload's.
  Metadata lives in a `chunk_id → metadata` dict instead of a
  positional list.
- `remove(id)` is native per-ID deletion — no rebuild needed for the
  delete path. `remove_document` deletes each chunk ID and drops the
  document's metadata.
- A `_rebuild()` from **all** retained per-document embeddings is kept
  as a safety net (and is regression-tested). Note: an early design
  sketch rebuilt from only newly added vectors, which would silently
  discard earlier documents — this implementation deliberately does
  not do that.

## Search and filtering behavior

| Aspect | FAISS edition | TurboVec edition |
|---|---|---|
| `search(q, top_k=3)` return | `[{metadata, score}]`, cosine | `[{metadata, score}]`, cosine-equivalent |
| Empty index | `[]` | `[]` |
| `top_k` larger than index | capped | capped |
| Filename filter | not supported | `filename_filter=` → native `allowlist` (unknown name → `[]`, never an error) |
| Score semantics | exact cosine | approximate cosine (±0.03) |
| Dimension mismatch | `DimensionMismatchError` | `DimensionMismatchError` (same) |
| Duplicate `document_id` | `ValueError` | `ValueError` (same) |

The public `VectorStore` interface (`dim`, `count`, `add_document`,
`remove_document`, `search`, `clear`) is unchanged, so `ingestion.py`,
`chat.py`, and all existing tests work unmodified.

## Memory / storage trade-offs

- FAISS `IndexFlatIP`: ~1.5 KB per 384-dim float32 chunk, exact scores.
- TurboVec 4-bit: ~192 bytes per chunk (8× smaller), approximate
  scores. At DevAssist's scale (tens of documents, hundreds of chunks)
  both are negligible in absolute terms; the difference matters if the
  corpus ever grows 100–1000×.
- Both editions are **in-memory and transient**: restarting the backend
  clears all documents, exactly as before. TurboVec offers
  `.write()`/`.load()` serialization if persistence is ever wanted;
  it is not wired up here.

## Measured observations (2026-10-10, this machine)

- Index construction: instant — no training step (verified: `add_document`
  returns immediately for small batches).
- Correctness: 15 new regression tests + all 72 pre-existing backend
  tests + 101 frontend tests pass. Live check: 2 documents uploaded
  (4 + 5 chunks), questions retrieved chunks from the correct document,
  sources carried filename/chunk_index/text, WebSocket streamed
  status → sources → tokens → done.
- **Not benchmarked:** no head-to-head FAISS-vs-TurboVec latency or
  recall measurement was performed, so no speed/accuracy claim is made
  here. The 4-bit quantization noise (±0.03 on cosine) is the only
  quantified approximation cost.

## Limitations

- Scores are approximate; a chunk scoring 0.39 vs 0.41 could flip across
  the 0.4 citation threshold where FAISS would not. In practice the
  threshold still behaves the same on real queries.
- `filename_filter` is implemented and tested at the store level but is
  not yet exposed through the chat API (the UI asks across all
  documents, as before).
- In-memory only — same as the original.
