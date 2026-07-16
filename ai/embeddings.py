from __future__ import annotations

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    _EMB_AVAILABLE = True
except Exception:
    # dependencies not installed; avoid crashing at import time
    SentenceTransformer = None
    np = None
    faiss = None
    _EMB_AVAILABLE = False

_model = None


def _require_embeddings():
    if not _EMB_AVAILABLE:
        raise RuntimeError(
            "Missing Python packages for embeddings. Install requirements: pip install sentence-transformers faiss-cpu numpy"
        )


def get_model(name: str = "all-MiniLM-L6-v2"):
    _require_embeddings()
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def embed_texts(texts, model=None):
    if model is None:
        model = get_model()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def search_index(index, query_embedding, k=5):
    import numpy as np
    q = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D[0], I[0]


def build_index_from_texts(texts, model=None):
    """Create embeddings + FAISS index from list of texts. Returns (index, embeddings, metadata)

    metadata is a list of dicts with 'id' and 'text'.
    """
    _require_embeddings()
    import numpy as _np
    if model is None:
        model = get_model()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    emb = emb.astype('float32')
    faiss.normalize_L2(emb)
    index = build_faiss_index(emb)
    metadata = [{'id': i, 'text': texts[i]} for i in range(len(texts))]
    return index, emb, metadata


def save_faiss_index(index, metadata, path_prefix: str):
    """Save index and metadata to disk. Writes index to {path_prefix}.index and metadata to {path_prefix}.meta.npy"""
    _require_embeddings()
    faiss.write_index(index, path_prefix + '.index')
    # save metadata as numpy object array
    import numpy as _np
    _np.save(path_prefix + '.meta.npy', _np.array(metadata, dtype=object), allow_pickle=True)


def load_faiss_index(path_prefix: str):
    """Load index and metadata from disk. Returns (index, metadata)."""
    _require_embeddings()
    import numpy as _np
    index = faiss.read_index(path_prefix + '.index')
    metadata = _np.load(path_prefix + '.meta.npy', allow_pickle=True).tolist()
    return index, metadata
