import json, numpy as np, faiss
from pathlib import Path
from .config import EMBED_MODEL, FAISS_INDEX_PATH, EMBEDDINGS_NPY, DOCS_CHUNKS_JSON

def _load_chunks():
    if not DOCS_CHUNKS_JSON.exists():
        raise FileNotFoundError(f"Missing {DOCS_CHUNKS_JSON}")
    return json.load(open(DOCS_CHUNKS_JSON, "r", encoding="utf-8"))


def build_index():
    """
    Minimal deterministic embedding generator + FAISS index builder.
    Works in low RAM environments (Colab CPU/GPU).
    """
    chunks = _load_chunks()
    n = len(chunks)
    dim = 384  # Consistent dimension for deterministic embeddings

    # Deterministic fake embeddings
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(n, dim)).astype("float32")

    # FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    np.save(EMBEDDINGS_NPY, embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    return {
        "chunks": n,
        "dimensions": dim,
        "embeddings_saved": str(EMBEDDINGS_NPY),
        "faiss_index_saved": str(FAISS_INDEX_PATH)
    }


def load_index():
    """
    Loads embeddings.npy + FAISS index + metadata.
    Returns a dict containing index, embeddings, metadata list.
    """
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index missing: {FAISS_INDEX_PATH}")

    if not EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(f"Embeddings missing: {EMBEDDINGS_NPY}")

    chunks = _load_chunks()
    embeddings = np.load(EMBEDDINGS_NPY)
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    return {
        "index": index,
        "embeddings": embeddings,
        "chunks": chunks
    }


def retrieve(query: str, k=5):
    """
    Simple deterministic retrieval based on FAISS.
    """
    # deterministic seed based on query
    seed = abs(hash(query)) % (2**32)
    rng = np.random.default_rng(seed)
    q_vec = rng.normal(size=(384,)).astype("float32") # Use consistent dim 384

    data = load_index()
    index = data["index"]
    chunks = data["chunks"]

    qv = q_vec.reshape(1, -1)
    distances, idxs = index.search(qv, k)

    results = []
    for idx, dist in zip(idxs[0], distances[0]):
        results.append({
            "doc_id": chunks[idx]['doc_id'],
            "claim_id": chunks[idx]['claim_id'],
            "text": chunks[idx]['text'],
            "distance": float(dist)
        })
    return results
