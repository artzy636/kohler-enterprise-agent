"""Query the per-domain FAISS indexes built by build_index.py."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

ROOT_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT_DIR / "indexes"

_model: SentenceTransformer | None = None
_index_cache: dict[str, tuple[faiss.Index, list[dict]]] = {}


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_domain(domain: str) -> tuple[faiss.Index, list[dict]]:
    if domain in _index_cache:
        return _index_cache[domain]

    domain_dir = INDEX_DIR / domain
    index_path = domain_dir / "index.faiss"
    chunks_path = domain_dir / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"No index found for domain '{domain}' in {domain_dir}. "
            "Run src/build_index.py first."
        )

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    _index_cache[domain] = (index, records)
    return index, records


def retrieve(query: str, domain: str, k: int = 4) -> list[dict]:
    """Return the top-k chunks for `query` within `domain`.

    Each result is a dict with "text" (the chunk content) and "source"
    (the source .md filename it came from), ordered by relevance.
    """
    index, records = _load_domain(domain)

    model = _get_model()
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    k = min(k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        record = records[idx]
        results.append({
            "text": record["text"],
            "source": record["source"],
            "score": float(score),
        })
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python src/retrieve.py <domain> <query>")
        sys.exit(1)

    domain_arg = sys.argv[1]
    query_arg = " ".join(sys.argv[2:])
    for i, r in enumerate(retrieve(query_arg, domain_arg), start=1):
        print(f"{i}. [{r['source']}] (score={r['score']:.3f})\n   {r['text'][:200]}...")
