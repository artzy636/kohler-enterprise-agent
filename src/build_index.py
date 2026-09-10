"""Build a per-domain FAISS index over the markdown knowledge base.

For each domain in data/<domain>/, every .md file is split into overlapping
word chunks, embedded with sentence-transformers, and written to
indexes/<domain>/ as a FAISS index plus a chunk-to-source mapping.

Usage:
    python src/build_index.py
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOMAINS = ["hr", "finance", "legal", "support"]
MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_WORDS = 200
OVERLAP_WORDS = 50
CHUNK_STRIDE = CHUNK_WORDS - OVERLAP_WORDS

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "indexes"


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, stride: int = CHUNK_STRIDE) -> list[str]:
    """Split text into ~chunk_words-word chunks with overlap between consecutive chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += stride
    return chunks


def load_domain_chunks(domain: str) -> list[dict]:
    """Read every .md file in data/<domain>/ and split it into chunks."""
    domain_dir = DATA_DIR / domain
    records = []
    for md_path in sorted(domain_dir.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        text = re.sub(r"\s+", " ", raw).strip()
        for chunk in chunk_text(text):
            records.append({"text": chunk, "source": md_path.name})
    return records


def build_domain_index(domain: str, model: SentenceTransformer) -> None:
    records = load_domain_chunks(domain)
    if not records:
        print(f"[{domain}] no chunks found, skipping")
        return

    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    domain_dir = INDEX_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(domain_dir / "index.faiss"))
    with open(domain_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[{domain}] indexed {len(records)} chunks from "
          f"{len(set(r['source'] for r in records))} files")


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    for domain in DOMAINS:
        build_domain_index(domain, model)


if __name__ == "__main__":
    main()
