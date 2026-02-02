from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils import load_documents


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


def simple_chunk(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Very simple character-based chunking (good enough for a demo).
    """
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


class RAGIndex:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embed_model_name)
        self.index = None
        self.chunks: List[Chunk] = []

    def build(self, docs_dir: str = "docs", chunk_size: int = 900, overlap: int = 150):
        docs = load_documents(docs_dir)
        all_chunks: List[Chunk] = []
        for source, text in docs:
            pieces = simple_chunk(text, chunk_size=chunk_size, overlap=overlap)
            for i, piece in enumerate(pieces):
                all_chunks.append(Chunk(source=source, chunk_id=i, text=piece))

        if not all_chunks:
            raise RuntimeError("No readable documents found in /docs. Add PDF/DOCX/TXT and try again.")

        texts = [c.text for c in all_chunks]
        emb = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = emb.astype("float32")

        dim = emb.shape[1]
        idx = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(emb)
        idx.add(emb)

        self.index = idx
        self.chunks = all_chunks

    def save(self, data_dir: str = "data"):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Index not built.")
        faiss.write_index(self.index, str(Path(data_dir) / "faiss.index"))
        meta = [{"source": c.source, "chunk_id": c.chunk_id, "text": c.text} for c in self.chunks]
        (Path(data_dir) / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, data_dir: str = "data"):
        idx_path = Path(data_dir) / "faiss.index"
        chunks_path = Path(data_dir) / "chunks.json"
        if not idx_path.exists() or not chunks_path.exists():
            raise RuntimeError("No saved index found. Click 'Build / Rebuild Index' first.")
        self.index = faiss.read_index(str(idx_path))
        meta = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**m) for m in meta]

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not loaded/built.")
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, top_k)
        results = []
        for i, idx in enumerate(ids[0]):
            if idx < 0:
                continue
            results.append((self.chunks[int(idx)], float(scores[0][i])))
        return results
